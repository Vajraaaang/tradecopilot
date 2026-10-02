"""Small, explicitly retrospective Jev comparisons against the same held-out examples."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tradecopilot.forecast.baselines import load_model, predict_model
from tradecopilot.forecast.contracts import DatasetManifest, ForecastExample, content_hash
from tradecopilot.forecast.dataset import _verify
from tradecopilot.forecast.evaluation import split_examples
from tradecopilot.forecast.experiment import (
    REPORT_VERSION,
    _cases,
    _json,
    _model_summary,
    _write_bundle,
    load_report,
    source_provenance,
)
from tradecopilot.forecast.jev import JevForecaster

SELECTION_VERSION = "chronological-quantile-midpoints-v1"


def select_cohort(examples: Sequence[ForecastExample], count: int = 10) -> list[ForecastExample]:
    if isinstance(count, bool) or not 1 <= count <= 10 or len(examples) < count:
        raise ValueError("select 1-10 distinct cases from a sufficiently sized holdout")
    rows = sorted(examples, key=lambda row: (row.as_of, row.symbol, row.example_id))
    if len({row.example_id for row in rows}) != len(rows):
        raise ValueError("duplicate cohort examples")
    return [rows[((2 * index + 1) * len(rows)) // (2 * count)] for index in range(count)]


def run_retrospective(
    manifest: DatasetManifest,
    examples: Sequence[ForecastExample],
    base_report: Path,
    out_dir: Path,
    forecaster: JevForecaster,
    *,
    count: int = 10,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    _verify(manifest, list(examples))
    if manifest.provenance != "historical":
        raise ValueError("retrospective stock comparison requires explicit historical provenance")
    base = load_report(base_report)
    if base["dataset"]["dataset_id"] != manifest.dataset_id or not base.get("data_source"):
        raise ValueError("baseline report must match the historical dataset and source")
    if out_dir.exists():
        raise FileExistsError("choose a new output directory; retrospective selections cannot be reset")
    split = split_examples(examples)
    selected = select_cohort(split["test"], count)
    selection = {
        "schema_version": SELECTION_VERSION,
        "dataset_id": manifest.dataset_id,
        "config_id": manifest.config.config_id,
        "count": count,
        "example_ids": [row.example_id for row in selected],
        "selection_policy": "Evenly spaced chronological midpoint indices; labels are not used.",
    }
    inputs = [
        row.model_copy(
            update={
                "label": None,
                "target_price": None,
                "label_observed_at": None,
                "target_return_bps": None,
                "exclusion_reason": "outcome_not_sent_to_model",
            }
        )
        for row in selected
    ]
    out_dir.mkdir(parents=True)
    # Freeze selection and label-free inputs before any paid request; preserve even interrupted runs.
    (out_dir / "selection.json").write_text(_json(selection), encoding="utf-8")
    (out_dir / "inputs.jsonl").write_text("".join(row.model_dump_json() + "\n" for row in inputs), encoding="utf-8")
    names = ("prior", "momentum", "logistic", "logistic-calibrated")
    models = {name: load_model(base_report.parent / (name + ".json")) for name in names}
    for model in models.values():
        if set(model.training_ids) != {row.example_id for row in split["train"]}:
            raise ValueError("baseline training provenance does not match the frozen split")
    groups = {
        name: predict_model(model, selected, manifest.dataset_id, manifest.config) for name, model in models.items()
    }
    jev_predictions = []
    for index, row in enumerate(inputs):
        if index:
            sleep(31)
        prediction = forecaster.predict(row, manifest.config, manifest.dataset_id, prospective=False)
        jev_predictions.append(prediction)
        with (out_dir / "jev-attempts.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(prediction.model_dump_json() + "\n")
            handle.flush()
        print(
            json.dumps(
                {
                    "case": index + 1,
                    "cases": count,
                    "symbol": row.symbol,
                    "as_of": row.as_of.isoformat(),
                    "status": prediction.status,
                    "reason": prediction.reason,
                    "input_tokens": prediction.input_tokens,
                    "estimated_cost_usd": prediction.estimated_cost_usd,
                }
            ),
            flush=True,
        )
    groups["jev-retrospective"] = jev_predictions
    report: dict[str, Any] = {
        "schema_version": REPORT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "evaluation_kind": "retrospective_historical_sample",
        "title": "Jev historical comparison sample",
        "dataset": manifest.model_dump(mode="json"),
        "data_source": base["data_source"],
        "source": source_provenance(),
        "selection": selection,
        "experiment_id": content_hash(
            {
                "dataset_id": manifest.dataset_id,
                "selection": selection,
                "model_ids": {name: model.model_id for name, model in models.items()},
            }
        ),
        "split": {
            "test_sample": {"examples": count, "sessions": sorted({row.session_date.isoformat() for row in selected})}
        },
        "models": [
            _model_summary(name, predictions, selected, models.get(name)) for name, predictions in groups.items()
        ],
        "cases": _cases(selected, groups),
        "limitations": [
            "Jev calls were made after these historical outcomes occurred; this is retrospective inference.",
            "Historical dates or patterns may appear in model training data; contamination cannot be ruled out.",
            "A maximum ten-case sample is too small to estimate dependable model accuracy or profitability.",
            "Every baseline is scored on the identical frozen cohort, including Jev errors and abstentions.",
            "Historical source availability is reconstructed at minute-bar end; real network receipt was not observed.",
            "Thresholds and the selection rule were fixed before inspecting the test results.",
        ],
    }
    files = {
        "selection.json": _json(selection),
        "historical-source.json": _json(base["data_source"]),
        "inputs.jsonl": "".join(row.model_dump_json() + "\n" for row in inputs),
        "examples.jsonl": "".join(row.model_dump_json() + "\n" for row in selected),
        "predictions.jsonl": "".join(p.model_dump_json() + "\n" for values in groups.values() for p in values),
    }
    # Complete an immutable verified bundle beside the persistent attempt/selection log.
    return _write_bundle(out_dir / "report", report, files)
