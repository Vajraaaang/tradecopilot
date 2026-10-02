"""Reproducible offline experiments and immutable, integrity-checked result bundles."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from tradecopilot.forecast.baselines import ModelArtifact, calibrate_model, fit_model, predict_model
from tradecopilot.forecast.contracts import (
    DatasetManifest,
    ForecastConfig,
    ForecastExample,
    ForecastPrediction,
    Observation,
    content_hash,
)
from tradecopilot.forecast.dataset import _verify
from tradecopilot.forecast.evaluation import score_predictions, split_examples
from tradecopilot.forecast.features import label_example

REPORT_VERSION = "forecast-report-v1"


def source_provenance() -> dict[str, Any]:
    package = Path(__file__).resolve().parents[1]
    sources = {str(path.relative_to(package)): _file_hash(path) for path in sorted(package.rglob("*.py"))}
    return {
        "source_sha256": content_hash(sources),
        "package_version": version("tradecopilot"),
        "python": platform.python_version(),
        "dependencies": {name: version(name) for name in ("numpy", "scikit-learn", "exchange-calendars", "pydantic")},
    }


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _group_metrics(
    examples: Sequence[ForecastExample],
    predictions: Sequence[ForecastPrediction],
) -> dict[str, Any]:
    groups = {}
    for symbol in sorted({row.symbol for row in examples}):
        rows = [row for row in examples if row.symbol == symbol]
        ids = {row.example_id for row in rows}
        groups[symbol] = score_predictions(rows, [p for p in predictions if p.example_id in ids])
    return groups


def _cases(
    examples: Sequence[ForecastExample],
    groups: dict[str, list[ForecastPrediction]],
) -> list[dict[str, Any]]:
    lookup = {key: {p.example_id: p for p in predictions} for key, predictions in groups.items()}
    return [
        {
            "example_id": row.example_id,
            "symbol": row.symbol,
            "as_of": row.as_of.isoformat(),
            "target_time": row.target_time.isoformat(),
            "anchor_price": str(row.anchor_price),
            "target_price": str(row.target_price) if row.target_price is not None else None,
            "actual": row.label,
            "return_bps": row.target_return_bps,
            "exclusion_reason": row.exclusion_reason,
            "predictions": {
                key: values[row.example_id].model_dump(mode="json")
                for key, values in lookup.items()
                if row.example_id in values
            },
        }
        for row in examples
    ]


def _write_bundle(out_dir: Path, report: dict[str, Any], files: dict[str, str]) -> Path:
    out_dir = out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError("choose a new output directory; experiment bundles are immutable")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{out_dir.name}.", dir=out_dir.parent) as temporary:
        staging = Path(temporary)
        for name, contents in files.items():
            (staging / name).write_text(contents, encoding="utf-8")
        report["artifacts"] = {name: _file_hash(staging / name) for name in sorted(files)}
        report["report_id"] = content_hash(report)
        (staging / "report.json").write_text(_json(report), encoding="utf-8")
        os.rename(staging, out_dir)
    return out_dir / "report.json"


def load_report(path: Path) -> dict[str, Any]:
    """Verify a complete bundle before displaying or serving it; never load executable artifacts."""
    path = path.resolve()
    if path.is_dir():
        path /= "report.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(report, dict)
            or report.get("schema_version") != REPORT_VERSION
            or report.get("report_id") != content_hash({k: v for k, v in report.items() if k != "report_id"})
            or not isinstance(report.get("artifacts"), dict)
            or not {"predictions.jsonl", "examples.jsonl"} <= report["artifacts"].keys()
        ):
            raise ValueError("invalid report")
        for name, digest in report["artifacts"].items():
            artifact = path.parent / name
            if (
                Path(name).name != name
                or name in {".", "..", "report.json"}
                or artifact.is_symlink()
                or _file_hash(artifact) != digest
            ):
                raise ValueError("invalid artifact")
    except (ValueError, OSError, TypeError, KeyError):
        raise ValueError("report integrity verification failed") from None
    return report


def run_experiment(
    manifest: DatasetManifest,
    examples: Sequence[ForecastExample],
    out_dir: Path,
    *,
    include_jev_fixture: bool = True,
    historical_source: dict[str, Any] | None = None,
) -> Path:
    _verify(manifest, list(examples))
    if manifest.provenance == "historical":
        if (
            not isinstance(historical_source, dict)
            or historical_source.get("dataset_id") != manifest.dataset_id
            or historical_source.get("observations_hash") != manifest.observations_hash
            or not historical_source.get("availability_assumption")
        ):
            raise ValueError("historical evaluation requires matching source and replay-availability metadata")
    elif historical_source is not None:
        raise ValueError("historical source metadata requires historical provenance")
    if out_dir.exists():
        raise FileExistsError("choose a new output directory; experiment bundles are immutable")
    split = split_examples(examples)
    config = manifest.config
    models = {name: fit_model(name, split["train"], config) for name in ("prior", "momentum", "logistic")}
    models["logistic-calibrated"] = calibrate_model(models["logistic"], split["validation"])
    groups = {name: predict_model(model, split["test"], manifest.dataset_id, config) for name, model in models.items()}
    if include_jev_fixture:
        from tradecopilot.forecast.jev import fixture_predictions

        groups["jev-fixture"] = fixture_predictions(split["test"], manifest.dataset_id, config)
    source = source_provenance()
    report: dict[str, Any] = {
        "schema_version": REPORT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "evaluation_kind": "synthetic_demo" if manifest.provenance == "synthetic" else "historical_benchmark",
        "title": "TradeCopilot forecast evaluation",
        "dataset": manifest.model_dump(mode="json"),
        "source": source,
        "experiment_id": content_hash(
            {
                "dataset_id": manifest.dataset_id,
                "source": source,
                "models": {name: model.model_id for name, model in models.items()},
                "jev_fixture": include_jev_fixture,
                "historical_source": historical_source,
            }
        ),
        "split": {
            name: {
                "examples": len(rows),
                "sessions": sorted({row.session_date.isoformat() for row in rows}),
                "start": min(row.as_of for row in rows).isoformat(),
                "end": max(row.as_of for row in rows).isoformat(),
            }
            for name, rows in split.items()
        },
        "models": [
            _model_summary(name, predictions, split["test"], models.get(name)) for name, predictions in groups.items()
        ],
        "cases": _cases(split["test"], groups),
        "limitations": [
            "Synthetic demo scores verify the pipeline; they do not estimate real-market performance."
            if manifest.provenance == "synthetic"
            else "This chronological holdout is not evidence of profitability or robustness across future regimes.",
            "The Jev fixture is a local contract stub, not Jev inference or a Jev accuracy measurement."
            if include_jev_fixture
            else "No Jev API inference was performed in this run.",
            "Adjacent 15-minute targets overlap. Rows are correlated; uncertainty uses whole sessions only.",
            "Accuracy intervals are omitted when fewer than five test sessions are available.",
            "Flat-band and confidence thresholds were fixed before evaluating the test split.",
        ],
    }
    files = {name + ".json": model.model_dump_json(indent=2) + "\n" for name, model in models.items()}
    files.update(
        {
            "dataset-manifest.json": manifest.model_dump_json(indent=2) + "\n",
            "examples.jsonl": "".join(row.model_dump_json() + "\n" for row in examples),
            "splits.json": _json({name: [row.example_id for row in rows] for name, rows in split.items()}),
            "predictions.jsonl": "".join(p.model_dump_json() + "\n" for values in groups.values() for p in values),
        }
    )
    if historical_source is not None:
        report["data_source"] = historical_source
        report["limitations"].extend(
            [
                "Historical minute closes use reconstructed bar-end availability, "
                "not measured live receipt timestamps.",
                "Previous close is the preceding regular session's last minute close; "
                "sample adjustment type is unspecified.",
                "This small date window and fixed instrument universe do not establish future or regime-wide accuracy.",
            ]
        )
        files["historical-source.json"] = _json(historical_source)
    return _write_bundle(out_dir, report, files)


def _model_summary(
    name: str,
    predictions: list[ForecastPrediction],
    examples: Sequence[ForecastExample],
    model: ModelArtifact | None = None,
) -> dict[str, Any]:
    return {
        "key": name,
        "label": {
            "prior": "Training class prior",
            "momentum": "Momentum baseline",
            "logistic": "Logistic regression",
            "logistic-calibrated": "Logistic + temperature calibration",
            "jev-fixture": "Jev contract fixture (local stub)",
            "jev-live": "Jev prospective pilot",
            "jev-retrospective": "Jev retrospective sample",
        }.get(name, name),
        "model_id": predictions[0].model_id if predictions else None,
        "execution": predictions[0].execution if predictions else None,
        "artifact": name + ".json" if model is not None else None,
        "temperature": model.temperature if model is not None else None,
        "metrics": score_predictions(examples, predictions),
        "by_symbol": _group_metrics(examples, predictions),
    }


def write_pilot_report(
    config: ForecastConfig,
    inputs: Sequence[ForecastExample],
    predictions: Sequence[ForecastPrediction],
    observations: Sequence[Observation],
    out_dir: Path,
    *,
    recorded_predictions: Sequence[ForecastPrediction],
) -> Path:
    from tradecopilot.forecast.jev import FORECAST_PROMPT_VERSION
    from tradecopilot.jev import JEV_MODEL

    if not inputs or len({row.example_id for row in inputs}) != len(inputs):
        raise ValueError("pilot needs distinct, persisted forecast inputs")
    lookup = {row.example_id: row for row in inputs}
    recorded_ids = {prediction.prediction_id for prediction in recorded_predictions}
    for row in inputs:
        if row.config_id != config.config_id or row.provenance != "market" or row.label is not None:
            raise ValueError("pilot inputs must be original, unlabeled market examples matching the config")
    for prediction in predictions:
        original = lookup.get(prediction.example_id)
        if (
            original is None
            or prediction.execution != "live_api"
            or (prediction.status != "error" and not original.as_of <= prediction.generated_at < original.target_time)
            or prediction.prediction_id not in recorded_ids
            or prediction.model_id != JEV_MODEL
            or prediction.model_version != JEV_MODEL
            or prediction.prompt_version != FORECAST_PROMPT_VERSION
            or prediction.reason == "retrospective_api_inference"
            or (prediction.status != "error" and not prediction.request_id)
            or prediction.dataset_id != content_hash({"config_id": config.config_id, "example_id": original.example_id})
        ):
            raise ValueError("pilot predictions must be recorded prospectively for the persisted inputs")
    labeled = [label_example(observations, row, config) for row in inputs]
    groups = {"jev-live": list(predictions)}
    observation_hash = content_hash(sorted(row.observation_id for row in observations))
    report = {
        "schema_version": REPORT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "evaluation_kind": "prospective_pilot",
        "title": "TradeCopilot Jev prospective pilot",
        "dataset": {
            "dataset_id": content_hash({"inputs": list(lookup), "observations": observation_hash}),
            "config": config.model_dump(mode="json"),
            "provenance": "market",
            "observation_count": len(observations),
            "example_count": len(labeled),
            "labeled_count": sum(row.label is not None for row in labeled),
            "sessions": sorted({row.session_date.isoformat() for row in inputs}),
            "observations_hash": observation_hash,
        },
        "source": source_provenance(),
        "experiment_id": content_hash({"inputs": list(lookup), "predictions": [p.prediction_id for p in predictions]}),
        "split": {"pilot": {"examples": len(inputs), "sessions": sorted({r.session_date.isoformat() for r in inputs})}},
        "models": [_model_summary("jev-live", list(predictions), labeled)],
        "cases": _cases(labeled, groups),
        "limitations": [
            "Small prospective integration pilot; not a validated benchmark or estimate of future performance.",
            "No model selection or calibration was fitted to these pilot outcomes.",
            "Prediction dataset IDs identify the original input snapshots, before later outcomes arrived.",
            "Inference records were matched to the local observation store; local records are not provider signatures.",
            "No trades were executed. Direction forecasts do not account for transaction costs or position sizing.",
        ],
    }
    files = {
        "inputs.jsonl": "".join(row.model_dump_json() + "\n" for row in inputs),
        "examples.jsonl": "".join(row.model_dump_json() + "\n" for row in labeled),
        "predictions.jsonl": "".join(row.model_dump_json() + "\n" for row in predictions),
        "config.json": config.model_dump_json(indent=2) + "\n",
    }
    return _write_bundle(out_dir, report, files)
