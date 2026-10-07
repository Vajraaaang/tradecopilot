"""Offline CPU selection followed by separately fitted calibration, gate and final test."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tradecopilot.forecast.bars import HistoricalBar, load_bar_dataset
from tradecopilot.forecast.baselines import ModelArtifact, fit_model
from tradecopilot.forecast.contracts import (
    LABELS,
    DatasetManifest,
    ForecastConfig,
    ForecastExample,
    ForecastPrediction,
    Observation,
    content_hash,
)
from tradecopilot.forecast.dataset import build_dataset, write_dataset
from tradecopilot.forecast.evaluation import score_predictions
from tradecopilot.forecast.experiment import (
    REPORT_VERSION,
    _cases,
    _group_metrics,
    _json,
    _write_bundle,
    source_provenance,
)
from tradecopilot.forecast.ohlcv_features import OhlcvFeatureBuilder, OhlcvFeatureRecord
from tradecopilot.forecast.ohlcv_models import HGB_SETTINGS, OhlcvModelArtifact, fit_ohlcv_model
from tradecopilot.forecast.selective import (
    GATE_REQUIREMENTS,
    GATE_THRESHOLDS,
    apply_temperature,
    chronological_blocks,
    fit_temperature,
    gate_qualifies,
    select_gate,
    selection_metrics,
    tuning_folds,
)
from tradecopilot.forecast.sessions import session_bounds, session_for
from tradecopilot.models import DataQuality

Model = ModelArtifact | OhlcvModelArtifact
CANDIDATES = (
    {"key": "price-only-unweighted", "kind": "logistic", "settings": {}},
    *({"key": f"ohlcv-logistic-C{c}", "kind": "ohlcv_logistic", "settings": {"C": c}} for c in (0.01, 0.1, 1.0)),
    *(
        {"key": f"ohlcv-hgb-L2{v}", "kind": "ohlcv_hist_gradient_boosting", "settings": {"l2_regularization": v}}
        for v in (1.0, 10.0, 50.0)
    ),
)
INSPECTED_START = date(2026, 9, 16)
INSPECTED_END = date(2026, 9, 30)
STUDY_SYMBOLS = ("AAPL", "AMZN", "MSFT", "NFLX", "TSLA")


def replay_observations(bars: Sequence[HistoricalBar]) -> list[Observation]:
    grouped: dict[tuple[str, date], list[HistoricalBar]] = defaultdict(list)
    for bar in bars:
        day = session_for(bar.end_time)
        if day is None:
            raise ValueError("historical bar outside regular session")
        grouped[(bar.symbol, day)].append(bar)
    result = []
    for (symbol, day), rows in sorted(grouped.items()):
        prior = day - timedelta(days=1)
        while session_bounds(prior) is None:
            prior -= timedelta(days=1)
        previous = sorted(grouped.get((symbol, prior), []), key=lambda r: r.end_time)
        if not previous:
            continue
        for bar in sorted(rows, key=lambda r: r.end_time):
            visible = [r for r in previous if r.available_at <= bar.end_time]
            if not visible:
                continue
            result.append(
                Observation(
                    symbol=symbol,
                    last=bar.close,
                    previous_close=visible[-1].close,
                    provider_timestamp=bar.end_time,
                    receipt_timestamp=bar.available_at,
                    age_seconds=(bar.available_at - bar.end_time).total_seconds(),
                    source=bar.source,
                    quality=DataQuality.LIMITED,
                    provenance="historical",
                )
            )
    return result


def _fit(
    candidate: dict[str, Any], rows: list[ForecastExample], records: list[OhlcvFeatureRecord], config: ForecastConfig
) -> tuple[Model, Any]:
    if candidate["kind"] == "logistic":
        return fit_model("logistic", rows, config, class_weight=None), None
    return fit_ohlcv_model(candidate["kind"], rows, records, config, settings=candidate["settings"])


def _predict(
    model: Model, rows: list[ForecastExample], records: list[OhlcvFeatureRecord], native: Any = None
) -> NDArray[np.float64]:
    if isinstance(model, OhlcvModelArtifact):
        probabilities = model.probabilities(records)
        if native is not None:
            np.testing.assert_allclose(probabilities, native.predict_proba(model.training_matrix(records)), atol=1e-10)
        return probabilities
    return model.probabilities(rows)


def _predictions(
    rows: list[ForecastExample],
    probabilities: NDArray[np.float64],
    model_id: str,
    dataset_id: str,
    threshold: float | None,
    calibrated: bool = False,
) -> list[ForecastPrediction]:
    return [
        ForecastPrediction(
            example_id=row.example_id,
            dataset_id=dataset_id,
            model_id=model_id,
            generated_at=row.as_of,
            status="ok" if threshold is not None and float(p.max()) >= threshold else "abstained",
            execution="local",
            probabilities=dict(zip(LABELS, p.tolist(), strict=True)),
            estimated_cost_usd=0,
            reason="offline_retrospective_cpu_inference",
            calibration_version="temperature-selective-v1" if calibrated else None,
        )
        for row, p in zip(rows, probabilities, strict=True)
    ]


def run_prepared_study(
    examples: Sequence[ForecastExample],
    records: Sequence[OhlcvFeatureRecord],
    config: ForecastConfig,
    bar_manifest: dict[str, Any],
    output_dir: Path,
    *,
    dataset_manifest: DatasetManifest | None = None,
) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("choose a new study directory; final test bundles are immutable")
    if any(INSPECTED_START <= r.session_date <= INSPECTED_END for r in examples):
        raise ValueError("previously inspected September 16-30 dates cannot enter this study")
    if len(examples) != len(records) or any(
        r.example_id != f.base_example_id
        or r.config_id != config.config_id
        or f.config_id != config.config_id
        or r.as_of != f.as_of
        or r.symbol != f.symbol
        or r.provenance != "historical"
        for r, f in zip(examples, records, strict=True)
    ):
        raise ValueError("historical examples and feature records must align")
    if dataset_manifest is not None and (
        dataset_manifest.config.config_id != config.config_id
        or dataset_manifest.provenance != "historical"
        or dataset_manifest.labeled_count != len(examples)
    ):
        raise ValueError("forecast dataset manifest does not align with study examples")
    dataset_id = (
        dataset_manifest.dataset_id
        if dataset_manifest is not None
        else content_hash(
            {
                "bar_dataset_id": bar_manifest["data_id"],
                "config_id": config.config_id,
                "example_ids": [row.example_id for row in examples],
            }
        )
    )
    blocks = chronological_blocks(examples)
    lookup = {r.base_example_id: r for r in records}

    def features(rows: list[ForecastExample]) -> list[OhlcvFeatureRecord]:
        return [lookup[r.example_id] for r in rows]

    protocol = {
        "version": "selective-accuracy-study-v1",
        "config": config.model_dump(mode="json"),
        "bar_dataset_id": bar_manifest["data_id"],
        "dataset_id": dataset_id,
        "candidates": CANDIDATES,
        "hgb_defaults": HGB_SETTINGS,
        "selection_objective": "pooled_walk_forward_log_loss",
        "temperature_grid": np.geomspace(0.25, 4, 81).tolist(),
        "gate_thresholds": GATE_THRESHOLDS,
        "gate_requirements": GATE_REQUIREMENTS,
        "block_ids": {k: [r.example_id for r in rows] for k, rows in blocks.items()},
        "paid_api_calls": 0,
        "test_policy": "score once after immutable selection; no subsequent retuning",
    }
    output_dir.mkdir(parents=True)
    (output_dir / "frozen-protocol.json").write_text(_json(protocol))
    tuning: list[dict[str, Any]] = []
    folds = tuning_folds(blocks["development"])
    fold_ids = [
        {"train": [r.example_id for r in train], "validation": [r.example_id for r in valid]} for train, valid in folds
    ]
    for candidate in CANDIDATES:
        losses = []
        fold_metrics = []
        for train, valid in folds:
            model, native = _fit(candidate, train, features(train), config)
            p = _predict(model, valid, features(valid), native)
            y = np.asarray([LABELS.index(str(r.label)) for r in valid])
            losses.extend((-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1))).tolist())
            prediction = _predictions(valid, p, model.model_id, dataset_id, config.abstention_threshold)
            fold_metrics.append(score_predictions(valid, prediction))
        tuning.append({"candidate": candidate, "pooled_log_loss": float(np.mean(losses)), "fold_metrics": fold_metrics})
    winner = min(tuning, key=lambda r: r["pooled_log_loss"])["candidate"]
    model, native = _fit(winner, blocks["development"], features(blocks["development"]), config)
    calibration = blocks["calibration"]
    calibration_p = _predict(model, calibration, features(calibration), native)
    temperature = fit_temperature(
        calibration_p, np.asarray([LABELS.index(str(r.label)) for r in calibration], dtype=np.int64)
    )
    gate_rows = blocks["gate"]
    gate_p = apply_temperature(_predict(model, gate_rows, features(gate_rows), native), temperature)
    gate = select_gate(gate_rows, gate_p)
    selection = {
        "candidate": winner,
        "model_id": model.model_id,
        "temperature": temperature,
        "gate": gate,
        "calibration_ids_hash": content_hash([r.example_id for r in calibration]),
        "gate_ids_hash": content_hash([r.example_id for r in gate_rows]),
    }
    # Persist selection before evaluating final outcomes. Test labels cannot influence any above choice.
    selection_id = content_hash(selection)
    (output_dir / "frozen-selection.json").write_text(_json(selection))
    frozen_model = model.model_dump_json(indent=2) + "\n"
    (output_dir / "frozen-model.json").write_text(frozen_model)
    prior = fit_model("prior", blocks["development"], config)
    control = fit_model("logistic", blocks["development"], config, class_weight=None)
    test = blocks["test"]
    raw = _predict(model, test, features(test), native)
    calibrated = apply_temperature(raw, temperature)
    groups = {
        "prior": _predictions(test, prior.probabilities(test), prior.model_id, dataset_id, config.abstention_threshold),
        "price-only-control": _predictions(
            test, control.probabilities(test), control.model_id, dataset_id, config.abstention_threshold
        ),
        "selected-raw": _predictions(test, raw, model.model_id, dataset_id, config.abstention_threshold),
        "selected-calibrated": _predictions(test, calibrated, selection_id, dataset_id, gate["threshold"], True),
    }
    selected = selection_metrics(test, calibrated, gate["threshold"])
    report = {
        "schema_version": REPORT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "evaluation_kind": "synthetic_selective_demo"
        if bar_manifest.get("source_metadata", {}).get("fixture")
        else "selective_accuracy_retrospective",
        "title": "TradeCopilot selective accuracy final test",
        "source": source_provenance(),
        "bar_dataset_id": bar_manifest["data_id"],
        "dataset": dataset_manifest.model_dump(mode="json")
        if dataset_manifest is not None
        else {
            "dataset_id": dataset_id,
            "config": config.model_dump(mode="json"),
            "provenance": "historical",
            "example_count": len(examples),
            "sessions": sorted({str(r.session_date) for r in examples}),
        },
        "data_source": bar_manifest.get("source_metadata", {}),
        "experiment_id": content_hash(protocol),
        "protocol": protocol,
        "split": {
            k: {"examples": len(rows), "sessions": sorted({str(r.session_date) for r in rows})}
            for k, rows in blocks.items()
        },
        "tuning": tuning,
        "selection": selection,
        "selection_metrics": selected,
        "target_status": "observed_target_met_on_test"
        if gate["enabled"] and gate_qualifies(selected)
        else "target_not_achieved",
        "production_promoted": False,
        "models": [
            {
                "key": key,
                "label": key,
                "execution": "local",
                "model_id": predictions[0].model_id,
                "metrics": score_predictions(test, predictions),
                "by_symbol": _group_metrics(test, predictions),
            }
            for key, predictions in groups.items()
        ],
        "cases": _cases(test, groups),
        "limitations": [
            "Retrospective held-out dates; design follows earlier experiments, not prospective proof.",
            "Historical bar availability is assumed; revised bars and execution costs are not modeled.",
            "UP/DOWN precision is directional accuracy, not profitable BUY/SELL recommendations.",
            "Selected accuracy must be read with coverage, counts, per-direction metrics and session intervals.",
            "Temperature calibration does not change argmax labels or all-case accuracy.",
            "No new Jev requests, no default change, and no promised 80% performance.",
        ],
    }
    files = {
        "model.json": frozen_model,
        "prior.json": prior.model_dump_json(indent=2) + "\n",
        "control.json": control.model_dump_json(indent=2) + "\n",
        "protocol.json": _json(protocol),
        "folds.json": _json(fold_ids),
        "frozen-selection.json": _json(selection),
        "source.json": _json(bar_manifest),
        "features.jsonl": "".join(r.model_dump_json() + "\n" for r in records),
        "examples.jsonl": "".join(r.model_dump_json() + "\n" for r in examples),
        "predictions.jsonl": "".join(p.model_dump_json() + "\n" for group in groups.values() for p in group),
    }
    return _write_bundle(output_dir / "report", report, files)


def run_selective_study(bars_directory: Path, output_dir: Path) -> Path:
    bars, manifest = load_bar_dataset(bars_directory.resolve())
    if not bars or any(bar.source != "alpaca_sip_1min_bar" for bar in bars):
        raise ValueError("selective study requires the separately acquired Alpaca SIP bar store")
    if any(INSPECTED_START <= bar.end_time.date() <= INSPECTED_END for bar in bars):
        raise ValueError("previously inspected dates cannot enter even as primer/reference bars")
    if tuple(sorted({bar.symbol for bar in bars})) != STUDY_SYMBOLS:
        raise ValueError("selective study requires the original five stock symbols")
    if output_dir.resolve().exists():
        raise FileExistsError("choose a new immutable study directory")
    config = ForecastConfig(symbols=tuple(sorted({bar.symbol for bar in bars})))
    dataset, all_examples = build_dataset(replay_observations(bars), config)
    examples = [row for row in all_examples if row.label is not None]
    # Check eligibility before expensive feature construction or creating output.
    chronological_blocks(examples)
    if any(INSPECTED_START <= r.session_date <= INSPECTED_END for r in examples):
        raise ValueError("previously inspected dates cannot enter this study")
    builder = OhlcvFeatureBuilder(bars)
    records = [builder.build(row) for row in examples]
    report = run_prepared_study(examples, records, config, manifest, output_dir, dataset_manifest=dataset)
    write_dataset(output_dir / "dataset", dataset, all_examples)
    return report
