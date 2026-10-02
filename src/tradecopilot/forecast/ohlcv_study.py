"""Five frozen CPU arms on a shared validation cohort; no paid inference or new test claim."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tradecopilot.forecast.bars import import_frd_bars, load_bar_dataset, write_bar_dataset
from tradecopilot.forecast.baselines import fit_model, load_model, predict_model
from tradecopilot.forecast.contracts import FEATURE_NAMES, LABELS, ForecastPrediction, content_hash
from tradecopilot.forecast.dataset import load_dataset
from tradecopilot.forecast.evaluation import score_predictions, split_examples
from tradecopilot.forecast.experiment import (
    REPORT_VERSION,
    _cases,
    _group_metrics,
    _json,
    _model_summary,
    _write_bundle,
    load_report,
    source_provenance,
)
from tradecopilot.forecast.ohlcv_features import OHLCV_FEATURE_NAMES, OHLCV_FEATURE_VERSION, OhlcvFeatureBuilder
from tradecopilot.forecast.ohlcv_models import HGB_SETTINGS, ModelKind, fit_ohlcv_model


def _audit(matrix: np.ndarray[Any, Any], names: tuple[str, ...]) -> dict[str, Any]:
    result = {}
    for index, name in enumerate(names):
        values = matrix[:, index]
        valid = values[np.isfinite(values)]
        result[name] = {
            "count": len(values),
            "non_null": len(valid),
            "missing": len(values) - len(valid),
            "min": float(valid.min()) if len(valid) else None,
            "max": float(valid.max()) if len(valid) else None,
            "mean": float(valid.mean()) if len(valid) else None,
            "std": float(valid.std()) if len(valid) else None,
            "distinct": len(np.unique(valid)),
        }
    return result


def run_ohlcv_validation(dataset: Path, archives: Path, baseline: Path, output_dir: Path) -> Path:
    dataset, archives, baseline, output_dir = (path.resolve() for path in (dataset, archives, baseline, output_dir))
    if output_dir.exists():
        raise FileExistsError("choose a new output directory; validation studies are immutable")
    manifest, examples = load_dataset(dataset)
    original = load_report(baseline)
    if manifest.provenance != "historical" or original["dataset"]["dataset_id"] != manifest.dataset_id:
        raise ValueError("historical validation must use the original dataset and verified baseline")
    split = split_examples(examples)
    # No test feature construction or candidate scoring. These dates were already inspected.
    train, validation = split["train"], split["validation"]
    bars, source = import_frd_bars(archives, manifest.config.symbols)
    if source["downloads"] != original["data_source"]["downloads"]:
        raise ValueError("OHLCV archives differ from the frozen price-only source")
    output_dir.mkdir(parents=True)
    write_bar_dataset(output_dir / "bars", bars, source)
    verified_bars, bar_manifest = load_bar_dataset(output_dir / "bars")
    builder = OhlcvFeatureBuilder(verified_bars)
    started = time.perf_counter()
    train_features = [builder.build(row) for row in train]
    validation_features = [builder.build(row) for row in validation]
    feature_elapsed = (time.perf_counter() - started) * 1000
    protocol = {
        "version": "ohlcv-five-arm-validation-v1",
        "feature_version": OHLCV_FEATURE_VERSION,
        "base_dataset_id": manifest.dataset_id,
        "bar_dataset_id": bar_manifest["data_id"],
        "horizon_minutes": manifest.config.horizon_minutes,
        "flat_band_bps": manifest.config.flat_threshold_bps,
        "abstention_threshold": manifest.config.abstention_threshold,
        "seed": manifest.config.seed,
        "train_ids": [row.example_id for row in train],
        "validation_ids": [row.example_id for row in validation],
        "test_policy": "Previously inspected test excluded; no fresh confirmation or significance claim",
        "arms": ["prior", "v1-balanced-logistic", "v1-unweighted-logistic", "ohlcv-logistic", "ohlcv-hgb"],
        "hgb_settings": HGB_SETTINGS,
        "lr_settings": {"C": 1.0, "solver": "lbfgs", "max_iter": 1000},
        "paid_api_calls": 0,
    }
    (output_dir / "frozen-protocol.json").write_text(_json(protocol), encoding="utf-8")
    prior = load_model(baseline.parent / "prior.json")
    balanced = load_model(baseline.parent / "logistic.json")
    if any(set(model.training_ids) != {row.example_id for row in train} for model in (prior, balanced)):
        raise ValueError("original model training provenance does not match the fixed split")
    unweighted = fit_model("logistic", train, manifest.config, class_weight=None)
    v1_models = {"prior": prior, "v1-balanced-logistic": balanced, "v1-unweighted-logistic": unweighted}
    groups = {
        name: predict_model(model, validation, manifest.dataset_id, manifest.config)
        for name, model in v1_models.items()
    }
    files = {name + ".json": model.model_dump_json(indent=2) + "\n" for name, model in v1_models.items()}
    summaries = [_model_summary(name, groups[name], validation, model) for name, model in v1_models.items()]
    timings: dict[str, Any] = {"feature_build_batch_ms": feature_elapsed}
    arms: tuple[tuple[str, ModelKind], ...] = (
        ("ohlcv-logistic", "ohlcv_logistic"),
        ("ohlcv-hgb", "ohlcv_hist_gradient_boosting"),
    )
    for name, kind in arms:
        fitted_at = time.perf_counter()
        model, native = fit_ohlcv_model(kind, train, train_features, manifest.config)
        fit_ms = (time.perf_counter() - fitted_at) * 1000
        inferred_at = time.perf_counter()
        probabilities = model.probabilities(validation_features)
        inference_ms = (time.perf_counter() - inferred_at) * 1000
        # Prove non-executable JSON inference on actual missing/branch paths against the fitted estimator.
        native_values = np.asarray(native.predict_proba(model.training_matrix(validation_features)))
        if not np.allclose(probabilities, native_values, atol=1e-10):
            raise ValueError("portable model inference disagrees with the fitted estimator")
        model_id = model.model_id
        predictions = [
            ForecastPrediction(
                example_id=row.example_id,
                dataset_id=manifest.dataset_id,
                model_id=model_id,
                generated_at=datetime.now(UTC),
                execution="local",
                status="ok" if max(values) >= manifest.config.abstention_threshold else "abstained",
                reason=None if max(values) >= manifest.config.abstention_threshold else "below_probability_threshold",
                probabilities={label: float(value) for label, value in zip(LABELS, values, strict=True)},
                model_version=model.schema_version,
                model_confidence=float(max(values)),
                estimated_cost_usd=0,
            )
            for row, values in zip(validation, probabilities, strict=True)
        ]
        groups[name] = predictions
        files[name + ".json"] = model.model_dump_json(indent=2) + "\n"
        summary = _model_summary(name, predictions, validation)
        summary["artifact"] = name + ".json"
        summaries.append(summary)
        timings[name] = {
            "fit_ms": fit_ms,
            "batch_inference_ms": inference_ms,
            "latency_policy": "batch measured; per-case latency intentionally unclaimed",
        }
    names = {
        "prior": "Training class prior",
        "v1-balanced-logistic": "Price-only logistic · balanced",
        "v1-unweighted-logistic": "Price-only logistic · unweighted",
        "ohlcv-logistic": "OHLCV logistic · unweighted",
        "ohlcv-hgb": "OHLCV histogram gradient boosting",
    }
    for summary in summaries:
        summary["label"] = names[summary["key"]]
        summary["by_session"] = {}
        for day in sorted({row.session_date for row in validation}):
            day_rows = [row for row in validation if row.session_date == day]
            day_ids = {row.example_id for row in day_rows}
            day_predictions = [p for p in groups[summary["key"]] if p.example_id in day_ids]
            summary["by_session"][day.isoformat()] = {
                "overall": score_predictions(day_rows, day_predictions),
                "by_symbol": _group_metrics(day_rows, day_predictions),
            }
    audits = {}
    for name, rows, features in (("train", train, train_features), ("validation", validation, validation_features)):
        matrix = np.asarray(
            [
                [np.nan if row.values[key] is None else row.values[key] for key in OHLCV_FEATURE_NAMES]
                for row in features
            ],
            dtype=float,
        )
        audits[name] = {
            "ohlcv_v2": _audit(matrix, OHLCV_FEATURE_NAMES),
            "price_v1": _audit(np.asarray([[row.features[k] for k in FEATURE_NAMES] for row in rows]), FEATURE_NAMES),
        }
    report = {
        "schema_version": REPORT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "evaluation_kind": "ohlcv_validation_development",
        "title": "OHLCV CPU development comparison",
        "dataset": manifest.model_dump(mode="json"),
        "data_source": source,
        "source": source_provenance(),
        "feature_version": OHLCV_FEATURE_VERSION,
        "bar_dataset_id": bar_manifest["data_id"],
        "experiment_id": content_hash(protocol),
        "protocol": protocol,
        "split": {
            "train": {"examples": len(train), "sessions": sorted({row.session_date.isoformat() for row in train})},
            "validation": {
                "examples": len(validation),
                "sessions": sorted({row.session_date.isoformat() for row in validation}),
            },
        },
        "models": summaries,
        "cases": _cases(validation, groups),
        "feature_audit": audits,
        "timings": timings,
        "limitations": [
            "Validation development comparison; the original inspected test was not used as fresh confirmation.",
            "Only six training and two validation sessions; correlated overlapping targets limit evidence.",
            "Missing intraday windows are explicit; LR imputation/scaling fits training only; HGB preserves NaNs.",
            "HGB settings were fixed before scoring and its random internal early-stopping holdout is disabled.",
            "Historical receipt timing, prior-close and bar VWAP are documented replay/proxy assumptions.",
            "Market/sector/spread/order-flow inputs are unavailable and omitted; no field was fabricated.",
            "No default model, confidence gate or label change. Unseen future sessions are required for confirmation.",
            "Zero Jev/API calls in this study; current global usage limits and previous artifacts remain intact.",
        ],
    }
    files |= {
        "protocol.json": _json(protocol),
        "historical-source.json": _json(source),
        "feature-audit.json": _json(audits),
        "timings.json": _json(timings),
        "examples.jsonl": "".join(row.model_dump_json() + "\n" for row in validation),
        "features.jsonl": "".join(row.model_dump_json() + "\n" for row in [*train_features, *validation_features]),
        "predictions.jsonl": "".join(p.model_dump_json() + "\n" for values in groups.values() for p in values),
    }
    return _write_bundle(output_dir / "report", report, files)
