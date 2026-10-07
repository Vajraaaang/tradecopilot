"""Fixed forecast correction on already consumed dates, never fresh confirmation.

This runner uses the existing verified retrospective cache without provider access.
The four arms and their settings are fixed before outcomes are scored. Feature
construction and inference accept no target labels; TRAIN is the only fit input.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import sklearn  # type: ignore[import-untyped]
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from tradecopilot.forecast.bars import HistoricalBar, load_bar_dataset
from tradecopilot.forecast.contracts import LABELS, content_hash
from tradecopilot.forecast.jev_rl import _artifact, _read, _registration, _safe_path, _seal, _write, load_cache
from tradecopilot.forecast.ohlcv_features import OHLCV_FEATURE_NAMES, OHLCV_FEATURE_VERSION, OhlcvFeatureBuilder
from tradecopilot.forecast.sessions import session_bounds
from tradecopilot.rl.data import _dates, _source_check
from tradecopilot.rl.jev_ablation import _validate_cache
from tradecopilot.rl.study import _lock_hash
from tradecopilot.rl.training import _source_provenance

ARMS = ("TRAIN_PRIOR", "RAW_JEV", "CONTEXT_LR", "JEV_FUSION_LR")
METADATA_NAMES = (
    "jev_available",
    "jev_status_ok",
    "jev_status_abstained",
    "jev_status_unavailable",
    "jev_simulated_latency_minutes",
    "jev_minutes_to_target",
)
JEV_NAMES = ("jev_p_down", "jev_p_flat", "jev_p_up", "jev_model_confidence")
SETTINGS = {
    "C": 0.01,
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 42,
    "class_weight": None,
    "threads": 1,
    "objective": "unweighted_multinomial_log_loss",
}
_EPS = 1e-15


@dataclass(frozen=True)
class FusionCase:
    """An exact cache slot and its causal context, with no outcome fields."""

    case_id: str
    symbol: str
    session_date: date
    split: str
    as_of: int
    target_time: int
    replay_available_at: int
    anchor_price: Decimal
    input_id: str
    input_bars_hash: str
    generated_at: str
    status: str
    context: tuple[float | None, ...]
    jev_probabilities: tuple[float, ...]
    model_confidence: float

    @property
    def available(self) -> bool:
        return self.status in ("ok", "abstained")


def build_cases(
    bars: list[HistoricalBar],
    source: dict[str, Any],
    manifest: dict[str, Any],
    registration: dict[str, Any],
    cache: dict[str, Any],
) -> list[FusionCase]:
    """Verify the old registry and reconstruct every 15m slot without labels."""
    days = _registration(registration)
    records = _validate_cache(cache, manifest, registration)
    _source_check(bars, source, registration)
    splits = _dates(registration)
    symbols = tuple(registration["symbols"])
    config_id = content_hash(
        {"registration_id": registration["registration_id"], "feature_version": OHLCV_FEATURE_VERSION}
    )
    feature_names = [
        *OHLCV_FEATURE_NAMES,
        *(f"missing_{n}" for n in OHLCV_FEATURE_NAMES),
        *(f"symbol_{s}" for s in symbols),
    ]
    if (
        manifest.get("schema_version") != "rl-prepared-market-v1"
        or manifest.get("registration") != registration
        or manifest.get("source_manifest") != source
        or manifest.get("source_data_id") != source["data_id"]
        or manifest.get("config_id") != config_id
        or manifest.get("feature_names") != feature_names
    ):
        raise ValueError("old prepared/source/registration configuration mismatch")
    expected = {(s, day.isoformat()) for day in days for s in symbols}
    slots = [(e["symbol"], e["session_date"]) for e in manifest["episodes"]]
    if len(slots) != len(expected) or set(slots) != expected:
        raise ValueError("old prepared must contain every unique registered slot")
    for episode in manifest["episodes"]:
        day = date.fromisoformat(episode["session_date"])
        role = next(role for role, role_days in splits.items() if day in role_days)
        bounds = session_bounds(day)
        assert bounds is not None
        if (
            episode.get("role") != role
            or episode.get("session_open") != int(bounds[0].timestamp())
            or episode.get("session_close") != int(bounds[1].timestamp())
        ):
            raise ValueError("old prepared slot split or session bounds mismatch")
    builder = OhlcvFeatureBuilder(bars)
    anchors = {(b.symbol, int(b.end_time.timestamp())): b for b in bars}
    for (symbol, day_text, horizon), record in records.items():
        bounds = session_bounds(date.fromisoformat(day_text))
        assert bounds is not None
        as_of = int(bounds[0].timestamp()) + 61 * 60
        target = int(bounds[1].timestamp()) if horizon == "close" else as_of + int(horizon[:-1]) * 60
        anchor = anchors.get((symbol, as_of))
        if (
            record["as_of"] != as_of
            or record["target_time"] != target
            or target > int(bounds[1].timestamp())
            or anchor is None
            or anchor.available_at.timestamp() > as_of
            or Decimal(str(record["anchor_price"])) != anchor.close
        ):
            raise ValueError("cache as-of/target/source anchor mismatch")
    output = []
    for day in days:
        role = next(role for role, role_days in splits.items() if day in role_days)
        for symbol in symbols:
            record = records[(symbol, day.isoformat(), "15m")]
            feature = builder.build_as_of(symbol, datetime.fromtimestamp(record["as_of"], UTC), config_id)
            identity = content_hash(
                {
                    "schema": "jev-fusion-case-v1",
                    "cache_id": cache["cache_id"],
                    "input_id": record["input_id"],
                    "feature_id": feature.feature_id,
                    "split": role,
                }
            )
            output.append(
                FusionCase(
                    identity,
                    symbol,
                    day,
                    role,
                    record["as_of"],
                    record["target_time"],
                    record["replay_available_at"],
                    Decimal(str(record["anchor_price"])),
                    record["input_id"],
                    feature.input_bars_hash,
                    record["generated_at"],
                    record["status"],
                    tuple(feature.values[n] for n in OHLCV_FEATURE_NAMES),
                    tuple(float(record["probabilities"][n]) for n in LABELS),
                    float(record["model_confidence"]),
                )
            )
    if not registration.get("fixture") and len(output) != 400:
        raise ValueError("consumed-date comparison requires exactly 400 cases")
    return output


def labels_for_cases(bars: Sequence[HistoricalBar], cases: Sequence[FusionCase]) -> NDArray[np.int64]:
    """Exact target close relative to the frozen anchor, with inclusive +/-10bp FLAT."""
    by_end = {(bar.symbol, int(bar.end_time.timestamp())): bar for bar in bars}
    result = []
    for case in cases:
        target = by_end.get((case.symbol, case.target_time))
        if target is None or target.end_time.timestamp() != case.target_time:
            raise ValueError("exact target close unavailable; no case dropping or forward filling")
        change = Decimal(10000) * (target.close / case.anchor_price - 1)
        result.append(0 if change < -10 else 2 if change > 10 else 1)
    return np.asarray(result, dtype=np.int64)


def _raw(cases: Sequence[FusionCase]) -> NDArray[np.float64]:
    return np.asarray(
        [[np.nan if value is None else value for value in c.context] for c in cases], dtype=np.float64
    ).reshape(len(cases), 55)


def _design_matrices(
    cases: Sequence[FusionCase],
    normalizer: dict[str, Any],
    symbols: tuple[str, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    raw = _raw(cases)
    missing = np.isnan(raw)
    medians, mean, scale = (np.asarray(normalizer[k], dtype=float) for k in ("medians", "mean", "scale"))
    context_values = (np.where(missing, medians, raw) - mean) / scale
    onehot = np.asarray([[float(c.symbol == s) for s in symbols] for c in cases], dtype=float).reshape(len(cases), 5)
    metadata = np.asarray(
        [
            [
                float(c.available),
                float(c.status == "ok"),
                float(c.status == "abstained"),
                float(c.status == "unavailable"),
                (c.replay_available_at - c.as_of) / 60,
                (c.target_time - c.as_of) / 60,
            ]
            for c in cases
        ],
        dtype=float,
    ).reshape(len(cases), 6)
    jev = np.asarray(
        [[*c.jev_probabilities, c.model_confidence] if c.available else [0.0] * 4 for c in cases], dtype=float
    ).reshape(len(cases), 4)
    context = np.concatenate((context_values, missing.astype(float), onehot, metadata), axis=1)
    fusion = np.concatenate((context, jev), axis=1)
    if any(c.symbol not in symbols for c in cases) or not np.isfinite(fusion).all():
        raise ValueError("nonfinite features or unknown symbol")
    return context, fusion


def _softmax(matrix: NDArray[np.float64], parameters: dict[str, Any]) -> NDArray[np.float64]:
    with threadpool_limits(limits=1):
        logits = matrix @ np.asarray(parameters["coefficients"]).T + np.asarray(parameters["intercept"])
    exponent = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exponent / exponent.sum(axis=1, keepdims=True)


def fit_models(
    training_cases: Sequence[FusionCase],
    training_labels: NDArray[np.int64],
    symbols: tuple[str, ...],
) -> dict[str, Any]:
    """Fit only TRAIN; fixed unweighted multinomial LR, with no TUNE selection."""
    if (
        not training_cases
        or len(training_cases) != len(training_labels)
        or any(c.split != "train" for c in training_cases)
        or len(set(symbols)) != 5
    ):
        raise ValueError("only aligned TRAIN cases may fit the fixed models")
    if set(training_labels.tolist()) != {0, 1, 2}:
        raise ValueError("TRAIN must include all three classes for multinomial fitting")
    raw = _raw(training_cases)
    medians = np.asarray(
        [float(np.median(column[np.isfinite(column)])) if np.isfinite(column).any() else 0.0 for column in raw.T]
    )
    imputed = np.where(np.isnan(raw), medians, raw)
    mean, scale = imputed.mean(axis=0), imputed.std(axis=0)
    scale[scale == 0] = 1
    normalizer = {
        "feature_names": list(OHLCV_FEATURE_NAMES),
        "medians": medians.tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "all_missing_train_imputation": 0,
        "fit_split": "train",
        "fit_case_ids": [c.case_id for c in training_cases],
        "fit_dates": sorted({c.session_date.isoformat() for c in training_cases}),
    }
    matrices = _design_matrices(training_cases, normalizer, symbols)
    names = [
        *OHLCV_FEATURE_NAMES,
        *(f"missing_{n}" for n in OHLCV_FEATURE_NAMES),
        *(f"symbol_{s}" for s in symbols),
        *METADATA_NAMES,
    ]
    models: dict[str, Any] = {}
    errors = []
    for arm, matrix in zip(ARMS[2:], matrices, strict=True):
        model = LogisticRegression(C=0.01, solver="lbfgs", max_iter=1000, random_state=42, class_weight=None)
        with threadpool_limits(limits=1):
            model.fit(matrix, training_labels)
            native = model.predict_proba(matrix)
        if list(model.classes_) != [0, 1, 2]:
            raise ValueError("unexpected multinomial class order")
        parameters = {
            "feature_names": names if arm == "CONTEXT_LR" else [*names, *JEV_NAMES],
            "coefficients": model.coef_.tolist(),
            "intercept": model.intercept_.tolist(),
            "iterations": int(model.n_iter_.max()),
            "converged": bool(model.n_iter_.max() < 1000),
        }
        reconstructed = _softmax(matrix, parameters)
        error = float(np.max(np.abs(native - reconstructed)))
        if not np.isfinite(reconstructed).all() or error > 1e-12:
            raise ValueError("finite JSON/native multinomial inference parity failed")
        errors.append(error)
        models[arm] = parameters
    result: dict[str, Any] = {
        "schema_version": "jev-fusion-models-v1",
        "evidence_mode": "consumed_date_exploration",
        "classes": list(LABELS),
        "symbols": list(symbols),
        "settings": SETTINGS,
        "normalizer": normalizer,
        "class_prior": (np.bincount(training_labels, minlength=3) / len(training_labels)).tolist(),
        "models": models,
        "sklearn_version": sklearn.__version__,
        "train_features_labels_hash": content_hash(
            {
                "case_ids": normalizer["fit_case_ids"],
                "context_matrix": matrices[0].tolist(),
                "fusion_matrix": matrices[1].tolist(),
                "labels": training_labels.tolist(),
            }
        ),
        "native_train_parity_max_abs_error": max(errors),
    }
    result["model_id"] = content_hash(result)
    return result


def predict_models(model: dict[str, Any], cases: Sequence[FusionCase]) -> dict[str, NDArray[np.float64]]:
    """Pure JSON inference for all four fixed arms; unavailable Jev uses TRAIN prior."""
    _seal(model, "model_id")
    if (
        model.get("schema_version") != "jev-fusion-models-v1"
        or model.get("classes") != list(LABELS)
        or model.get("settings") != SETTINGS
        or len(set(model["symbols"])) != 5
    ):
        raise ValueError("invalid fusion model schema/settings")
    normalizer = model["normalizer"]
    if normalizer.get("feature_names") != list(OHLCV_FEATURE_NAMES) or normalizer.get("fit_split") != "train":
        raise ValueError("invalid TRAIN normalization feature order")
    for key in ("medians", "mean", "scale"):
        value = np.asarray(normalizer[key], dtype=float)
        if value.shape != (55,) or not np.isfinite(value).all() or (key == "scale" and (value <= 0).any()):
            raise ValueError("invalid finite feature normalization")
    matrices = _design_matrices(cases, normalizer, tuple(model["symbols"]))
    prior = np.asarray(model["class_prior"], dtype=float)
    if prior.shape != (3,) or not np.isfinite(prior).all() or (prior < 0).any() or not np.isclose(prior.sum(), 1):
        raise ValueError("invalid TRAIN class prior")
    output = {
        ARMS[0]: np.tile(prior, (len(cases), 1)),
        ARMS[1]: np.asarray([c.jev_probabilities if c.available else prior for c in cases], dtype=float).reshape(
            len(cases), 3
        ),
    }
    for arm, matrix in zip(ARMS[2:], matrices, strict=True):
        parameters = model["models"][arm]
        expected_names = [
            *OHLCV_FEATURE_NAMES,
            *(f"missing_{n}" for n in OHLCV_FEATURE_NAMES),
            *(f"symbol_{s}" for s in model["symbols"]),
            *METADATA_NAMES,
            *(JEV_NAMES if arm == "JEV_FUSION_LR" else ()),
        ]
        if parameters.get("feature_names") != expected_names:
            raise ValueError("invalid multinomial feature order")
        if (
            np.asarray(parameters["coefficients"]).shape != (3, matrix.shape[1])
            or np.asarray(parameters["intercept"]).shape != (3,)
            or not all(np.isfinite(np.asarray(parameters[k], dtype=float)).all() for k in ("coefficients", "intercept"))
        ):
            raise ValueError("invalid finite multinomial weights")
        output[arm] = _softmax(matrix, parameters) if len(cases) else np.empty((0, 3), dtype=float)
    for probabilities in output.values():
        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6)
        ):
            raise ValueError("invalid complete-case inference probabilities")
    return output


def _nll(probabilities: NDArray[np.float64], truth: NDArray[np.int64]) -> NDArray[np.float64]:
    return -np.log(np.clip(probabilities[np.arange(len(truth)), truth], _EPS, 1))


def score_probabilities(
    probabilities: NDArray[np.float64],
    truth: NDArray[np.int64],
    available: NDArray[np.bool_],
) -> dict[str, Any]:
    """All argmax predictions count, including abstained Jev and unavailable fallback."""
    n = len(truth)
    if probabilities.shape != (n, 3) or available.shape != (n,):
        raise ValueError("metric alignment mismatch")
    confusion = np.zeros((3, 3), dtype=np.int64)
    predicted = probabilities.argmax(axis=1)
    np.add.at(confusion, (truth, predicted), 1)
    support, called, correct = confusion.sum(axis=1), confusion.sum(axis=0), confusion.diagonal()
    precision = np.divide(correct, called, out=np.zeros(3), where=called > 0)
    recall = np.divide(correct, support, out=np.zeros(3), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3), where=precision + recall > 0)
    return {
        "eligible": n,
        "scored": n,
        "coverage": 1.0 if n else 0.0,
        "jev_available": int(available.sum()),
        "jev_availability_coverage": float(available.mean()) if n else 0.0,
        "accuracy": float((predicted == truth).mean()) if n else None,
        "balanced_accuracy": float(recall[support > 0].mean()) if n else None,
        "macro_f1": float(f1.mean()) if n else None,
        "confusion_matrix": confusion.tolist(),
        "labels": list(LABELS),
        "per_class": {
            name: {
                "support": int(support[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
            }
            for i, name in enumerate(LABELS)
        },
        "brier_score": float(np.square(probabilities - np.eye(3)[truth]).sum(axis=1).mean()) if n else None,
        "log_loss": float(_nll(probabilities, truth).mean()) if n else None,
    }


def paired_test_nll(
    cases: Sequence[FusionCase],
    truth: NDArray[np.int64],
    context: NDArray[np.float64],
    fusion: NDArray[np.float64],
) -> dict[str, Any]:
    if not cases or any(c.split != "test" for c in cases) or len(cases) != len(truth):
        raise ValueError("paired uncertainty requires complete ordered old TEST cases")
    difference = _nll(fusion, truth) - _nll(context, truth)
    days = sorted({c.session_date for c in cases})
    groups = [np.asarray([i for i, c in enumerate(cases) if c.session_date == day]) for day in days]
    rng = np.random.default_rng(42)
    draws = [
        float(difference[np.concatenate([groups[int(i)] for i in rng.integers(0, len(days), len(days))])].mean())
        for _ in range(1000)
    ]
    return {
        "comparison": "JEV_FUSION_LR minus CONTEXT_LR negative favors fusion",
        "metric": "per_case_log_loss",
        "mean_difference": float(difference.mean()),
        "low": float(np.percentile(draws, 2.5)),
        "high": float(np.percentile(draws, 97.5)),
        "resamples": 1000,
        "seed": 42,
        "dates": len(days),
        "cases": len(cases),
        "unit": "whole_date_all_symbols",
        "interpretation": "descriptive_consumed_dates_only",
        "confirmatory": False,
    }


def _provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    return {
        "source": _source_provenance(),
        "lock_sha256": _lock_hash(),
        "script": _artifact(root / "scripts" / "run_jev_fusion.py"),
        "python": str(Path(sys.executable).absolute()),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
    }


def run_jev_fusion(
    source_bars: Path,
    old_prepared: Path,
    old_registration: Path,
    cache: Path,
    output_dir: Path,
) -> Path:
    """Run the fixed consumed-date exploration once and return its sealed report."""
    if output_dir.exists():
        raise ValueError("fusion output is immutable; choose a new directory")
    for path in (source_bars, old_prepared, old_registration, cache, output_dir):
        _safe_path(path)
    registration, manifest = _read(old_registration), _read(old_prepared / "manifest.json")
    cached = load_cache(cache)
    bars, source = load_bar_dataset(source_bars)
    cases = build_cases(bars, source, manifest, registration, cached)
    provenance = _provenance()
    input_paths = {
        source_bars / "manifest.json",
        source_bars / "bars.jsonl",
        old_prepared / "manifest.json",
        old_registration,
        cache,
        *(Path(item["path"]) for item in cached["inventory"]),
    }
    inputs = [_artifact(p) for p in sorted(input_paths)]
    protocol: dict[str, Any] = {
        "schema_version": "jev-fusion-exploration-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "evidence_mode": "consumed_date_exploration",
        "confirmatory": False,
        "fresh_model_selection_allowed": False,
        "cache_id": cached["cache_id"],
        "source_data_id": source["data_id"],
        "old_prepared_data_id": manifest["data_id"],
        "old_registration_id": registration["registration_id"],
        "arms": list(ARMS),
        "settings": SETTINGS,
        "case_count": len(cases),
        "ordered_case_ids": [c.case_id for c in cases],
        "class_order": list(LABELS),
        "target": {
            "horizon_minutes": 15,
            "flat_threshold_bps_inclusive": 10,
            "anchor": "exact completed minute-bar close at cache as_of",
            "outcome": "exact target minute-bar close; no forward filling",
        },
        "splits": registration["splits"],
        "all_old_dates_consumed": True,
        "features": {
            "raw": list(OHLCV_FEATURE_NAMES),
            "missing_masks": 55,
            "symbol_onehots": list(registration["symbols"]),
            "shared_jev_metadata": list(METADATA_NAMES),
            "fusion_additions": list(JEV_NAMES),
            "fit_normalizer": "TRAIN only",
            "actual_generated_at_is_feature": False,
        },
        "missingness": "keep every case; zero unavailable Jev p/confidence in fusion; "
        "TRAIN-prior fallback for raw Jev; same metadata in both LR arms",
        "availability_mode": cached["availability_mode"],
        "no_selection": "no grid, tuning, calibration, gate, seed or learning-rate selection",
        "uncertainty": "1000 whole-date paired TEST NLL resamples with seed42; descriptive only",
        "provenance": provenance,
        "inputs": inputs,
    }
    protocol["protocol_id"] = content_hash(protocol)
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    _write(output_dir / "protocol.json", protocol)
    train_cases = [c for c in cases if c.split == "train"]
    train_labels = labels_for_cases(bars, train_cases)
    model = fit_models(train_cases, train_labels, tuple(registration["symbols"]))
    model.pop("model_id")
    model["protocol_id"] = protocol["protocol_id"]
    model["model_id"] = content_hash(model)
    _write(output_dir / "models.json", model)
    probabilities = predict_models(model, cases)
    reloaded = predict_models(_read(output_dir / "models.json"), cases)
    roundtrip_error = max(float(np.max(np.abs(probabilities[a] - reloaded[a]))) for a in ARMS)
    # The model and every prediction are frozen before TUNE/TEST outcome calculation.
    all_labels = labels_for_cases(bars, cases)
    available = np.asarray([c.available for c in cases], dtype=np.bool_)
    predictions: dict[str, Any] = {
        "schema_version": "jev-fusion-predictions-v1",
        "model_id": model["model_id"],
        "class_order": list(LABELS),
        "records": [
            {
                "case_id": c.case_id,
                "symbol": c.symbol,
                "session_date": c.session_date.isoformat(),
                "split": c.split,
                "as_of": c.as_of,
                "target_time": c.target_time,
                "replay_available_at": c.replay_available_at,
                "generated_at": c.generated_at,
                "status": c.status,
                "jev_available": c.available,
                "input_id": c.input_id,
                "input_bars_hash": c.input_bars_hash,
                "label": LABELS[int(all_labels[i])],
                "probabilities": {a: probabilities[a][i].tolist() for a in ARMS},
            }
            for i, c in enumerate(cases)
        ],
    }
    predictions["predictions_id"] = content_hash(predictions)
    _write(output_dir / "predictions.json", predictions)
    metrics: dict[str, Any] = {}
    for arm in ARMS:
        metrics[arm] = {}
        for role in ("train", "tune", "test"):
            mask = np.asarray([c.split == role for c in cases], dtype=np.bool_)
            matched = mask & available
            metrics[arm][role] = {
                "all_cases": score_probabilities(probabilities[arm][mask], all_labels[mask], available[mask]),
                "matched_available": score_probabilities(
                    probabilities[arm][matched], all_labels[matched], available[matched]
                ),
            }
    test = np.asarray([c.split == "test" for c in cases], dtype=np.bool_)
    paired = paired_test_nll(
        [c for c in cases if c.split == "test"],
        all_labels[test],
        probabilities["CONTEXT_LR"][test],
        probabilities["JEV_FUSION_LR"][test],
    )
    if (
        provenance != _provenance()
        or any(_artifact(Path(item["path"])) != item for item in inputs)
        or load_cache(cache) != cached
    ):
        raise ValueError("source/code/lock/cache changed during immutable exploration")
    report: dict[str, Any] = {
        "schema_version": "jev-fusion-report-v1",
        "evidence_mode": "consumed_date_exploration",
        "confirmatory": False,
        "fresh_model_selection_allowed": False,
        "protocol_id": protocol["protocol_id"],
        "model_id": model["model_id"],
        "predictions_id": predictions["predictions_id"],
        "cache_id": cached["cache_id"],
        "case_count": len(cases),
        "case_order_hash": content_hash([c.case_id for c in cases]),
        "split_counts": {r: sum(c.split == r for c in cases) for r in ("train", "tune", "test")},
        "split_dates": {
            r: sorted({c.session_date.isoformat() for c in cases if c.split == r}) for r in ("train", "tune", "test")
        },
        "metrics": metrics,
        "paired_test_nll": paired,
        "metric_definitions": {
            "coverage": "scored/eligible, including raw Jev prior fallback",
            "matched_available": "same ok/abstained subset for every arm",
            "balanced_accuracy": "mean recall of classes present in split",
            "macro_f1": "mean over all three classes; undefined values zero",
            "brier_score": "mean sum of squared errors over three classes",
            "log_loss": "mean negative log true-class probability clipped at 1e-15",
        },
        "inference_parity": {
            "max_abs_error": model["native_train_parity_max_abs_error"],
            "native_scope": "TRAIN cases only",
            "json_roundtrip_max_abs_error": roundtrip_error,
            "json_roundtrip_cases": len(cases),
        },
        "convergence": {a: model["models"][a]["converged"] for a in ARMS[2:]},
        "limitations": [
            "All old TRAIN/TUNE/TEST dates were already inspected; this is consumed-date exploration.",
            "Consumed TEST uncertainty is descriptive and cannot establish confirmation or select fresh models.",
            "Retrospective cached forecasts retain hypothetical latency; vendor pretraining cutoff is unknown.",
            "Exactly the existing five symbols, dates and 15m minute-bar close target are represented.",
        ],
        "provenance": provenance,
        "inputs": inputs,
        "artifacts": [_artifact(output_dir / n) for n in ("protocol.json", "models.json", "predictions.json")],
    }
    report["report_id"] = content_hash(report)
    report_path = output_dir / "report.json"
    _write(report_path, report)
    inventory: dict[str, Any] = {
        "schema_version": "jev-fusion-inventory-v1",
        "report_id": report["report_id"],
        "artifacts": [_artifact(p) for p in sorted(output_dir.iterdir())],
    }
    inventory["inventory_id"] = content_hash(inventory)
    _write(output_dir / "inventory.json", inventory)
    return report_path
