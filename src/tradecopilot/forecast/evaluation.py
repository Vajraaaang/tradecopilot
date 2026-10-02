from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score  # type: ignore[import-untyped]

from tradecopilot.forecast.contracts import LABELS, ForecastExample, ForecastPrediction


def split_examples(examples: Sequence[ForecastExample]) -> dict[str, list[ForecastExample]]:
    labeled = sorted((row for row in examples if row.label is not None), key=lambda row: (row.as_of, row.symbol))
    sessions = sorted({row.session_date for row in labeled})
    if len(sessions) < 5:
        raise ValueError("A benchmark needs at least five distinct labeled sessions; a smaller capture is a pilot.")
    train_end = max(3, int(len(sessions) * 0.6))
    validation_end = min(len(sessions) - 1, train_end + max(1, int(len(sessions) * 0.2)))
    split = {
        "train": [row for row in labeled if row.session_date in sessions[:train_end]],
        "validation": [row for row in labeled if row.session_date in sessions[train_end:validation_end]],
        "test": [row for row in labeled if row.session_date in sessions[validation_end:]],
    }
    validation_start = min(row.as_of for row in split["validation"])
    test_start = min(row.as_of for row in split["test"])
    split["train"] = [
        row for row in split["train"] if row.label_observed_at is not None and row.label_observed_at < validation_start
    ]
    split["validation"] = [
        row for row in split["validation"] if row.label_observed_at is not None and row.label_observed_at < test_start
    ]
    if any(not rows for rows in split.values()):
        raise ValueError("Insufficient examples after purging overlapping outcome windows.")
    return split


def score_predictions(
    examples: Sequence[ForecastExample],
    predictions: Sequence[ForecastPrediction],
) -> dict[str, Any]:
    known = {row.example_id: row for row in examples}
    if len(known) != len(examples):
        raise ValueError("duplicate evaluation examples")
    ids = [prediction.example_id for prediction in predictions]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate predictions must be resolved before scoring")
    if any(identity not in known for identity in ids):
        raise ValueError("prediction refers to an unknown evaluation example")
    labeled = {identity: row for identity, row in known.items() if row.label is not None}
    eligible = len(labeled)
    attempted = [prediction for prediction in predictions if prediction.example_id in labeled]
    valid = [prediction for prediction in attempted if prediction.status != "error"]
    active = [prediction for prediction in valid if prediction.status == "ok"]
    latency = [prediction.latency_ms for prediction in predictions if prediction.latency_ms is not None]
    result: dict[str, Any] = {
        "eligible": eligible,
        "pending_outcomes": len(known) - eligible,
        "attempted": len(attempted),
        "scored": len(valid),
        "errors": len(attempted) - len(valid),
        "abstained": len(valid) - len(active),
        "missing_predictions": eligible - len(attempted),
        "coverage": len(active) / eligible if eligible else 0,
        "accuracy": None,
        "macro_f1": None,
        "log_loss": None,
        "brier_score": None,
        "selective_accuracy": None,
        "confusion_matrix": [[0] * 3 for _ in range(3)],
        "labels": list(LABELS),
        "reliability": {label: [] for label in LABELS},
        "coverage_curve": [],
        "class_counts": {label: sum(row.label == label for row in labeled.values()) for label in LABELS},
        "api_cost_estimate_usd": sum(p.estimated_cost_usd or 0 for p in predictions),
        "unknown_cost_count": sum(p.estimated_cost_usd is None and p.execution == "live_api" for p in predictions),
        "input_tokens": sum(p.input_tokens or 0 for p in predictions),
        "latency_ms": {
            "p50": float(np.percentile(latency, 50)) if latency else None,
            "p95": float(np.percentile(latency, 95)) if latency else None,
        },
        "accuracy_interval": None,
    }
    if not valid:
        return result
    probabilities = np.asarray(
        [[p.probabilities[label] for label in LABELS] for p in valid if p.probabilities is not None]
    )
    truth = np.asarray([LABELS.index(str(labeled[p.example_id].label)) for p in valid])
    predicted = probabilities.argmax(axis=1)
    correct = predicted == truth
    onehot = np.eye(3)[truth]
    active_mask = np.asarray([p.status == "ok" for p in valid])
    result.update(
        accuracy=float(correct.mean()),
        macro_f1=float(f1_score(truth, predicted, labels=[0, 1, 2], average="macro", zero_division=0)),
        log_loss=float(-np.log(np.clip(probabilities[np.arange(len(truth)), truth], 1e-15, 1)).mean()),
        brier_score=float(np.square(probabilities - onehot).sum(axis=1).mean()),
        selective_accuracy=float(correct[active_mask].mean()) if active_mask.any() else None,
        confusion_matrix=confusion_matrix(truth, predicted, labels=[0, 1, 2]).tolist(),
    )
    for index, label in enumerate(LABELS):
        for bucket in range(10):
            lower = bucket / 10
            upper = (bucket + 1) / 10
            mask = (probabilities[:, index] >= lower) & (
                (probabilities[:, index] < upper) if upper < 1 else (probabilities[:, index] <= 1)
            )
            if mask.any():
                result["reliability"][label].append(
                    {
                        "lower": round(float(lower), 1),
                        "upper": upper,
                        "count": int(mask.sum()),
                        "mean_probability": float(probabilities[mask, index].mean()),
                        "observed_frequency": float((truth[mask] == index).mean()),
                    }
                )
    decision_confidence = np.asarray(
        [
            min(max(p.probabilities.values()), p.model_confidence)
            if p.model_confidence is not None
            else max(p.probabilities.values())
            for p in valid
            if p.probabilities is not None
        ]
    )
    for threshold in (0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        mask = decision_confidence >= threshold
        result["coverage_curve"].append(
            {
                "threshold": threshold,
                "coverage": int(mask.sum()) / eligible if eligible else 0,
                "accuracy": float(correct[mask].mean()) if mask.any() else None,
                "count": int(mask.sum()),
            }
        )
    # Resample whole sessions, not correlated adjacent rows. Tiny cohorts do not receive a misleading interval.
    dates = sorted({labeled[p.example_id].session_date for p in valid})
    if len(dates) >= 5:
        indices = {
            day: np.asarray([i for i, p in enumerate(valid) if labeled[p.example_id].session_date == day])
            for day in dates
        }
        rng = np.random.default_rng(42)
        values = []
        for _ in range(500):
            sampled = np.concatenate([indices[dates[int(i)]] for i in rng.integers(0, len(dates), len(dates))])
            values.append(float(correct[sampled].mean()))
        result["accuracy_interval"] = {
            "method": "session_bootstrap_95pct",
            "sessions": len(dates),
            "low": float(np.percentile(values, 2.5)),
            "high": float(np.percentile(values, 97.5)),
        }
    return result
