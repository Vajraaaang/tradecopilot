"""Chronological tuning, independent calibration and fixed selective-accuracy criteria."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tradecopilot.forecast.contracts import LABELS, ForecastExample

GATE_THRESHOLDS = tuple(round(0.35 + i * 0.05, 2) for i in range(13))
GATE_REQUIREMENTS = {
    "target_accuracy": 0.8,
    "minimum_coverage": 0.1,
    "minimum_selected": 100,
    "minimum_direction_count": 30,
    "minimum_selected_sessions": 5,
}


def _ordered(examples: Sequence[ForecastExample]) -> list[ForecastExample]:
    rows = sorted(examples, key=lambda r: (r.as_of, r.symbol))
    if not rows or any(r.label is None or r.label_observed_at is None for r in rows):
        raise ValueError("labeled chronological examples are required")
    if len({r.example_id for r in rows}) != len(rows):
        raise ValueError("duplicate chronological examples")
    if len({r.config_id for r in rows}) != 1:
        raise ValueError("chronological target configuration mismatch")
    return rows


def _purge(left: list[ForecastExample], right: list[ForecastExample]) -> list[ForecastExample]:
    start = min(r.as_of for r in right)
    result = [r for r in left if r.label_observed_at is not None and r.label_observed_at < start]
    if not result:
        raise ValueError("empty chronological block after horizon purge")
    return result


def chronological_blocks(examples: Sequence[ForecastExample]) -> dict[str, list[ForecastExample]]:
    rows = _ordered(examples)
    days = sorted({r.session_date for r in rows})
    if len(days) < 100:
        raise ValueError("selective confirmation requires at least 100 labeled sessions")
    sets = (days[:-30], days[-30:-20], days[-20:-10], days[-10:])
    values = [[r for r in rows if r.session_date in set(group)] for group in sets]
    for i in range(3):
        values[i] = _purge(values[i], values[i + 1])
    return dict(zip(("development", "calibration", "gate", "test"), values, strict=True))


def tuning_folds(development: Sequence[ForecastExample]) -> list[tuple[list[ForecastExample], list[ForecastExample]]]:
    rows = _ordered(development)
    days = sorted({r.session_date for r in rows})
    if len(days) < 25:
        raise ValueError("walk-forward tuning requires at least 25 development sessions")
    folds = []
    for i in range(3):
        end = len(days) - 15 + i * 5
        train = [r for r in rows if r.session_date in set(days[:end])]
        valid = [r for r in rows if r.session_date in set(days[end : end + 5])]
        folds.append((_purge(train, valid), valid))
    return folds


def _probabilities(values: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=float)
    if (
        result.ndim != 2
        or result.shape[1] != 3
        or not len(result)
        or not np.isfinite(result).all()
        or (result < 0).any()
        or (result > 1).any()
        or not np.allclose(result.sum(axis=1), 1, atol=1e-8, rtol=0)
    ):
        raise ValueError("invalid three-class probabilities")
    return result


def apply_temperature(probabilities: NDArray[np.float64], temperature: float) -> NDArray[np.float64]:
    p = _probabilities(probabilities)
    if isinstance(temperature, bool) or not np.isfinite(temperature) or not 0.25 <= temperature <= 4:
        raise ValueError("invalid calibration temperature")
    logits = np.log(np.clip(p, 1e-15, 1)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_temperature(probabilities: NDArray[np.float64], targets: NDArray[np.int64]) -> float:
    p = _probabilities(probabilities)
    y = np.asarray(targets)
    if y.shape != (len(p),) or y.dtype.kind not in "iu" or not np.isin(y, [0, 1, 2]).all():
        raise ValueError("calibration labels and probabilities must align")
    grid = np.geomspace(0.25, 4, 81)
    losses = [-np.log(np.clip(apply_temperature(p, float(t))[np.arange(len(y)), y], 1e-15, 1)).mean() for t in grid]
    return float(grid[int(np.argmin(losses))])


def selection_metrics(
    examples: Sequence[ForecastExample],
    probabilities: NDArray[np.float64],
    threshold: float | None,
    *,
    confidence_interval: bool = True,
) -> dict[str, Any]:
    p = _probabilities(probabilities)
    if len(examples) != len(p) or any(r.label not in LABELS for r in examples):
        raise ValueError("evaluation labels and probabilities must align")
    if len({r.example_id for r in examples}) != len(examples):
        raise ValueError("duplicate evaluation examples")
    if threshold is not None and (isinstance(threshold, bool) or not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("invalid selection threshold")
    y = np.asarray([LABELS.index(str(r.label)) for r in examples])
    predicted = p.argmax(axis=1)
    correct = predicted == y
    selected = np.zeros(len(p), dtype=bool) if threshold is None else p.max(axis=1) >= threshold
    by_label = {}
    for i, label in enumerate(LABELS):
        mask = selected & (predicted == i)
        by_label[label] = {"count": int(mask.sum()), "precision": float(correct[mask].mean()) if mask.any() else None}
    days = sorted({r.session_date for r in examples})
    by_session = {}
    indices = {}
    for day in days:
        indices[day] = np.asarray([i for i, r in enumerate(examples) if r.session_date == day])
        members = indices[day]
        active = members[selected[members]]
        by_session[str(day)] = {
            "eligible": len(members),
            "selected": len(active),
            "accuracy": float(correct[members].mean()),
            "selective_accuracy": float(correct[active].mean()) if len(active) else None,
        }
    selected_sessions = int(sum(bool(value["selected"]) for value in by_session.values()))
    interval = None
    if confidence_interval and selected_sessions >= 5:
        rng = np.random.default_rng(42)
        values = []
        for _ in range(500):
            sampled = np.concatenate([indices[days[int(i)]] for i in rng.integers(0, len(days), len(days))])
            active = sampled[selected[sampled]]
            if len(active):
                values.append(float(correct[active].mean()))
        interval = {
            "method": "session_bootstrap_95pct",
            "low": float(np.percentile(values, 2.5)),
            "high": float(np.percentile(values, 97.5)),
            "sessions": len(days),
        }
    return {
        "eligible": len(p),
        "accuracy": float(correct.mean()),
        "selected": int(selected.sum()),
        "coverage": float(selected.mean()),
        "selective_accuracy": float(correct[selected].mean()) if selected.any() else None,
        "by_predicted_label": by_label,
        "selected_sessions": selected_sessions,
        "selective_accuracy_interval": interval,
        "by_session": by_session,
    }


def gate_qualifies(metrics: dict[str, Any]) -> bool:
    req = GATE_REQUIREMENTS
    accuracy = metrics["selective_accuracy"]
    return bool(
        accuracy is not None
        and accuracy >= req["target_accuracy"]
        and metrics["coverage"] >= req["minimum_coverage"]
        and metrics["selected"] >= req["minimum_selected"]
        and metrics["selected_sessions"] >= req["minimum_selected_sessions"]
        and all(
            metrics["by_predicted_label"][label]["count"] >= req["minimum_direction_count"]
            and metrics["by_predicted_label"][label]["precision"] >= req["target_accuracy"]
            for label in ("UP", "DOWN")
        )
    )


def select_gate(examples: Sequence[ForecastExample], probabilities: NDArray[np.float64]) -> dict[str, Any]:
    # Bootstrap only the chosen gate during final reporting; tuning has no interval claim.
    curve = []
    for threshold in GATE_THRESHOLDS:
        metrics = selection_metrics(examples, probabilities, threshold, confidence_interval=False)
        curve.append(
            {
                "threshold": threshold,
                "qualifies": gate_qualifies(metrics),
                **{k: v for k, v in metrics.items() if k not in {"by_session", "selective_accuracy_interval"}},
            }
        )
    valid = [r for r in curve if r["qualifies"]]
    chosen = max(valid, key=lambda r: (r["coverage"], -r["threshold"])) if valid else None
    return {
        "enabled": chosen is not None,
        "threshold": chosen["threshold"] if chosen else None,
        "requirements": dict(GATE_REQUIREMENTS),
        "curve": curve,
        "reason": "qualified_on_gate_selection" if chosen else "target_not_achieved_on_gate_selection",
    }
