"""Input-only Kronos cohorts and descriptive price metrics; no model fitting or orders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from numpy.typing import NDArray

from tradecopilot.forecast.bars import HistoricalBar
from tradecopilot.forecast.contracts import content_hash, utc
from tradecopilot.forecast.sessions import session_bounds

_OFFSETS = (61, 121, 181, 241, 301)
_MINUTE = timedelta(minutes=1)
_NEW_YORK = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class KronosCase:
    case_id: str
    symbol: str
    as_of: datetime
    history: tuple[HistoricalBar, ...]
    future_times: tuple[datetime, ...]
    actual: tuple[HistoricalBar, ...]


def future_grid(
    history: Sequence[HistoricalBar],
    clock: Mapping[str, Any],
    recorded_at: datetime,
) -> tuple[tuple[datetime, ...], str]:
    """Name a closed-market gap explicitly; never backdate a live observation."""
    recorded_at = utc(recorded_at)
    if not history or type(clock.get("is_open")) is not bool:
        raise ValueError("history and a verified paper market clock are required")
    anchor = history[-1].end_time
    if anchor > recorded_at:
        raise ValueError("context has not completed")
    if clock["is_open"]:
        if recorded_at - anchor > 2 * _MINUTE:
            raise ValueError("intraday context is stale")
        bounds = session_bounds(anchor.astimezone(_NEW_YORK).date())
        if bounds is None or not bounds[0] < anchor < bounds[1]:
            raise ValueError("intraday anchor is outside the session")
        future = tuple(anchor + i * _MINUTE for i in range(1, 16))
        if future[-1] > bounds[1]:
            raise ValueError("intraday horizon extends beyond session close")
        return future, "intraday"
    opening = utc(datetime.fromisoformat(str(clock["next_open"])))
    bounds = session_bounds(opening.astimezone(_NEW_YORK).date())
    if bounds is None or bounds[0] != opening or opening <= recorded_at:
        raise ValueError("next session is not a future XNYS opening")
    return tuple(opening + i * _MINUTE for i in range(1, 16)), "next_session_open"


def build_cases(
    bars: Sequence[HistoricalBar],
    symbols: Sequence[str],
    dates: Sequence[date],
    *,
    offsets: Sequence[int] = _OFFSETS,
) -> tuple[tuple[KronosCase, ...], list[dict[str, Any]]]:
    if (
        not symbols
        or len(set(symbols)) != len(symbols)
        or not dates
        or len(set(dates)) != len(dates)
        or not offsets
        or len(set(offsets)) != len(offsets)
        or any(isinstance(i, bool) or not isinstance(i, int) or not 60 <= i <= 375 for i in offsets)
    ):
        raise ValueError("invalid fixed case grid")
    lookup = {(bar.symbol, bar.end_time): bar for bar in bars}
    if len(lookup) != len(bars):
        raise ValueError("duplicate source bar")
    cases, catalog = [], []
    for day in sorted(dates):
        bounds = session_bounds(day)
        if bounds is None:
            raise ValueError("case date is not an XNYS session")
        for offset in sorted(offsets):
            as_of = bounds[0] + offset * _MINUTE
            future = tuple(as_of + i * _MINUTE for i in range(1, 16))
            for symbol in sorted(symbols):
                input_times = tuple(as_of - i * _MINUTE for i in reversed(range(60)))
                history = tuple(lookup.get((symbol, t)) for t in input_times)
                actual = tuple(lookup.get((symbol, t)) for t in future)
                reason = None
                if future[-1] > bounds[1]:
                    reason = "target_outside_session"
                elif any(b is None for b in history):
                    reason = "incomplete_input_window"
                elif any(b is not None and (b.available_at > as_of or b.start_time < bounds[0]) for b in history):
                    reason = "late_input"
                elif any(b is None for b in actual):
                    reason = "incomplete_outcome_window"
                elif any(b is not None and b.available_at > b.end_time for b in actual):
                    reason = "late_outcome"
                row = {
                    "symbol": symbol,
                    "session_date": day.isoformat(),
                    "as_of": as_of.isoformat(),
                    "target_time": future[-1].isoformat(),
                    "eligible": reason is None,
                    "reason": reason,
                }
                if reason is None:
                    inputs = tuple(b for b in history if b is not None)
                    outcomes = tuple(b for b in actual if b is not None)
                    identity = content_hash(
                        {
                            "version": "kronos-input-v1",
                            "symbol": symbol,
                            "history_bar_ids": [b.bar_id for b in inputs],
                            "future_times": future,
                            "model_timezone": "America/New_York",
                            "horizon": 15,
                            "lookback": 60,
                        }
                    )
                    row["case_id"] = identity
                    cases.append(KronosCase(identity, symbol, as_of, inputs, future, outcomes))
                catalog.append(row)
    return tuple(cases), catalog


def baseline_paths(history: Sequence[HistoricalBar], horizon: int) -> dict[str, NDArray[np.float64]]:
    if len(history) < 6 or isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 60:
        raise ValueError("baseline requires six past bars and a bounded horizon")
    prices = [float(b.close) for b in history]
    anchor = prices[-1]
    slope = (prices[-1] - prices[-6]) / 5
    result = {}
    for name, closes in (
        ("persistence", np.full(horizon, anchor)),
        ("momentum", anchor + slope * np.arange(1, horizon + 1)),
    ):
        if (closes <= 0).any() or not np.isfinite(closes).all():
            raise ValueError("nonphysical momentum prediction")
        path = np.zeros((1, horizon, 6), dtype=np.float64)
        path[:, :, :4] = closes[None, :, None]
        result[name] = path
    return result


def _direction(price: Decimal | float, anchor: Decimal) -> int:
    change = (Decimal(str(price)) / anchor - 1) * 10_000
    return 0 if change < -10 else 2 if change > 10 else 1


def price_metrics(
    cases: Sequence[KronosCase],
    predictions: Mapping[str, NDArray[np.float64] | None],
    *,
    sample_intervals: bool,
) -> dict[str, Any]:
    ids = [case.case_id for case in cases]
    if len(set(ids)) != len(ids) or set(predictions) != set(ids):
        raise ValueError("complete identical prediction cohorts required")
    errors = sum(predictions[identity] is None for identity in ids)
    terminal, path_errors, correct, covered, cm = [], [], [], [], np.zeros((3, 3), dtype=np.int64)
    for case in cases:
        value = predictions[case.case_id]
        if value is None:
            continue
        p = np.asarray(value, dtype=np.float64)
        if (
            p.ndim != 3
            or not 1 <= p.shape[0] <= 32
            or p.shape[1:] != (15, 6)
            or not np.isfinite(p).all()
            or (p[:, :, 3] <= 0).any()
        ):
            raise ValueError("invalid forecast paths")
        actual = np.asarray([float(b.close) for b in case.actual])
        anchor = case.history[-1].close
        mean = p[:, :, 3].mean(axis=0)
        absolute_bps = np.abs(mean - actual) / float(anchor) * 10_000
        terminal.append(float(absolute_bps[-1]))
        path_errors.append(float(absolute_bps.mean()))
        truth, predicted = _direction(case.actual[-1].close, anchor), _direction(float(mean[-1]), anchor)
        correct.append(truth == predicted)
        cm[truth, predicted] += 1
        if sample_intervals:
            low, high = np.quantile(p[:, :, 3], (0.1, 0.9), axis=0)
            covered.append(float(((actual >= low) & (actual <= high)).mean()))
    scored = len(correct)
    return {
        "eligible": len(cases),
        "scored": scored,
        "errors": errors,
        "forecast_availability": scored / len(cases) if cases else 0,
        "terminal_mae_bps": float(np.mean(terminal)) if terminal else None,
        "path_mae_bps": float(np.mean(path_errors)) if path_errors else None,
        "direction_accuracy": float(np.mean(correct)) if correct else None,
        "direction_accuracy_errors_as_incorrect": sum(correct) / len(cases) if cases else None,
        "confusion_matrix": cm.tolist(),
        "class_order": ["DOWN", "FLAT", "UP"],
        "raw_interval_coverage": float(np.mean(covered)) if covered else None,
        "interval_nominal_mass": 0.8 if sample_intervals else None,
        "intervals_calibrated": False,
        "uncertainty_interval": None,
        "evidence": "retrospective_development_pilot; no confirmation or promotion",
    }
