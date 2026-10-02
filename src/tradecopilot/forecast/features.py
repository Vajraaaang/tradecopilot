"""Price-only inputs fixed at receipt time, with separately matured outcomes."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from itertools import pairwise
from statistics import pstdev

from tradecopilot.forecast.contracts import (
    ForecastConfig,
    ForecastExample,
    ForecastLabel,
    Observation,
    utc,
)
from tradecopilot.forecast.sessions import session_bounds, session_for


def normalize_observations(observations: Sequence[Observation]) -> list[Observation]:
    unique: dict[str, Observation] = {}
    for item in observations:
        row = Observation.model_validate(item.model_dump(mode="json"))
        unique[row.observation_id] = row
    return [row for _, row in sorted(
        unique.items(), key=lambda pair: (
            pair[1].receipt_timestamp, pair[1].provider_timestamp, pair[1].symbol, pair[0],
        ),
    )]


def _return_bps(last: Decimal, first: Decimal) -> Decimal:
    return (last / first - 1) * 10_000


def feature_result(
    observations: Sequence[Observation], as_of: datetime, symbol: str, config: ForecastConfig,
) -> tuple[ForecastExample | None, str | None]:
    """Build from validated rows; return an exclusion code for dataset accounting."""
    as_of = utc(as_of)
    if symbol not in config.symbols:
        return None, "symbol_not_configured"
    day = session_for(as_of)
    if day is None:
        return None, "outside_session"
    bounds = session_bounds(day)
    assert bounds is not None
    target = as_of + timedelta(minutes=config.horizon_minutes)
    if target > bounds[1]:
        return None, "target_outside_session"
    start = as_of - timedelta(minutes=config.lookback_minutes)
    # Keep the predecessor just before the history boundary for exact as-of returns.
    lower = start - timedelta(seconds=30)
    causal = [row for row in observations if (
        row.symbol == symbol
        and lower <= row.provider_timestamp <= row.receipt_timestamp <= as_of
        and bounds[0] <= row.provider_timestamp <= bounds[1]
    )]
    if not causal:
        return None, "insufficient_history"
    if len({row.provenance for row in causal}) != 1:
        raise ValueError("feature provenance cannot mix synthetic and market observations")
    by_provider: dict[datetime, Observation] = {}
    for row in sorted(causal, key=lambda item: (item.receipt_timestamp, item.observation_id)):
        by_provider[row.provider_timestamp] = row
    ordered = sorted(by_provider.values(), key=lambda item: item.provider_timestamp)
    anchor = ordered[-1]
    source_age = (as_of - anchor.provider_timestamp).total_seconds()
    if source_age > config.max_source_age_seconds:
        return None, "stale_source"
    history = [row for row in ordered if row.provider_timestamp >= start]
    if len(history) < config.min_history_points:
        return None, "insufficient_history"
    span = (history[-1].provider_timestamp - history[0].provider_timestamp).total_seconds()
    if span < config.lookback_minutes * 60 - 30:
        return None, "insufficient_history"
    times = [row.provider_timestamp for row in history] + [as_of]
    if any((right - left).total_seconds() > config.max_history_gap_seconds for left, right in pairwise(times)):
        return None, "history_gap"
    endpoints: list[Observation] = []
    for minutes in (1, 5):
        endpoint = as_of - timedelta(minutes=minutes)
        eligible = [row for row in ordered if row.provider_timestamp <= endpoint]
        if not eligible or (endpoint - eligible[-1].provider_timestamp).total_seconds() > 30:
            return None, "missing_return_endpoint"
        endpoints.append(eligible[-1])
    recent = [row for row in history if row.provider_timestamp >= as_of - timedelta(minutes=5)]
    changes = [float(_return_bps(right.last, left.last)) for left, right in pairwise(recent)]
    prices = [row.last for row in recent]
    inputs = {row.observation_id: row for row in [*history, *endpoints]}
    observation_ids = tuple(key for key, _ in sorted(
        inputs.items(), key=lambda pair: (pair[1].provider_timestamp, pair[1].receipt_timestamp, pair[0]),
    ))
    return ForecastExample(
        config_id=config.config_id, symbol=symbol, as_of=as_of, target_time=target, session_date=day,
        anchor_price=anchor.last, observation_ids=observation_ids, provenance=anchor.provenance,
        features={
            "return_1m_bps": float(_return_bps(anchor.last, endpoints[0].last)),
            "return_5m_bps": float(_return_bps(anchor.last, endpoints[1].last)),
            "range_5m_bps": float((max(prices) - min(prices)) / anchor.last * 10_000),
            "volatility_5m_bps": pstdev(changes),
            "change_from_previous_close_bps": float(_return_bps(anchor.last, anchor.previous_close)),
            "source_age_seconds": source_age,
            "history_points": float(len(history)),
        },
        exclusion_reason="outcome_unavailable",
    ), None


def feature_example(
    observations: Sequence[Observation], as_of: datetime, symbol: str, config: ForecastConfig,
) -> ForecastExample | None:
    return feature_result(normalize_observations(observations), as_of, symbol, config)[0]


def label_example(
    observations: Sequence[Observation], example: ForecastExample, config: ForecastConfig,
) -> ForecastExample:
    """Use the earliest provider outcome within the bounded receipt window."""
    if example.config_id != config.config_id:
        raise ValueError("label config does not match the forecast input")
    deadline = example.target_time + timedelta(seconds=config.max_outcome_delay_seconds)
    bounds = session_bounds(example.session_date)
    reason = "outcome_unavailable"
    candidates = []
    for row in observations:
        if (
            row.symbol != example.symbol or row.provenance != example.provenance or bounds is None
            or not example.target_time <= row.provider_timestamp <= bounds[1]
            or row.provider_timestamp < bounds[0] or row.receipt_timestamp < row.provider_timestamp
        ):
            continue
        if row.receipt_timestamp > deadline:
            reason = "outcome_late"
        else:
            # Revalidate only candidate outcomes, not every historic row for every anchor.
            candidates.append(Observation.model_validate(row.model_dump(mode="json")))
    if not candidates:
        return example.model_copy(update={
            "label": None, "target_price": None, "label_observed_at": None,
            "target_return_bps": None, "exclusion_reason": reason,
        })
    outcome = min(candidates, key=lambda row: (row.provider_timestamp, row.receipt_timestamp, row.observation_id))
    change = _return_bps(outcome.last, example.anchor_price)
    threshold = Decimal(str(config.flat_threshold_bps))
    label: ForecastLabel = "UP" if change > threshold else "DOWN" if change < -threshold else "FLAT"
    return example.model_copy(update={
        "label": label, "target_price": outcome.last, "label_observed_at": outcome.receipt_timestamp,
        "target_return_bps": float(change), "exclusion_reason": None,
    })
