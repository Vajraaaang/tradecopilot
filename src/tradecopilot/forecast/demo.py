"""Deterministic synthetic prices for software checks, never market-performance evidence."""

from __future__ import annotations

import random
from datetime import date, timedelta
from decimal import Decimal

from tradecopilot.forecast.contracts import ForecastConfig, Observation
from tradecopilot.forecast.sessions import session_bounds
from tradecopilot.models import DataQuality


def synthetic_observations(
    config: ForecastConfig, sessions: int = 8, minutes_per_session: int = 60, seed: int = 42,
) -> list[Observation]:
    days = [day for number in range(1, 31) if session_bounds(day := date(2026, 9, number)) is not None]
    if not 1 <= sessions <= len(days):
        raise ValueError(f"synthetic sessions must be between 1 and {len(days)}")
    if not 1 <= minutes_per_session <= 390:
        raise ValueError("synthetic minutes_per_session must be between 1 and 390")
    rng = random.Random(seed)
    result = []
    for day_index, day in enumerate(days[:sessions]):
        bounds = session_bounds(day)
        assert bounds is not None
        for symbol_index, symbol in enumerate(config.symbols):
            previous_close = Decimal(100 + 37 * symbol_index + 2 * day_index)
            last = previous_close
            session_open = last
            high = low = last
            for step in range(minutes_per_session * 4 + 1):
                if step:
                    regime = (day_index + symbol_index + (step - 1) // 80) % 3 - 1
                    move = Decimal(str(regime * 0.7 + rng.uniform(-0.06, 0.06)))
                    last = (last * (1 + move / 10_000)).quantize(Decimal("0.0001"))
                    high, low = max(high, last), min(low, last)
                timestamp = bounds[0] + timedelta(seconds=step * 15)
                result.append(Observation(
                    symbol=symbol, last=last, previous_close=previous_close,
                    session_open=session_open, session_high=high, session_low=low,
                    provider_timestamp=timestamp, receipt_timestamp=timestamp, age_seconds=0,
                    source="synthetic_fixture", quality=DataQuality.LIMITED, provenance="synthetic",
                ))
    return sorted(result, key=lambda row: (row.receipt_timestamp, row.provider_timestamp, row.symbol))
