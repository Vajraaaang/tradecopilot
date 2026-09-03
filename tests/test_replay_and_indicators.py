from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from tradecopilot.config import StrategyConfig
from tradecopilot.indicators import atr, ema, rsi, session_vwap
from tradecopilot.models import DataQuality, DecisionState, OHLCVBar
from tradecopilot.monitor import transition_states
from tradecopilot.providers.replay import ReplayFormatError, ReplayProvider
from tradecopilot.strategy import DecisionEngine


def _bar(index: int, close: str, *, high: str | None = None, low: str | None = None) -> OHLCVBar:
    timestamp = datetime(2026, 8, 10, 13, 30, tzinfo=UTC) + timedelta(minutes=index)
    close_value = Decimal(close)
    return OHLCVBar(
        provider_timestamp=timestamp,
        receipt_timestamp=timestamp,
        age_seconds=0,
        source="trusted_fixture",
        quality=DataQuality.GOOD,
        symbol="TST",
        timeframe="1m",
        open=close_value,
        high=Decimal(high) if high else close_value + Decimal("1"),
        low=Decimal(low) if low else close_value - Decimal("1"),
        close=close_value,
        volume=100,
    )


def test_indicator_calculations_match_trusted_fixtures() -> None:
    values = [Decimal(value) for value in ("1", "2", "3", "4", "5")]
    assert ema(values, 3) == Decimal("4")
    bars = [_bar(index, str(index + 2)) for index in range(14)]
    assert atr(bars, 14) == Decimal("2")
    assert rsi([Decimal(index) for index in range(1, 17)], 14) == Decimal("100")


def test_session_vwap_uses_typical_price_and_volume() -> None:
    bars = [_bar(0, "10", high="11", low="9"), _bar(1, "12", high="13", low="11")]
    assert session_vwap(bars, "extended_hours") == Decimal("11")


def test_replay_has_no_lookahead(tmp_path) -> None:
    path = tmp_path / "future.jsonl"
    event = {
        "event_time": "2026-08-10T13:30:00Z",
        "seed_1m": [
            {
                "timestamp": "2026-08-10T13:31:00Z",
                "symbol": "YXT",
                "open": "9",
                "high": "9.1",
                "low": "8.9",
                "close": "9",
                "volume": 100,
            }
        ],
    }
    path.write_text(json.dumps(event) + "\n", encoding="utf-8")

    async def collect() -> None:
        async for _ in ReplayProvider(path, 0).frames():
            pass

    with pytest.raises(ReplayFormatError, match="look-ahead"):
        asyncio.run(collect())


def test_yxt_replay_transition_sequence(yxt_frames) -> None:
    engine = DecisionEngine(StrategyConfig())
    decisions = [engine.evaluate(frame) for frame in yxt_frames]
    assert transition_states(decisions) == [
        DecisionState.WATCH,
        DecisionState.ARMED,
        DecisionState.BUY,
        DecisionState.HOLD,
        DecisionState.EXIT_WARNING,
        DecisionState.SELL,
        DecisionState.REENTRY_WATCH,
    ]
