from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from tradecopilot.config import StrategyConfig
from tradecopilot.level2 import Level2Analyzer
from tradecopilot.models import DataQuality, Level2Level, Level2Snapshot, SignalAvailability


def _snapshot(second: int, big_size: int | None) -> Level2Snapshot:
    timestamp = datetime(2026, 8, 10, 13, 30, tzinfo=UTC) + timedelta(seconds=second)

    def level(side: str, price: str, size: int) -> Level2Level:
        return Level2Level(
            provider_timestamp=timestamp,
            receipt_timestamp=timestamp,
            age_seconds=0,
            source="test",
            quality=DataQuality.GOOD,
            side=side,
            price=Decimal(price),
            size=size,
        )

    asks = [level("ask", "9.16", 2_000)]
    if big_size is not None:
        asks.insert(0, level("ask", "9.50", big_size))
    return Level2Snapshot(
        provider_timestamp=timestamp,
        receipt_timestamp=timestamp,
        age_seconds=0,
        source="test",
        quality=DataQuality.GOOD,
        symbol="YXT",
        bids=(level("bid", "9.14", 3_000),),
        asks=tuple(asks),
    )


def _persistent(history: list[Level2Snapshot]):
    return Level2Analyzer(StrategyConfig()).signals(history, None)[0]


def test_one_level2_snapshot_cannot_confirm_persistence() -> None:
    signal = _persistent([_snapshot(0, 50_000)])
    assert not signal.confirmed


def test_flashing_seller_does_not_confirm_exit() -> None:
    signal = _persistent([_snapshot(0, 50_000), _snapshot(1, None), _snapshot(2, None)])
    assert not signal.confirmed


def test_persistent_normalized_seller_can_confirm_exit() -> None:
    signal = _persistent([_snapshot(0, 50_000), _snapshot(1, 52_000), _snapshot(2, 54_000)])
    assert signal.confirmed
    assert signal.availability == SignalAvailability.ACTIVE


def test_hidden_seller_cannot_confirm_without_time_and_sales() -> None:
    signals = Level2Analyzer(StrategyConfig()).signals([_snapshot(0, None)], None)
    hidden = next(signal for signal in signals if signal.name == "hidden_seller")
    assert not hidden.confirmed
    assert hidden.availability == SignalAvailability.UNAVAILABLE


def test_red_tape_burst_cannot_confirm_without_time_and_sales() -> None:
    signals = Level2Analyzer(StrategyConfig()).signals([_snapshot(0, None)], None)
    red_burst = next(signal for signal in signals if signal.name == "red_tape_burst")
    assert not red_burst.confirmed
    assert red_burst.availability == SignalAvailability.UNAVAILABLE
