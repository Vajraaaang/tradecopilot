from __future__ import annotations

import asyncio
from datetime import timedelta

from tradecopilot.config import StrategyConfig
from tradecopilot.journal import Journal
from tradecopilot.level2 import Level2Analyzer
from tradecopilot.models import (
    HistoricalContextEvidence,
    PositionChangeKind,
    RunMode,
    SignalAvailability,
    TimeAndSalesPrint,
)
from tradecopilot.positions import PositionTracker
from tradecopilot.providers.live import LiveMcpFrameProvider
from tradecopilot.providers.polling import PollingFrameProvider
from tradecopilot.security import READ_ONLY_ALLOWLIST


class FixtureMarket:
    def __init__(self, frame) -> None:
        self.frame = frame

    async def get_quote(self, symbol):
        assert symbol == self.frame.quote.symbol
        return self.frame.quote.model_copy(update={"average_daily_volume_50d": None})

    async def get_bars(self, symbol, timeframe, start, end):
        del symbol, start, end
        return self.frame.bars_1m if timeframe == "1m" else self.frame.bars_5m

    async def get_level2(self, symbol):
        del symbol
        return self.frame.level2_history[-1]

    async def is_tradable(self, symbol):
        del symbol
        return True


class FixtureBroker:
    def __init__(self, frame) -> None:
        self.frame = frame

    async def get_position(self, symbol):
        del symbol
        return self.frame.position

    async def get_account_risk(self):
        return self.frame.account_risk

    async def get_trade_history(self):
        return ()

    async def get_open_orders(self, symbol):
        del symbol
        return ()


class FixtureSupplemental:
    def __init__(self, frame, context) -> None:
        self.frame = frame
        self.context = context

    async def get_float(self, symbol):
        del symbol
        return self.frame.float_evidence

    async def get_catalyst(self, symbol):
        del symbol
        return self.frame.catalyst_evidence

    async def get_historical_context(self, symbol):
        del symbol
        return self.context


class FixtureTape:
    def __init__(self, frame) -> None:
        self.frame = frame

    async def get_prints(self, symbol):
        del symbol
        return self.frame.time_and_sales


def test_shibui_context_enriches_volume_without_overwriting_live_market_data(yxt_frames) -> None:
    source = yxt_frames[0]
    assert source.quote is not None
    context_time = source.event_time - timedelta(hours=1)
    context = HistoricalContextEvidence(
        provider_timestamp=context_time,
        receipt_timestamp=source.event_time,
        age_seconds=3600,
        source="shibui_mcp:query_database",
        quality=source.quote.quality,
        symbol=source.quote.symbol,
        average_daily_volume_50d=123_456,
        latest_daily_date=(source.event_time - timedelta(days=1)).date(),
        ownership_snapshot_timestamp=source.event_time - timedelta(days=1),
    )
    provider = PollingFrameProvider(
        source.quote.symbol,
        FixtureMarket(source),
        FixtureBroker(source),
        FixtureSupplemental(source, context),
        FixtureTape(source),
        StrategyConfig(),
        mode=RunMode.LIVE,
        clock=lambda: source.event_time,
    )

    frame = asyncio.run(provider.snapshot())

    assert frame.quote is not None
    assert frame.quote.last == source.quote.last
    assert frame.quote.bid == source.quote.bid
    assert frame.quote.ask == source.quote.ask
    assert frame.quote.source == source.quote.source
    assert frame.bars_1m == source.bars_1m
    assert frame.quote.average_daily_volume_50d == 123_456
    assert frame.historical_context == context
    assert not ({"last", "bid", "ask", "price", "bars"} & set(HistoricalContextEvidence.model_fields))


def test_manual_position_entry_and_exit_are_journaled(tmp_path, yxt_frames) -> None:
    tracker = PositionTracker()
    changes = [change for frame in yxt_frames if (change := tracker.observe(frame)) is not None]

    assert [change.kind for change in changes] == [PositionChangeKind.ENTRY, PositionChangeKind.EXIT]
    path = tmp_path / "journal.sqlite3"
    with Journal(path) as journal:
        for change in changes:
            journal.record_position_change(change)
    with Journal(path) as journal:
        report = journal.report(changes[0].provider_timestamp.date())
    assert report["position_events"] == {"ENTRY": 1, "EXIT": 1}


def test_live_symbol_switch_is_normalized_and_has_no_order_capability() -> None:
    provider = LiveMcpFrameProvider("aapl", None, StrategyConfig())
    assert provider.symbol == "AAPL"
    assert provider.select_symbol(" msft ") is True
    assert provider.symbol == "MSFT"
    assert not hasattr(provider, "place_order")
    assert not hasattr(provider, "cancel_order")


def test_application_robinhood_allowlist_is_exact_and_read_only() -> None:
    assert {
        "get_accounts",
        "get_portfolio",
        "get_realized_pnl",
        "get_pnl_trade_history",
        "search",
        "get_equity_historicals",
        "get_equity_fundamentals",
        "get_financials",
        "get_equity_price_book",
        "get_equity_technical_indicators",
        "get_earnings_results",
        "get_earnings_calendar",
        "get_indexes",
        "get_index_quotes",
        "get_equity_positions",
        "get_equity_tax_lots",
        "get_equity_quotes",
        "get_equity_orders",
        "get_equity_tradability",
        "get_scans",
        "get_scanner_filter_specs",
        "run_scan",
    } == READ_ONLY_ALLOWLIST


def test_unclassified_alpaca_tape_cannot_confirm_red_burst(yxt_frames) -> None:
    frame = yxt_frames[6]
    assert frame.quote is not None
    unknown_prints = tuple(
        TimeAndSalesPrint(
            provider_timestamp=frame.quote.provider_timestamp,
            receipt_timestamp=frame.quote.receipt_timestamp,
            age_seconds=frame.quote.age_seconds,
            source="alpaca_stream:sip",
            quality=frame.quote.quality,
            symbol=frame.quote.symbol,
            price=frame.quote.last,
            size=10_000,
            side="unknown",
        )
        for _ in range(6)
    )
    signals = Level2Analyzer(StrategyConfig()).signals(frame.level2_history, unknown_prints)
    red_burst = next(signal for signal in signals if signal.name == "red_tape_burst")
    assert red_burst.confirmed is False
    assert red_burst.availability == SignalAvailability.LIMITED
