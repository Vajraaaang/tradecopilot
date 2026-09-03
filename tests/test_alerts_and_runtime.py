from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tradecopilot.alerts import AlertMessage, TransitionAlertDispatcher, WebhookAlertSink
from tradecopilot.backtest import ReplayBacktestRunner, write_nightly_report
from tradecopilot.config import StrategyConfig
from tradecopilot.journal import Journal
from tradecopilot.models import (
    AccountRiskSnapshot,
    CatalystEvidence,
    DecisionState,
    FloatEvidence,
    Level2Snapshot,
    MarketFrame,
    OHLCVBar,
    PositionSnapshot,
    Quote,
    RunMode,
    TimeAndSalesPrint,
)
from tradecopilot.monitor import Monitor
from tradecopilot.providers.polling import PollingFrameProvider
from tradecopilot.providers.supplemental import JsonSupplementalProvider
from tradecopilot.strategy import DecisionEngine

ROOT = Path(__file__).resolve().parents[1]


class RecordingSink:
    def __init__(self) -> None:
        self.messages: list[AlertMessage] = []

    async def send(self, message: AlertMessage) -> None:
        self.messages.append(message)


def test_monitor_sends_only_deduplicated_alert_state_changes(tmp_path, yxt_frames) -> None:
    sink = RecordingSink()
    dispatcher = TransitionAlertDispatcher(sink)

    class Provider:
        async def frames(self):
            for frame in yxt_frames:
                yield frame

    with Journal(tmp_path / "journal.sqlite3") as journal:
        monitor = Monitor(
            StrategyConfig(),
            journal,
            compact=True,
            render_terminal=False,
            alert_dispatcher=dispatcher,
        )
        asyncio.run(monitor.run(Provider()))

    assert [message.state for message in sink.messages] == [
        DecisionState.ARMED,
        DecisionState.BUY,
        DecisionState.EXIT_WARNING,
        DecisionState.SELL,
    ]
    assert all(message.manual_execution for message in sink.messages)


def test_webhook_rejects_unencrypted_remote_url() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        WebhookAlertSink("http://example.com/alerts")


def test_supplemental_feed_requires_sources_and_normalizes_all_three_inputs(tmp_path) -> None:
    path = tmp_path / "supplemental.json"
    path.write_text(
        json.dumps(
            {
                "symbols": {
                    "YXT": {
                        "float": {
                            "shares": 2_900_000,
                            "source": "verified float provider",
                            "timestamp": "2026-08-10T12:00:00Z",
                        },
                        "news": {
                            "headline": "Company published a factual release",
                            "source": "company release",
                            "timestamp": "2026-08-10T12:01:00Z",
                        },
                        "time_and_sales": [
                            {
                                "price": "9.60",
                                "size": 100,
                                "side": "buy",
                                "source": "licensed tape provider",
                                "timestamp": "2026-08-10T12:02:00Z",
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    provider = JsonSupplementalProvider(path)

    async def read():
        return await asyncio.gather(
            provider.get_float("YXT"),
            provider.get_catalyst("YXT"),
            provider.get_prints("YXT"),
        )

    float_evidence, catalyst, tape = asyncio.run(read())
    assert float_evidence is not None and float_evidence.verified
    assert float_evidence.shares == 2_900_000
    assert catalyst is not None and catalyst.reference == "company release"
    assert tape is not None and tape[0].side == "buy"


def test_supplemental_feed_rejects_unsourced_float(tmp_path) -> None:
    path = tmp_path / "supplemental.json"
    path.write_text(
        json.dumps(
            {
                "symbols": {
                    "YXT": {
                        "float": {"shares": 2_900_000, "timestamp": "2026-08-10T12:00:00Z"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source"):
        asyncio.run(JsonSupplementalProvider(path).get_float("YXT"))


class FakeMarket:
    def __init__(self, frame: MarketFrame) -> None:
        self.frame = frame
        self.quote_calls = 0

    async def get_quote(self, symbol: str) -> Quote:
        del symbol
        self.quote_calls += 1
        if self.quote_calls > 1:
            raise ConnectionError("synthetic quote outage")
        assert self.frame.quote is not None
        return self.frame.quote

    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> Sequence[OHLCVBar]:
        del symbol, start, end
        return self.frame.bars_1m if timeframe == "1m" else self.frame.bars_5m

    async def get_level2(self, symbol: str) -> Level2Snapshot:
        del symbol
        return self.frame.level2_history[-1]

    async def is_tradable(self, symbol: str) -> bool:
        del symbol
        return True


class FakeBroker:
    def __init__(self, frame: MarketFrame) -> None:
        self.frame = frame

    async def get_position(self, symbol: str) -> PositionSnapshot | None:
        del symbol
        return self.frame.position

    async def get_account_risk(self) -> AccountRiskSnapshot:
        assert self.frame.account_risk is not None
        return self.frame.account_risk

    async def get_trade_history(self) -> Sequence[dict[str, object]]:
        return ()

    async def get_open_orders(self, symbol: str) -> Sequence[dict[str, object]]:
        del symbol
        return ()


class FakeSupplemental:
    def __init__(self, frame: MarketFrame) -> None:
        self.frame = frame

    async def get_float(self, symbol: str) -> FloatEvidence | None:
        del symbol
        return self.frame.float_evidence

    async def get_catalyst(self, symbol: str) -> CatalystEvidence | None:
        del symbol
        return self.frame.catalyst_evidence


class FakeTape:
    def __init__(self, frame: MarketFrame) -> None:
        self.frame = frame

    async def get_prints(self, symbol: str) -> Sequence[TimeAndSalesPrint] | None:
        del symbol
        return self.frame.time_and_sales


def test_polling_provider_reuses_failed_quote_only_as_aged_stale_data(yxt_frames) -> None:
    source = yxt_frames[0]
    first_time = source.event_time
    times = iter((first_time, first_time, first_time + timedelta(seconds=3), first_time + timedelta(seconds=3)))
    provider = PollingFrameProvider(
        "YXT",
        FakeMarket(source),
        FakeBroker(source),
        FakeSupplemental(source),
        FakeTape(source),
        StrategyConfig(),
        clock=lambda: next(times),
    )
    first = asyncio.run(provider.snapshot())
    second = asyncio.run(provider.snapshot())
    assert first.mode == RunMode.LIVE
    assert first.quote is not None and first.quote.age_seconds == 0
    assert first.quote.current_minute_volume == source.bars_1m[-1].volume
    assert second.quote is not None and second.quote.age_seconds == 3
    assert second.bars_1m[-1].age_seconds == first.bars_1m[-1].age_seconds + 3


def test_stale_one_minute_bar_fails_closed_even_with_fresh_quote_and_book(yxt_frames) -> None:
    original = yxt_frames[0]
    event_time = original.event_time + timedelta(seconds=91)
    assert original.quote is not None
    quote = original.quote.model_copy(
        update={"provider_timestamp": event_time, "receipt_timestamp": event_time, "age_seconds": 0}
    )
    latest_book = original.level2_history[-1]
    book = latest_book.model_copy(
        update={"provider_timestamp": event_time, "receipt_timestamp": event_time, "age_seconds": 0}
    )
    frame = original.model_copy(update={"event_time": event_time, "quote": quote, "level2_history": (book,)})
    decision = DecisionEngine(StrategyConfig()).evaluate(frame)
    assert decision.state == DecisionState.DATA_STALE
    assert any("one-minute bar age" in reason for reason in decision.reasons)


def test_nightly_backtest_runs_simulator_and_writes_report(tmp_path) -> None:
    report = asyncio.run(ReplayBacktestRunner(StrategyConfig()).run((ROOT / "examples" / "yxt_replay.jsonl",)))
    target = write_nightly_report(report, tmp_path / "reports")
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert report.metrics.signals == 1
    assert report.incomplete_signals == 0
    assert report.result == "MORE_DATA"
    assert saved["strategy_version"] == "1.0.0-candidate"
