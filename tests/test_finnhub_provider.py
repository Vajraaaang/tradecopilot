from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from tradecopilot.config import StrategyConfig
from tradecopilot.models import DataQuality, DecisionState
from tradecopilot.strategy import DecisionEngine


def client(reply=None):
    from tradecopilot.providers.finnhub import FinnhubClient

    now = datetime.now(UTC)
    payload = {"c": 101.25, "pc": 100.0, "o": 100.1, "h": 102, "l": 99, "t": int(now.timestamp())}
    if reply is not None:
        payload.update(reply)
    calls = []

    def transport(symbol, key):
        calls.append((symbol, key))
        return payload

    return FinnhubClient("TEST_SECRET", transport=transport, clock=lambda: now), calls, now


def test_finnhub_preserves_real_price_and_timestamp_without_inventing_fields():
    instance, calls, now = client()
    price = instance.quote(" aapl ")
    assert price.symbol == "AAPL"
    assert price.last == Decimal("101.25") and price.previous_close == Decimal("100")
    assert price.provider_timestamp == datetime.fromtimestamp(int(now.timestamp()), UTC)
    assert price.age_seconds == (now - price.provider_timestamp).total_seconds()
    assert price.source == "finnhub_quote" and price.quality == DataQuality.LIMITED
    assert not any(key in price.model_dump() for key in ("bid", "ask", "volume", "total_volume"))
    assert "TEST_SECRET" not in price.model_dump_json()
    assert calls == [("AAPL", "TEST_SECRET")]


@pytest.mark.parametrize(
    "bad",
    [
        {"c": 0},
        {"c": -1},
        {"c": float("nan")},
        {"c": float("inf")},
        {"pc": 0},
        {"pc": None},
        {"t": 0},
        {"t": True},
        {"t": "not-a-timestamp"},
        {"t": 9999999999},
        {"error": "TEST_SECRET"},
    ],
)
def test_invalid_quote_fails_without_exposing_credentials(bad):
    instance, calls, _ = client(bad)
    with pytest.raises(ConnectionError) as failure:
        instance.quote("AAPL")
    assert "TEST_SECRET" not in str(failure.value)
    assert len(calls) == 1


def test_old_quote_keeps_its_age_and_is_marked_stale():
    old = datetime.now(UTC) - timedelta(hours=12)
    instance, _, now = client({"t": int(old.timestamp())})
    price = instance.quote("AAPL")
    assert price.quality == DataQuality.STALE
    assert price.age_seconds > 43000
    assert price.provider_timestamp < now


@pytest.mark.parametrize("symbol", ["", "AAPL&token=anything", "../../secret", "AAPL/MSFT", "1234"])
def test_invalid_symbol_never_reaches_transport(symbol):
    instance, calls, _ = client()
    with pytest.raises(ValueError):
        instance.quote(symbol)
    assert calls == []


def test_price_only_frame_never_becomes_a_strategy_quote(tmp_path):
    from tradecopilot.journal import Journal
    from tradecopilot.monitor import Monitor
    from tradecopilot.providers.finnhub import FinnhubFrameProvider

    instance, calls, _ = client()
    provider = FinnhubFrameProvider("AAPL", "TEST_SECRET", client=instance)

    async def once():
        frames = provider.frames()
        try:
            return await anext(frames)
        finally:
            await frames.aclose()

    frame = asyncio.run(once())
    assert frame.quote is None and frame.price_snapshot is not None
    assert frame.bars_1m == () and frame.bars_5m == ()
    assert frame.position is None and frame.account_risk is None
    decision = DecisionEngine(StrategyConfig()).evaluate(frame)
    assert decision.state == DecisionState.DATA_INSUFFICIENT
    assert decision.symbol == "AAPL" and decision.current_price == Decimal("101.25")
    assert decision.provider_timestamp == frame.price_snapshot.provider_timestamp
    assert decision.trade_plan is None and decision.indicator is None
    assert any("bid/ask" in value for value in decision.missing_data)
    assert len(calls) == 1

    class OneFrame:
        async def frames(self):
            yield frame

    with Journal(tmp_path / "journal.sqlite3") as journal:
        decisions = asyncio.run(Monitor(StrategyConfig(), journal, render_terminal=False).run(OneFrame()))
    assert decisions[0].symbol == "AAPL"


def test_symbol_change_discards_inflight_old_quote(monkeypatch):
    from tradecopilot.providers.finnhub import FinnhubFrameProvider

    instance, _, _ = client()
    original = instance.quote
    fetched = []

    def quote(symbol):
        fetched.append(symbol)
        if symbol == "AAPL":
            provider.select_symbol("MSFT")
        return original(symbol)

    async def no_delay(seconds):
        assert seconds >= 5

    monkeypatch.setattr(instance, "quote", quote)
    monkeypatch.setattr("tradecopilot.providers.finnhub.asyncio.sleep", no_delay)
    provider = FinnhubFrameProvider("AAPL", "TEST_SECRET", client=instance)

    async def once():
        frames = provider.frames()
        try:
            return await anext(frames)
        finally:
            await frames.aclose()

    frame = asyncio.run(once())
    assert frame.price_snapshot.symbol == "MSFT"
    assert fetched == ["AAPL", "MSFT"]


def test_refresh_interval_cannot_exceed_free_plan_cadence():
    from tradecopilot.providers.finnhub import FinnhubFrameProvider

    with pytest.raises(ValueError):
        FinnhubFrameProvider("AAPL", "test", poll_interval_seconds=0.1)


def test_full_quote_journal_provenance_is_not_relabelled_as_finnhub(yxt_frames):
    from tradecopilot.monitor import _observation

    instance, _, _ = client()
    frame = yxt_frames[0].model_copy(update={"price_snapshot": instance.quote("AAPL")})
    decision = DecisionEngine(StrategyConfig()).evaluate(frame)
    observation = _observation(decision, frame, explanation_model=None, prompt_version="test")
    assert observation.market_snapshot["last"] == str(frame.quote.last)
    assert observation.market_snapshot.get("price_only") is not True
    assert observation.market_snapshot.get("price_source") != "finnhub_quote"


@pytest.mark.parametrize("status", [200, 401, 429, 500])
def test_transport_uses_header_auth_and_never_retries(monkeypatch, status):
    from tradecopilot.providers.finnhub import _get_quote

    events = []

    class Connection:
        def __init__(self, host, timeout):
            events.append((host, timeout))

        def request(self, method, path, *, headers):
            events.append((method, path, headers))

        def getresponse(self):
            self.status = status
            return self

        def read(self, size):
            return json.dumps({"c": 101, "pc": 100, "t": 100} if status == 200 else {"error": "TEST_SECRET"}).encode()

        def close(self):
            events.append("closed")

    monkeypatch.setattr("tradecopilot.providers.finnhub.http.client.HTTPSConnection", Connection)
    if status == 200:
        assert _get_quote("AAPL", "TEST_SECRET")["c"] == 101
    else:
        with pytest.raises(ConnectionError) as failure:
            _get_quote("AAPL", "TEST_SECRET")
        assert "TEST_SECRET" not in str(failure.value)
    assert events[0] == ("finnhub.io", 5)
    assert events[1][0:2] == ("GET", "/api/v1/quote?symbol=AAPL")
    assert events[1][2]["X-Finnhub-Token"] == "TEST_SECRET"
    assert len(events) == 3 and events[-1] == "closed"
