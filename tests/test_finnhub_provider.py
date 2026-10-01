from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from tradecopilot.config import StrategyConfig
from tradecopilot.models import DataQuality, DecisionState, MarketFrame, PriceSnapshot
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
    assert price.session_open == Decimal("100.1")
    assert price.session_high == Decimal("102") and price.session_low == Decimal("99")
    assert price.provider_timestamp == datetime.fromtimestamp(int(now.timestamp()), UTC)
    assert price.age_seconds == (now - price.provider_timestamp).total_seconds()
    assert price.source == "finnhub_quote" and price.quality == DataQuality.LIMITED
    assert not any(key in price.model_dump() for key in ("bid", "ask", "volume", "total_volume"))
    assert "TEST_SECRET" not in price.model_dump_json()
    assert calls == [("AAPL", "TEST_SECRET")]


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({}, (None, None, None)),
        ({"o": 0, "h": 0, "l": 0}, (None, None, None)),
        ({"o": 100.1, "l": 99}, (Decimal("100.1"), None, Decimal("99"))),
        ({"o": None, "h": 102, "l": 99}, (None, Decimal("102"), Decimal("99"))),
        ({"o": float("nan"), "h": 102, "l": 99}, (None, Decimal("102"), Decimal("99"))),
        ({"o": 100.1, "h": float("inf"), "l": 99}, (Decimal("100.1"), None, Decimal("99"))),
        ({"o": 100.1, "h": 102, "l": -1}, (Decimal("100.1"), Decimal("102"), None)),
        ({"o": True, "h": "102", "l": []}, (None, None, None)),
        ({"o": 10**400, "h": 102, "l": 99}, (None, Decimal("102"), Decimal("99"))),
        ({"o": 100.1, "h": 10**400, "l": 99}, (Decimal("100.1"), None, Decimal("99"))),
        ({"o": 100.1, "h": 102, "l": 10**400}, (Decimal("100.1"), Decimal("102"), None)),
    ],
)
def test_missing_or_unusable_session_fields_preserve_the_last_price(fields, expected):
    from tradecopilot.providers.finnhub import FinnhubClient

    now = datetime(2026, 10, 1, 20, tzinfo=UTC)
    instance = FinnhubClient(
        "TEST_SECRET",
        transport=lambda symbol, key: {"c": 101.25, "pc": 100, "t": int(now.timestamp()), **fields},
        clock=lambda: now,
    )

    price = instance.quote("AAPL")

    assert price.last == Decimal("101.25")
    assert (price.session_open, price.session_high, price.session_low) == expected


@pytest.mark.parametrize("fields", [{"h": 98}, {"l": 103}, {"o": 103}, {"o": 98}])
def test_contradictory_session_range_is_omitted_without_losing_the_last_price(fields):
    instance, _, _ = client(fields)

    price = instance.quote("AAPL")

    assert price.last == Decimal("101.25")
    assert (price.session_open, price.session_high, price.session_low) == (None, None, None)


@pytest.mark.parametrize(
    "fields",
    [
        {"session_open": Decimal("0")},
        {"session_high": Decimal("Infinity")},
        {"session_low": Decimal("NaN")},
        {"session_high": Decimal("98")},
        {"session_open": Decimal("103")},
    ],
)
def test_price_snapshot_rejects_invalid_session_prices(fields):
    instance, _, _ = client()
    price = instance.quote("AAPL")
    valid = {**price.model_dump(), "session_open": 100, "session_high": 102, "session_low": 99}
    assert PriceSnapshot.model_validate(valid)

    with pytest.raises(ValidationError):
        PriceSnapshot.model_validate({**valid, **fields})


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
    assert frame.price_history == (frame.price_snapshot,)
    without_history = frame.model_dump(exclude={"price_history"})
    assert MarketFrame.model_validate(without_history).price_history == ()
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


def scripted_provider(monkeypatch, samples):
    from tradecopilot.providers.finnhub import FinnhubClient, FinnhubFrameProvider

    now = datetime(2026, 10, 1, 20, tzinfo=UTC)
    pending = iter(samples)
    calls = []
    sleeps = []

    def transport(symbol, key):
        expected_symbol, seconds_ago, last = next(pending)
        assert symbol == expected_symbol
        calls.append(symbol)
        return {"c": last, "pc": 100, "t": int((now - timedelta(seconds=seconds_ago)).timestamp())}

    async def no_delay(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("tradecopilot.providers.finnhub.asyncio.sleep", no_delay)
    instance = FinnhubClient("TEST_SECRET", transport=transport, clock=lambda: now)
    provider = FinnhubFrameProvider("AAPL", "TEST_SECRET", client=instance)
    return provider, calls, sleeps


def test_history_deduplicates_unchanged_provider_timestamps(monkeypatch):
    provider, calls, sleeps = scripted_provider(monkeypatch, [("AAPL", 60, 101), ("AAPL", 60, 101.25)])

    async def twice():
        frames = provider.frames()
        try:
            return await anext(frames), await anext(frames)
        finally:
            await frames.aclose()

    first, second = asyncio.run(twice())

    assert first.price_history == (first.price_snapshot,)
    assert second.price_history == (second.price_snapshot,)
    assert second.price_snapshot.last == Decimal("101.25")
    assert calls == ["AAPL", "AAPL"] and sleeps == [5.0]


def test_history_keeps_distinct_observations_in_provider_time_order(monkeypatch):
    provider, _, _ = scripted_provider(monkeypatch, [("AAPL", 60, 100), ("AAPL", 50, 102), ("AAPL", 40, 101)])

    async def three():
        frames = provider.frames()
        try:
            await anext(frames)
            await anext(frames)
            return await anext(frames)
        finally:
            await frames.aclose()

    frame = asyncio.run(three())

    assert [price.last for price in frame.price_history] == [Decimal("100"), Decimal("102"), Decimal("101")]
    assert [price.provider_timestamp for price in frame.price_history] == sorted(
        price.provider_timestamp for price in frame.price_history
    )
    assert frame.price_history[-1] == frame.price_snapshot


def test_history_is_bounded_to_sixty_observations(monkeypatch):
    samples = [("AAPL", 120 - index, 100 + index) for index in range(65)]
    provider, _, _ = scripted_provider(monkeypatch, samples)

    async def collect():
        frames = provider.frames()
        try:
            return [await anext(frames) for _ in samples]
        finally:
            await frames.aclose()

    frames = asyncio.run(collect())
    frame = frames[-1]

    assert all(len(item.price_history) <= 60 for item in frames)
    assert len(frame.price_history) == 60
    assert frame.price_history[0].last == Decimal("105")
    assert frame.price_history[-1] == frame.price_snapshot
    with pytest.raises(ValidationError):
        MarketFrame.model_validate({**frame.model_dump(), "price_history": (frame.price_snapshot,) * 61})


def test_history_is_isolated_by_symbol_and_retained_when_switching_back(monkeypatch):
    provider, calls, _ = scripted_provider(monkeypatch, [("AAPL", 60, 101), ("MSFT", 50, 201), ("AAPL", 40, 102)])

    async def switch():
        frames = provider.frames()
        try:
            first = await anext(frames)
            provider.select_symbol("MSFT")
            second = await anext(frames)
            provider.select_symbol("AAPL")
            return first, second, await anext(frames)
        finally:
            await frames.aclose()

    first, second, third = asyncio.run(switch())

    assert first.price_history == (first.price_snapshot,)
    assert second.price_history == (second.price_snapshot,)
    assert third.price_history == (first.price_snapshot, third.price_snapshot)
    assert calls == ["AAPL", "MSFT", "AAPL"]
    assert all(price.symbol == "AAPL" for price in third.price_history)


def test_backwards_response_never_exposes_future_observations(monkeypatch):
    provider, _, _ = scripted_provider(monkeypatch, [("AAPL", 60, 100), ("AAPL", 40, 102), ("AAPL", 50, 101)])

    async def backwards():
        frames = provider.frames()
        try:
            first = await anext(frames)
            await anext(frames)
            return first, await anext(frames)
        finally:
            await frames.aclose()

    first, frame = asyncio.run(backwards())

    assert frame.price_history == (first.price_snapshot, frame.price_snapshot)
    assert all(price.provider_timestamp <= frame.price_snapshot.provider_timestamp for price in frame.price_history)


def test_response_older_than_the_retained_window_still_ends_its_history(monkeypatch):
    samples = [("AAPL", 60 - index, 100 + index) for index in range(60)] + [("AAPL", 120, 99)]
    provider, _, _ = scripted_provider(monkeypatch, samples)

    async def old():
        frames = provider.frames()
        try:
            for _ in range(60):
                await anext(frames)
            return await anext(frames)
        finally:
            await frames.aclose()

    frame = asyncio.run(old())

    assert frame.price_history == (frame.price_snapshot,)


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
