from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from datetime import UTC, datetime, timedelta
from itertools import pairwise

import pytest

from tradecopilot.forecast.contracts import ForecastConfig
from tradecopilot.providers.finnhub import FinnhubClient


class Clock:
    def __init__(self, now):
        self.now = now
        self.sleeps = []

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += timedelta(seconds=seconds)


def client_for(clock, calls, failures=0):
    def transport(symbol, key):
        calls.append((symbol, clock()))
        if len(calls) <= failures:
            raise ConnectionError("TEST_SECRET raw provider body")
        return {"c": 100, "pc": 100, "t": int(clock().timestamp())}
    return FinnhubClient("TEST_SECRET", transport=transport, clock=clock)


def test_collector_is_paced_resumable_price_only_and_sanitizes_retries(tmp_path):
    from tradecopilot.forecast.collector import collect_quotes
    from tradecopilot.forecast.data import ForecastStore

    clock = Clock(datetime(2026, 9, 1, 14, tzinfo=UTC))
    calls = []
    client = client_for(clock, calls, failures=2)
    with ForecastStore(tmp_path / "observations.sqlite3") as store:
        metrics = asyncio.run(collect_quotes(
            store, ForecastConfig(), cycles=2, client=client, clock=clock, sleep=clock.sleep,
        ))
        assert metrics["successes"] == 10 and metrics["retries"] == 2
        assert metrics["inserted"] == 10 and metrics["failures"] == 0
        assert all((right[1] - left[1]).total_seconds() >= 1.5 for left, right in pairwise(calls))
        assert (calls[7][1] - calls[0][1]).total_seconds() >= 15
        rows = store.observations()
        assert all(row.provenance == "market" and row.source == "finnhub_quote" for row in rows)
        assert not {"volume", "bid", "ask", "position"}.intersection(rows[0].model_dump())
        assert "TEST_SECRET" not in json.dumps(store.events())
    with ForecastStore(tmp_path / "observations.sqlite3") as resumed:
        assert len(resumed.observations()) == 10


def test_collector_stops_retrying_after_three_attempts_and_continues_symbols(tmp_path):
    from tradecopilot.forecast.collector import collect_quotes
    from tradecopilot.forecast.data import ForecastStore

    clock = Clock(datetime(2026, 9, 1, 14, tzinfo=UTC))
    calls = []
    with ForecastStore(tmp_path / "observations.sqlite3") as store:
        metrics = asyncio.run(collect_quotes(
            store, ForecastConfig(symbols=("AAPL", "MSFT")), cycles=1,
            client=client_for(clock, calls, failures=3), clock=clock, sleep=clock.sleep,
        ))
        assert [symbol for symbol, _ in calls] == ["AAPL", "AAPL", "AAPL", "MSFT"]
        assert metrics["failures"] == 1 and metrics["successes"] == 1
        assert [row.symbol for row in store.observations()] == ["MSFT"]


def test_collector_never_requests_closed_session_data(tmp_path):
    from tradecopilot.forecast.collector import collect_quotes
    from tradecopilot.forecast.data import ForecastStore

    clock = Clock(datetime(2026, 9, 5, 14, tzinfo=UTC))
    calls = []
    with ForecastStore(tmp_path / "observations.sqlite3") as store:
        metrics = asyncio.run(collect_quotes(
            store, ForecastConfig(), cycles=2, client=client_for(clock, calls), clock=clock, sleep=clock.sleep,
        ))
        assert calls == [] and store.observations() == []
        assert metrics["skipped_closed"] == 2


@pytest.mark.parametrize("interval", [0, 14.9, float("nan"), float("inf")])
def test_collector_rejects_unsafe_intervals(tmp_path, interval):
    from tradecopilot.forecast.collector import collect_quotes
    from tradecopilot.forecast.data import ForecastStore

    clock = Clock(datetime(2026, 9, 1, 14, tzinfo=UTC))
    with ForecastStore(tmp_path / "observations.sqlite3") as store, pytest.raises(ValueError, match="15"):
        asyncio.run(collect_quotes(
            store, ForecastConfig(), cycles=1, interval_seconds=interval,
            client=client_for(clock, []), clock=clock, sleep=clock.sleep,
        ))


def test_twenty_symbols_are_rate_limited_and_an_early_close_stops_requests(tmp_path):
    from tradecopilot.forecast.collector import collect_quotes
    from tradecopilot.forecast.data import ForecastStore

    clock = Clock(datetime(2026, 11, 27, 17, 59, 58, tzinfo=UTC))
    calls = []
    config = ForecastConfig(symbols=tuple(f"S{number}" for number in range(20)))
    with ForecastStore(tmp_path / "observations.sqlite3") as store:
        metrics = asyncio.run(collect_quotes(
            store, config, cycles=2, client=client_for(clock, calls), clock=clock, sleep=clock.sleep,
        ))
        assert metrics["interval_seconds"] == 30
        assert len(calls) == 2
        assert all(at < datetime(2026, 11, 27, 18, tzinfo=UTC) for _, at in calls)


def test_collection_rerun_deduplicates_the_same_received_observation(tmp_path):
    from tradecopilot.forecast.collector import collect_quotes
    from tradecopilot.forecast.data import ForecastStore

    clock = Clock(datetime(2026, 9, 1, 14, tzinfo=UTC))
    config = ForecastConfig(symbols=("AAPL",))
    database = tmp_path / "observations.sqlite3"
    # A replayed response carries its original receipt timestamp even when the next request waits.
    fixed_client = client_for(Clock(clock()), [])
    calls = []
    with ForecastStore(database) as store:
        first = asyncio.run(collect_quotes(
            store, config, cycles=1, client=client_for(clock, calls), clock=clock, sleep=clock.sleep,
        ))
    with ForecastStore(database) as store:
        second = asyncio.run(collect_quotes(
            store, config, cycles=1, client=fixed_client, clock=clock, sleep=clock.sleep,
        ))
        assert first["inserted"] == 1 and second["inserted"] == 0 and second["duplicates"] == 1
        assert len(store.observations()) == 1
        assert (clock() - calls[0][1]).total_seconds() >= 1.5


def test_permanent_provider_error_is_not_retried(tmp_path):
    from tradecopilot.forecast.collector import collect_quotes
    from tradecopilot.forecast.data import ForecastStore

    clock = Clock(datetime(2026, 9, 1, 14, tzinfo=UTC))
    client = FinnhubClient("TEST_SECRET", transport=lambda *_: {"error": "invalid symbol"}, clock=clock)
    with ForecastStore(tmp_path / "observations.sqlite3") as store:
        result = asyncio.run(collect_quotes(
            store, ForecastConfig(symbols=("AAPL",)), cycles=1, client=client, clock=clock, sleep=clock.sleep,
        ))
        assert result["attempts"] == 1 and result["retries"] == 0
        assert store.events()[0]["payload"]["reason"] == "invalid_quote"


def test_late_sleep_does_not_catch_up_with_an_early_request(tmp_path):
    from tradecopilot.forecast.collector import collect_quotes
    from tradecopilot.forecast.data import ForecastStore

    clock = Clock(datetime(2026, 9, 1, 14, tzinfo=UTC))
    calls = []

    async def late_sleep(seconds):
        await clock.sleep(seconds + (0.2 if not clock.sleeps else 0))

    with ForecastStore(tmp_path / "observations.sqlite3") as store:
        asyncio.run(collect_quotes(
            store, ForecastConfig(), cycles=1, client=client_for(clock, calls), clock=clock, sleep=late_sleep,
        ))
    assert all((right[1] - left[1]).total_seconds() >= 1.5 for left, right in pairwise(calls))


def test_request_gate_samples_time_after_waiting_for_sqlite_writer(tmp_path):
    from tradecopilot.forecast.data import ForecastStore

    database = tmp_path / "observations.sqlite3"
    clock = Clock(datetime(2026, 9, 1, 14, tzinfo=UTC))
    with ForecastStore(database) as store:
        blocker = sqlite3.connect(database, check_same_thread=False)
        blocker.execute("BEGIN IMMEDIATE")

        def release():
            time.sleep(0.05)
            clock.now += timedelta(seconds=2)
            blocker.commit()

        thread = threading.Thread(target=release)
        thread.start()
        try:
            assert store.reserve_collection_slot(clock) == 0
            assert store.reserve_collection_slot(clock) == pytest.approx(1.5)
        finally:
            thread.join()
            blocker.close()
