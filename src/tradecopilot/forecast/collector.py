"""Bounded, read-only collection using the existing price-only Finnhub client."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from time import perf_counter
from typing import Any

from tradecopilot.forecast.contracts import ForecastConfig, Observation, utc
from tradecopilot.forecast.data import ForecastStore
from tradecopilot.forecast.sessions import session_bounds, session_for
from tradecopilot.providers.finnhub import FinnhubClient, FinnhubQuoteError

_REQUEST_SPACING_SECONDS = 1.5  # At most 40 requests/minute, retries included.


def _is_open(at: datetime) -> bool:
    day = session_for(at)
    bounds = session_bounds(day) if day is not None else None
    return bounds is not None and at < bounds[1]


async def collect_quotes(
    store: ForecastStore,
    config: ForecastConfig,
    *,
    cycles: int,
    interval_seconds: float = 15,
    client: FinnhubClient,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    if not math.isfinite(interval_seconds) or interval_seconds < 15:
        raise ValueError("collection interval must be at least 15 seconds")
    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 1:
        raise ValueError("cycles must be a positive integer")
    clock = clock or (lambda: datetime.now(UTC))
    interval_seconds = max(interval_seconds, len(config.symbols) * _REQUEST_SPACING_SECONDS)
    metrics: dict[str, Any] = {
        "cycles": cycles, "attempts": 0, "successes": 0, "failures": 0, "retries": 0,
        "inserted": 0, "duplicates": 0, "skipped_closed": 0, "stale_quotes": 0,
        "interval_seconds": interval_seconds, "max_source_age_seconds": 0.0,
        "request_latency_ms": 0.0,
    }
    for cycle in range(cycles):
        cycle_started = utc(clock())
        if not _is_open(cycle_started):
            metrics["skipped_closed"] += 1
            store.record_event("collection_skipped", {"cycle": cycle, "reason": "session_closed"})
        else:
            for symbol in config.symbols:
                for attempt in range(1, 4):
                    while (remaining := store.reserve_collection_slot(clock, _REQUEST_SPACING_SECONDS)) > 0:
                        await sleep(remaining)
                    request_time = utc(clock())
                    if not _is_open(request_time):
                        metrics["skipped_closed"] += 1
                        store.record_event("collection_skipped", {"symbol": symbol, "reason": "session_closed"})
                        break
                    metrics["attempts"] += 1
                    if attempt > 1:
                        metrics["retries"] += 1
                    started = perf_counter()
                    try:
                        quote = await asyncio.to_thread(client.quote, symbol)
                        observation = Observation.model_validate({**quote.model_dump(), "provenance": "market"})
                        if observation.symbol != symbol:
                            raise ValueError("symbol mismatch")
                    except (ConnectionError, ValueError) as error:
                        latency_ms = max(0.0, (perf_counter() - started) * 1000)
                        metrics["request_latency_ms"] += latency_ms
                        transient = isinstance(error, ConnectionError) and (
                            not isinstance(error, FinnhubQuoteError) or error.retryable
                        )
                        retry = transient and attempt < 3
                        store.record_event("quote_failure", {
                            "symbol": symbol, "attempt": attempt, "latency_ms": latency_ms,
                            "reason": "quote_unavailable" if transient else "invalid_quote",
                            "retry": retry,
                        })
                        if not retry:
                            metrics["failures"] += 1
                            break
                        continue
                    latency_ms = max(0.0, (perf_counter() - started) * 1000)
                    inserted = store.ingest([observation])
                    metrics["request_latency_ms"] += latency_ms
                    metrics["successes"] += 1
                    metrics["inserted"] += inserted
                    metrics["duplicates"] += 1 - inserted
                    metrics["max_source_age_seconds"] = max(
                        metrics["max_source_age_seconds"], observation.age_seconds,
                    )
                    metrics["stale_quotes"] += int(observation.age_seconds > config.max_source_age_seconds)
                    store.record_event("quote_received", {
                        "symbol": symbol, "attempt": attempt, "latency_ms": latency_ms,
                        "source_age_seconds": observation.age_seconds, "inserted": inserted,
                    })
                    break
        if cycle + 1 < cycles:
            next_cycle = cycle_started + timedelta(seconds=interval_seconds)
            remaining = (next_cycle - utc(clock())).total_seconds()
            if remaining > 0:
                await sleep(remaining)
    metrics["mean_latency_ms"] = metrics["request_latency_ms"] / max(1, metrics["attempts"])
    return metrics
