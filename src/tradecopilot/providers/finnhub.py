"""Finnhub's free quote endpoint, explicitly limited to price display."""

from __future__ import annotations

import asyncio
import http.client
import json
import math
import re
import threading
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from urllib.parse import urlencode

from tradecopilot.models import DataQuality, MarketFrame, PriceSnapshot, RunMode

_SYMBOL = re.compile(r"[A-Z][A-Z0-9.-]{0,9}")


def _normalize_symbol(value: str) -> str:
    symbol = value.strip().upper()
    if not _SYMBOL.fullmatch(symbol):
        raise ValueError("Use a valid US stock symbol")
    return symbol


def _get_quote(symbol: str, api_key: str) -> dict[str, Any]:
    connection = http.client.HTTPSConnection("finnhub.io", timeout=5)
    try:
        connection.request(
            "GET",
            "/api/v1/quote?" + urlencode({"symbol": symbol}),
            headers={"X-Finnhub-Token": api_key, "Accept": "application/json"},
        )
        response = connection.getresponse()
        body = response.read(16_385)
        if response.status != 200:
            raise ConnectionError(f"Finnhub quote request failed (HTTP {response.status}); no retry was made")
        if len(body) > 16_384:
            raise ConnectionError("Finnhub quote response exceeded the size limit")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ConnectionError("Finnhub quote response was invalid")
        return payload
    except (OSError, http.client.HTTPException, ValueError):
        raise ConnectionError("Finnhub quote is unavailable; check the key, symbol and provider status") from None
    finally:
        connection.close()


class FinnhubClient:
    def __init__(
        self,
        api_key: str,
        *,
        transport: Callable[[str, str], dict[str, Any]] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Configure Finnhub using `tradecopilot auth finnhub` or FINNHUB_API_KEY")
        self._api_key = api_key.strip()
        self._transport = transport or _get_quote
        self._clock = clock or (lambda: datetime.now(UTC))

    def quote(self, symbol: str) -> PriceSnapshot:
        symbol = _normalize_symbol(symbol)
        try:
            payload = self._transport(symbol, self._api_key)
            if payload.get("error"):
                raise ValueError("Provider error")
            for key in ("c", "pc", "t"):
                value = payload[key]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError("Invalid quote field")
            provider_time = datetime.fromtimestamp(payload["t"], UTC)
            receipt_time = self._clock()
            age = (receipt_time - provider_time).total_seconds()
            return PriceSnapshot(
                symbol=symbol,
                last=Decimal(str(payload["c"])),
                previous_close=Decimal(str(payload["pc"])),
                provider_timestamp=provider_time,
                receipt_timestamp=receipt_time,
                age_seconds=age,
                source="finnhub_quote",
                quality=DataQuality.STALE if age > 2 else DataQuality.LIMITED,
            )
        except Exception:
            # Neither raw provider bodies nor exception text may disclose the supplied API key.
            raise ConnectionError("Finnhub did not provide a valid price and timestamp; no retry was made") from None


class FinnhubFrameProvider:
    def __init__(
        self,
        symbol: str,
        api_key: str,
        *,
        client: FinnhubClient | None = None,
        poll_interval_seconds: float = 5.0,
    ) -> None:
        if not math.isfinite(poll_interval_seconds) or poll_interval_seconds < 5:
            raise ValueError("Finnhub refresh interval must be at least 5 seconds")
        self._symbol = _normalize_symbol(symbol)
        self._symbol_lock = threading.Lock()
        self._client = client or FinnhubClient(api_key)
        self._interval = poll_interval_seconds

    def select_symbol(self, symbol: str) -> bool:
        normalized = _normalize_symbol(symbol)
        with self._symbol_lock:
            self._symbol = normalized
        return True

    def _current_symbol(self) -> str:
        with self._symbol_lock:
            return self._symbol

    async def frames(self) -> AsyncIterator[MarketFrame]:
        while True:
            symbol = self._current_symbol()
            price = await asyncio.to_thread(self._client.quote, symbol)
            if symbol == self._current_symbol():
                yield MarketFrame(
                    event_time=price.receipt_timestamp,
                    mode=RunMode.LIVE,
                    quote=None,
                    price_snapshot=price,
                    bars_1m=(),
                    bars_5m=(),
                    level2_history=(),
                    time_and_sales=None,
                    position=None,
                    account_risk=None,
                    float_evidence=None,
                    catalyst_evidence=None,
                    resistance_levels=(),
                    gap_percent=None,
                    market_leader=False,
                    no_a_quality_candidates=False,
                    bearish_momentum_environment=False,
                    tradability_known=False,
                    halted=False,
                )
            # At most 12 REST requests/minute in this process, including symbol changes; no retry burst.
            await asyncio.sleep(self._interval)
