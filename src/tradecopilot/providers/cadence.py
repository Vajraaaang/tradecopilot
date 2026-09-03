from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

from tradecopilot.config import StrategyConfig
from tradecopilot.models import AccountRiskSnapshot, Level2Snapshot, OHLCVBar, PositionSnapshot, Quote
from tradecopilot.providers.base import BrokerReadProvider, MarketDataProvider


@dataclass
class CadencedMarketDataProvider:
    """Cache read results at configured provider-safe cadences without overlapping calls."""

    inner: MarketDataProvider
    config: StrategyConfig
    monotonic_clock: Callable[[], float] = time.monotonic
    _cache: dict[str, tuple[float, Any]] = field(default_factory=dict)
    _locks: dict[str, asyncio.Lock] = field(default_factory=dict)

    async def get_quote(self, symbol: str) -> Quote:
        key = f"quote:{symbol.upper()}"
        return cast(
            Quote,
            await self._read(key, self.config.selected_quote_interval_seconds, lambda: self.inner.get_quote(symbol)),
        )

    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> Sequence[OHLCVBar]:
        interval = (
            self.config.one_minute_bars_interval_seconds
            if timeframe == "1m"
            else self.config.five_minute_bars_interval_seconds
        )
        key = f"bars:{symbol.upper()}:{timeframe}"
        return cast(
            Sequence[OHLCVBar],
            await self._read(key, interval, lambda: self.inner.get_bars(symbol, timeframe, start, end)),
        )

    async def get_level2(self, symbol: str) -> Level2Snapshot:
        key = f"level2:{symbol.upper()}"
        return cast(
            Level2Snapshot,
            await self._read(key, self.config.level2_interval_seconds, lambda: self.inner.get_level2(symbol)),
        )

    async def is_tradable(self, symbol: str) -> bool:
        key = f"tradable:{symbol.upper()}"
        return cast(bool, await self._read(key, 5.0, lambda: self.inner.is_tradable(symbol)))

    async def _read(self, key: str, interval: float, reader: Callable[[], Awaitable[Any]]) -> Any:
        now = self.monotonic_clock()
        cached = self._cache.get(key)
        if cached is not None and now - cached[0] < interval:
            return cached[1]
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            now = self.monotonic_clock()
            cached = self._cache.get(key)
            if cached is not None and now - cached[0] < interval:
                return cached[1]
            value = await reader()
            self._cache[key] = (now, value)
            return value


@dataclass
class CadencedBrokerReadProvider:
    inner: BrokerReadProvider
    config: StrategyConfig
    monotonic_clock: Callable[[], float] = time.monotonic
    _cache: dict[str, tuple[float, Any]] = field(default_factory=dict)
    _locks: dict[str, asyncio.Lock] = field(default_factory=dict)

    async def get_positions(self) -> Sequence[PositionSnapshot]:
        reader = getattr(self.inner, "get_positions", None)
        if not callable(reader):
            raise NotImplementedError("broker provider does not expose all positions")
        return cast(
            Sequence[PositionSnapshot],
            await self._read("positions", self.config.position_interval_seconds, reader),
        )

    async def get_position(self, symbol: str) -> PositionSnapshot | None:
        return cast(
            PositionSnapshot | None,
            await self._read(
                f"position:{symbol.upper()}",
                self.config.position_interval_seconds,
                lambda: self.inner.get_position(symbol),
            ),
        )

    async def get_account_risk(self) -> AccountRiskSnapshot:
        return cast(
            AccountRiskSnapshot,
            await self._read("account-risk", self.config.account_pnl_interval_seconds, self.inner.get_account_risk),
        )

    async def get_trade_history(self) -> Sequence[dict[str, object]]:
        return await self.inner.get_trade_history()

    async def get_open_orders(self, symbol: str) -> Sequence[dict[str, object]]:
        return await self.inner.get_open_orders(symbol)

    async def _read(self, key: str, interval: float, reader: Callable[[], Awaitable[Any]]) -> Any:
        now = self.monotonic_clock()
        cached = self._cache.get(key)
        if cached is not None and now - cached[0] < interval:
            return cached[1]
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            now = self.monotonic_clock()
            cached = self._cache.get(key)
            if cached is not None and now - cached[0] < interval:
                return cached[1]
            value = await reader()
            self._cache[key] = (now, value)
            return value
