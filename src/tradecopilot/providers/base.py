from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import datetime
from typing import Protocol

from tradecopilot.models import (
    AccountRiskSnapshot,
    CatalystEvidence,
    FloatEvidence,
    HistoricalContextEvidence,
    Level2Snapshot,
    MarketFrame,
    OHLCVBar,
    PositionSnapshot,
    Quote,
    TimeAndSalesPrint,
)


class MarketDataProvider(Protocol):
    async def get_quote(self, symbol: str) -> Quote: ...

    async def get_bars(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> Sequence[OHLCVBar]: ...

    async def get_level2(self, symbol: str) -> Level2Snapshot: ...

    async def is_tradable(self, symbol: str) -> bool: ...


class BrokerReadProvider(Protocol):
    async def get_positions(self) -> Sequence[PositionSnapshot]: ...

    async def get_position(self, symbol: str) -> PositionSnapshot | None: ...

    async def get_account_risk(self) -> AccountRiskSnapshot: ...

    async def get_trade_history(self) -> Sequence[dict[str, object]]: ...

    async def get_open_orders(self, symbol: str) -> Sequence[dict[str, object]]: ...


class SupplementalDataProvider(Protocol):
    async def get_float(self, symbol: str) -> FloatEvidence | None: ...

    async def get_catalyst(self, symbol: str) -> CatalystEvidence | None: ...

    async def get_historical_context(self, symbol: str) -> HistoricalContextEvidence | None: ...


class TimeAndSalesProvider(Protocol):
    async def get_prints(self, symbol: str) -> Sequence[TimeAndSalesPrint] | None: ...


class FrameProvider(Protocol):
    def frames(self) -> AsyncIterator[MarketFrame]: ...
