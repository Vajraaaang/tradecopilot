from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from tradecopilot.config import StrategyConfig
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
    RunMode,
    StampedModel,
    TimeAndSalesPrint,
)
from tradecopilot.providers.base import (
    BrokerReadProvider,
    MarketDataProvider,
    SupplementalDataProvider,
    TimeAndSalesProvider,
)


@dataclass
class PollingFrameProvider:
    """Compose normalized read providers into an always-on, fail-closed frame stream."""

    symbol: str
    market: MarketDataProvider
    broker: BrokerReadProvider
    supplemental: SupplementalDataProvider
    tape: TimeAndSalesProvider
    config: StrategyConfig
    mode: RunMode = RunMode.LIVE
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    _last_quote: Quote | None = None
    _last_bars_1m: tuple[OHLCVBar, ...] = ()
    _last_bars_5m: tuple[OHLCVBar, ...] = ()
    _level2_history: list[Level2Snapshot] = field(default_factory=list)
    _last_float: FloatEvidence | None = None
    _last_catalyst: CatalystEvidence | None = None
    _last_historical_context: HistoricalContextEvidence | None = None

    def select_symbol(self, symbol: str) -> None:
        normalized = symbol.strip().upper()
        valid = (
            normalized
            and len(normalized) <= 10
            and all(character.isalnum() or character in ".-" for character in normalized)
        )
        if not valid:
            raise ValueError("unsupported stock symbol format")
        if normalized == self.symbol:
            return
        self.symbol = normalized
        self._last_quote = None
        self._last_bars_1m = ()
        self._last_bars_5m = ()
        self._level2_history.clear()
        self._last_float = None
        self._last_catalyst = None
        self._last_historical_context = None

    async def frames(self) -> AsyncIterator[MarketFrame]:
        while True:
            started = asyncio.get_running_loop().time()
            yield await self.snapshot()
            elapsed = asyncio.get_running_loop().time() - started
            await self.sleep(max(0.0, self.config.poll_interval_seconds - elapsed))

    async def snapshot(self) -> MarketFrame:
        requested_at = self.clock()
        if requested_at.tzinfo is None or requested_at.utcoffset() is None:
            raise ValueError("polling clock must return a timezone-aware datetime")
        start_1m = requested_at - timedelta(hours=8)
        start_5m = requested_at - timedelta(days=5)
        quote_task: asyncio.Task[Quote | None] = asyncio.create_task(_optional(self.market.get_quote(self.symbol)))
        bars_1m_task: asyncio.Task[tuple[OHLCVBar, ...]] = asyncio.create_task(
            _sequence(self.market.get_bars(self.symbol, "1m", start_1m, requested_at))
        )
        bars_5m_task: asyncio.Task[tuple[OHLCVBar, ...]] = asyncio.create_task(
            _sequence(self.market.get_bars(self.symbol, "5m", start_5m, requested_at))
        )
        level2_task: asyncio.Task[Level2Snapshot | None] = asyncio.create_task(
            _optional(self.market.get_level2(self.symbol))
        )
        tradable_task: asyncio.Task[bool | None] = asyncio.create_task(_optional(self.market.is_tradable(self.symbol)))
        positions_reader = getattr(self.broker, "get_positions", None)
        positions_task: asyncio.Task[tuple[bool, tuple[PositionSnapshot, ...]]] | None = None
        position_task: asyncio.Task[tuple[bool, PositionSnapshot | None]] | None = None
        if callable(positions_reader):
            positions_task = asyncio.create_task(_known_sequence(positions_reader()))
        else:
            position_task = asyncio.create_task(_known_optional(self.broker.get_position(self.symbol)))
        risk_task: asyncio.Task[AccountRiskSnapshot | None] = asyncio.create_task(
            _optional(self.broker.get_account_risk())
        )
        float_task: asyncio.Task[FloatEvidence | None] = asyncio.create_task(
            _optional(self.supplemental.get_float(self.symbol))
        )
        catalyst_task: asyncio.Task[CatalystEvidence | None] = asyncio.create_task(
            _optional(self.supplemental.get_catalyst(self.symbol))
        )
        context_reader = getattr(self.supplemental, "get_historical_context", None)
        context_task: asyncio.Task[HistoricalContextEvidence | None]
        if callable(context_reader):
            context_task = asyncio.create_task(_optional(context_reader(self.symbol)))
        else:
            context_task = asyncio.create_task(asyncio.sleep(0, result=None))
        tape_task: asyncio.Task[Sequence[TimeAndSalesPrint] | None] = asyncio.create_task(
            _optional(self.tape.get_prints(self.symbol))
        )
        quote = await quote_task
        bars_1m = await bars_1m_task
        bars_5m = await bars_5m_task
        level2 = await level2_task
        tradable = await tradable_task
        if positions_task is not None:
            positions_known, positions = await positions_task
            position = next((item for item in positions if item.symbol == self.symbol), None)
            position_result = (positions_known, position)
        else:
            assert position_task is not None
            position_result = await position_task
            positions = (position_result[1],) if position_result[1] is not None else ()
        account_risk = await risk_task
        float_evidence = await float_task
        catalyst = await catalyst_task
        historical_context = await context_task
        tape = await tape_task
        received_at = self.clock()

        if quote is not None:
            self._last_quote = quote
        current_quote = _refresh_optional(self._last_quote, received_at)
        if bars_1m:
            self._last_bars_1m = _dedupe_bars(bars_1m)
        if bars_5m:
            self._last_bars_5m = _dedupe_bars(bars_5m)
        if current_quote is not None and self._last_bars_1m:
            current_quote = current_quote.model_copy(update={"current_minute_volume": self._last_bars_1m[-1].volume})
        if level2 is not None:
            signature = (
                level2.provider_timestamp,
                tuple((level.price, level.size) for level in level2.bids),
                tuple((level.price, level.size) for level in level2.asks),
            )
            latest_signature = None
            if self._level2_history:
                latest = self._level2_history[-1]
                latest_signature = (
                    latest.provider_timestamp,
                    tuple((item.price, item.size) for item in latest.bids),
                    tuple((item.price, item.size) for item in latest.asks),
                )
            if signature != latest_signature:
                self._level2_history.append(level2)
                self._level2_history = self._level2_history[-20:]
        if float_evidence is not None:
            self._last_float = float_evidence
        if catalyst is not None:
            self._last_catalyst = catalyst
        if historical_context is not None:
            self._last_historical_context = historical_context
        if current_quote is not None and self._last_historical_context is not None:
            current_quote = current_quote.model_copy(
                update={"average_daily_volume_50d": self._last_historical_context.average_daily_volume_50d}
            )

        position_known, position = position_result
        if position is not None and current_quote is not None and position.average_entry is not None:
            unrealized = (current_quote.last - position.average_entry) * position.quantity
            position = position.model_copy(update={"unrealized_pnl": unrealized})
            positions = tuple(position if item.symbol == position.symbol else item for item in positions)
            if account_risk is not None:
                account_risk = account_risk.model_copy(update={"unrealized_pnl": unrealized})
        if not position_known:
            account_risk = None

        resistance_levels = _resistance_levels(self._last_bars_5m, current_quote)
        gap = None
        if current_quote is not None:
            gap = (
                (current_quote.session_origin_price - current_quote.previous_close) / current_quote.previous_close
            ) * Decimal(100)
        return MarketFrame(
            event_time=received_at,
            mode=self.mode,
            quote=current_quote,
            bars_1m=tuple(_refresh(item, received_at) for item in self._last_bars_1m),
            bars_5m=tuple(_refresh(item, received_at) for item in self._last_bars_5m),
            level2_history=tuple(_refresh(item, received_at) for item in self._level2_history),
            time_and_sales=tuple(tape) if tape is not None else None,
            position=position,
            account_risk=account_risk,
            float_evidence=_refresh_optional(self._last_float, received_at),
            catalyst_evidence=_refresh_optional(self._last_catalyst, received_at),
            resistance_levels=resistance_levels,
            gap_percent=gap,
            market_leader=False,
            no_a_quality_candidates=False,
            bearish_momentum_environment=False,
            tradability_known=tradable is not None,
            halted=tradable is False,
            historical_context=_refresh_optional(self._last_historical_context, received_at),
            positions=tuple(_refresh(item, received_at) for item in positions),
        )


async def _optional[T](awaitable: Awaitable[T]) -> T | None:
    try:
        return await awaitable
    except Exception:
        return None


async def _known_optional[T](awaitable: Awaitable[T | None]) -> tuple[bool, T | None]:
    try:
        return True, await awaitable
    except Exception:
        return False, None


async def _known_sequence[T](awaitable: Awaitable[Sequence[T]]) -> tuple[bool, tuple[T, ...]]:
    try:
        return True, tuple(await awaitable)
    except Exception:
        return False, ()


async def _sequence[T](awaitable: Awaitable[Sequence[T]]) -> tuple[T, ...]:
    try:
        return tuple(await awaitable)
    except Exception:
        return ()


def _refresh[StampedT: StampedModel](model: StampedT, receipt_timestamp: datetime) -> StampedT:
    provider_timestamp = model.provider_timestamp
    age_seconds = max(0.0, (receipt_timestamp - provider_timestamp).total_seconds())
    return model.model_copy(update={"receipt_timestamp": receipt_timestamp, "age_seconds": age_seconds})


def _refresh_optional[StampedT: StampedModel](
    model: StampedT | None,
    receipt_timestamp: datetime,
) -> StampedT | None:
    return _refresh(model, receipt_timestamp) if model is not None else None


def _dedupe_bars(bars: Sequence[OHLCVBar]) -> tuple[OHLCVBar, ...]:
    indexed = {(bar.timeframe, bar.provider_timestamp): bar for bar in bars}
    return tuple(sorted(indexed.values(), key=lambda bar: bar.provider_timestamp))


def _resistance_levels(bars: Sequence[OHLCVBar], quote: Quote | None) -> tuple[Decimal, ...]:
    if quote is None:
        return ()
    values = sorted({bar.high for bar in bars if bar.high > quote.last})
    return tuple(values[:5])
