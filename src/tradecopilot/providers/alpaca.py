from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from tradecopilot.models import DataQuality, OHLCVBar, Quote, TimeAndSalesPrint

EASTERN = ZoneInfo("America/New_York")


@dataclass
class AlpacaStreamingProvider:
    """Read-only Alpaca stream for trades, NBBO quotes, and minute bars."""

    api_key: str
    secret_key: str
    feed_name: str = "sip"
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    maximum_tape_prints: int = 2_000
    _symbols: set[str] = field(default_factory=set, init=False)
    _quotes: dict[str, Any] = field(default_factory=dict, init=False)
    _trades: dict[str, Any] = field(default_factory=dict, init=False)
    _bars: dict[str, list[Any]] = field(default_factory=dict, init=False)
    _tape: dict[str, list[Any]] = field(default_factory=dict, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    _stream: StockDataStream | None = field(default=None, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _historical: StockHistoricalDataClient | None = field(default=None, init=False)
    _bootstrapped: set[str] = field(default_factory=set, init=False)
    _previous_closes: dict[tuple[str, date], Decimal] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.feed_name not in {"iex", "sip"}:
            raise ValueError("Alpaca feed must be iex or sip")
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca market-data credentials are required")

    async def subscribe(self, symbols: Sequence[str]) -> None:
        normalized = tuple(sorted({_symbol(value) for value in symbols}))
        if not normalized:
            raise ValueError("at least one Alpaca symbol is required")
        with self._lock:
            new_symbols = tuple(symbol for symbol in normalized if symbol not in self._symbols)
            if not new_symbols:
                return
            self._symbols.update(new_symbols)
            if self._stream is None:
                feed = DataFeed.SIP if self.feed_name == "sip" else DataFeed.IEX
                self._stream = StockDataStream(self.api_key, self.secret_key, feed=feed)
                self._historical = StockHistoricalDataClient(self.api_key, self.secret_key)
                self._stream.subscribe_quotes(self._on_quote, *normalized)
                self._stream.subscribe_trades(self._on_trade, *normalized)
                self._stream.subscribe_bars(self._on_bar, *normalized)
                self._stream.subscribe_updated_bars(self._on_bar, *normalized)
                self._thread = threading.Thread(target=self._stream.run, name="tradecopilot-alpaca", daemon=True)
                self._thread.start()
                return
            self._stream.subscribe_quotes(self._on_quote, *new_symbols)
            self._stream.subscribe_trades(self._on_trade, *new_symbols)
            self._stream.subscribe_bars(self._on_bar, *new_symbols)
            self._stream.subscribe_updated_bars(self._on_bar, *new_symbols)

    async def close(self) -> None:
        with self._lock:
            stream = self._stream
            thread = self._thread
            self._stream = None
            self._thread = None
        if stream is not None:
            await asyncio.to_thread(stream.stop)
        if thread is not None:
            await asyncio.to_thread(thread.join, 5)

    async def get_quote(self, symbol: str) -> Quote:
        normalized = _symbol(symbol)
        await self.subscribe((normalized,))
        received_at = self.clock()
        await self._bootstrap_bars(normalized, received_at - timedelta(hours=16), received_at)
        with self._lock:
            raw_quote = self._quotes.get(normalized)
            raw_trade = self._trades.get(normalized)
            bars = tuple(self._bars.get(normalized, ()))
        if raw_quote is None or raw_trade is None:
            raise ConnectionError(f"Alpaca stream has not received a complete quote for {normalized}")
        quote_timestamp = _timestamp(raw_quote.timestamp)
        trade_timestamp = _timestamp(raw_trade.timestamp)
        # Quote and last trade are both required. Age the composite from the
        # older component so a fresh trade cannot hide a stale spread (or vice versa).
        provider_timestamp = min(quote_timestamp, trade_timestamp)
        session_bars = [bar for bar in bars if _timestamp(bar.timestamp).date() == provider_timestamp.date()]
        previous_close = await self._previous_close(normalized, provider_timestamp)
        session_origin = Decimal(str(session_bars[0].open)) if session_bars else Decimal(str(raw_trade.price))
        total_volume = sum(int(bar.volume) for bar in session_bars)
        current_volume = int(session_bars[-1].volume) if session_bars else 0
        return Quote(
            provider_timestamp=provider_timestamp,
            receipt_timestamp=received_at,
            age_seconds=max(0.0, (received_at - provider_timestamp).total_seconds()),
            source=f"alpaca_stream:{self.feed_name}",
            quality=DataQuality.GOOD,
            symbol=normalized,
            bid=_positive(raw_quote.bid_price, "bid price"),
            ask=_positive(raw_quote.ask_price, "ask price"),
            last=_positive(raw_trade.price, "trade price"),
            previous_close=previous_close,
            total_volume=total_volume,
            current_minute_volume=current_volume,
            session_origin_price=session_origin,
            average_daily_volume_50d=None,
            average_cumulative_volume_same_time=None,
        )

    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> Sequence[OHLCVBar]:
        normalized = _symbol(symbol)
        await self.subscribe((normalized,))
        await self._bootstrap_bars(normalized, start, end)
        with self._lock:
            raw_bars = tuple(self._bars.get(normalized, ()))
        one_minute = tuple(
            _normalize_bar(bar, "1m", self.feed_name, self.clock())
            for bar in raw_bars
            if start <= _timestamp(bar.timestamp) <= end
        )
        if timeframe == "1m":
            return _dedupe(one_minute)
        if timeframe == "5m":
            return _aggregate_five_minute(_dedupe(one_minute))
        raise ValueError(f"unsupported Alpaca timeframe: {timeframe}")

    async def get_prints(self, symbol: str) -> Sequence[TimeAndSalesPrint] | None:
        normalized = _symbol(symbol)
        await self.subscribe((normalized,))
        received_at = self.clock()
        with self._lock:
            trades = tuple(self._tape.get(normalized, ()))
        return tuple(
            TimeAndSalesPrint(
                provider_timestamp=_timestamp(trade.timestamp),
                receipt_timestamp=received_at,
                age_seconds=max(0.0, (received_at - _timestamp(trade.timestamp)).total_seconds()),
                source=f"alpaca_stream:{self.feed_name}",
                quality=DataQuality.GOOD,
                symbol=normalized,
                price=_positive(trade.price, "trade price"),
                size=int(trade.size),
                side="unknown",
            )
            for trade in trades
        )

    async def _bootstrap_bars(self, symbol: str, start: datetime, end: datetime) -> None:
        with self._lock:
            if symbol in self._bootstrapped:
                return
            client = self._historical
        if client is None:
            raise ConnectionError("Alpaca historical client is unavailable")
        feed = DataFeed.SIP if self.feed_name == "sip" else DataFeed.IEX
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            start=start,
            end=end,
            timeframe=TimeFrame.Minute,
            feed=feed,
        )
        result = await asyncio.to_thread(client.get_stock_bars, request)
        data = getattr(result, "data", {})
        rows = data.get(symbol, []) if isinstance(data, dict) else []
        with self._lock:
            streamed = self._bars.get(symbol, [])
            indexed = {_timestamp(bar.timestamp): bar for bar in (*rows, *streamed)}
            self._bars[symbol] = [indexed[key] for key in sorted(indexed)]
            self._bootstrapped.add(symbol)

    async def _previous_close(self, symbol: str, current_time: datetime) -> Decimal:
        cache_key = (symbol, current_time.date())
        with self._lock:
            cached = self._previous_closes.get(cache_key)
        if cached is not None:
            return cached
        client = self._historical
        if client is None:
            raise ConnectionError("Alpaca historical client is unavailable")
        feed = DataFeed.SIP if self.feed_name == "sip" else DataFeed.IEX
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            start=current_time - timedelta(days=10),
            end=current_time,
            timeframe=TimeFrame.Day,
            limit=5,
            feed=feed,
        )
        result = await asyncio.to_thread(client.get_stock_bars, request)
        data = getattr(result, "data", {})
        rows = data.get(symbol, []) if isinstance(data, dict) else []
        completed = [bar for bar in rows if _timestamp(bar.timestamp).date() < current_time.date()]
        if not completed:
            raise ConnectionError(f"Alpaca returned no completed daily bar for {symbol}")
        close = _positive(completed[-1].close, "previous close")
        with self._lock:
            self._previous_closes[cache_key] = close
        return close

    async def _on_quote(self, quote: Any) -> None:
        with self._lock:
            self._quotes[_symbol(str(quote.symbol))] = quote

    async def _on_trade(self, trade: Any) -> None:
        symbol = _symbol(str(trade.symbol))
        with self._lock:
            self._trades[symbol] = trade
            tape = self._tape.setdefault(symbol, [])
            tape.append(trade)
            del tape[: max(0, len(tape) - self.maximum_tape_prints)]

    async def _on_bar(self, bar: Any) -> None:
        symbol = _symbol(str(bar.symbol))
        timestamp = _timestamp(bar.timestamp)
        with self._lock:
            bars = self._bars.setdefault(symbol, [])
            for index, prior in enumerate(bars):
                if _timestamp(prior.timestamp) == timestamp:
                    bars[index] = bar
                    break
            else:
                bars.append(bar)
            bars.sort(key=lambda item: _timestamp(item.timestamp))
            del bars[: max(0, len(bars) - 2_000)]


def _normalize_bar(raw: Any, timeframe: str, feed: str, received_at: datetime) -> OHLCVBar:
    timestamp = _timestamp(raw.timestamp)
    return OHLCVBar(
        provider_timestamp=timestamp,
        receipt_timestamp=received_at,
        age_seconds=max(0.0, (received_at - timestamp).total_seconds()),
        source=f"alpaca_stream:{feed}",
        quality=DataQuality.GOOD,
        symbol=_symbol(str(raw.symbol)),
        timeframe=timeframe,
        open=_positive(raw.open, "open"),
        high=_positive(raw.high, "high"),
        low=_positive(raw.low, "low"),
        close=_positive(raw.close, "close"),
        volume=int(raw.volume),
        extended_hours=not (time(9, 30) <= timestamp.astimezone(EASTERN).time() < time(16, 0)),
    )


def _aggregate_five_minute(bars: Sequence[OHLCVBar]) -> tuple[OHLCVBar, ...]:
    groups: dict[datetime, list[OHLCVBar]] = {}
    for bar in bars:
        bucket = bar.provider_timestamp.replace(
            minute=(bar.provider_timestamp.minute // 5) * 5,
            second=0,
            microsecond=0,
        )
        groups.setdefault(bucket, []).append(bar)
    output: list[OHLCVBar] = []
    for bucket, values in sorted(groups.items()):
        values.sort(key=lambda item: item.provider_timestamp)
        latest_receipt = max(item.receipt_timestamp for item in values)
        output.append(
            OHLCVBar(
                provider_timestamp=bucket,
                receipt_timestamp=latest_receipt,
                age_seconds=max(0.0, (latest_receipt - bucket).total_seconds()),
                source=values[-1].source,
                quality=max((item.quality for item in values), key=lambda value: list(DataQuality).index(value)),
                symbol=values[-1].symbol,
                timeframe="5m",
                open=values[0].open,
                high=max(item.high for item in values),
                low=min(item.low for item in values),
                close=values[-1].close,
                volume=sum(item.volume for item in values),
                extended_hours=any(item.extended_hours for item in values),
            )
        )
    return tuple(output)


def _dedupe(bars: Sequence[OHLCVBar]) -> tuple[OHLCVBar, ...]:
    indexed = {bar.provider_timestamp: bar for bar in bars}
    return tuple(indexed[key] for key in sorted(indexed))


def _positive(value: object, label: str) -> Decimal:
    result = Decimal(str(value))
    if result <= 0:
        raise ValueError(f"Alpaca returned non-positive {label}")
    return result


def _timestamp(value: object) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Alpaca timestamp must be timezone-aware")
    return value.astimezone(UTC)


def _symbol(value: str) -> str:
    normalized = value.strip().upper()
    valid_characters = all(character.isalnum() or character in ".-" for character in normalized)
    if not normalized or len(normalized) > 10 or not valid_characters:
        raise ValueError("unsupported stock symbol format")
    return normalized
