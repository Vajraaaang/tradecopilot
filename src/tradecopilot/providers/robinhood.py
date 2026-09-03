from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, cast
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

from tradecopilot.config import StrategyConfig
from tradecopilot.models import (
    AccountRiskSnapshot,
    DataQuality,
    Level2Level,
    Level2Snapshot,
    OHLCVBar,
    PositionSnapshot,
    Quote,
    ScreenerCandidate,
)
from tradecopilot.security import ReadOnlyToolSurface

EASTERN = ZoneInfo("America/New_York")


class CircuitOpenError(ConnectionError):
    pass


@dataclass
class RobinhoodReadAdapter:
    """Deterministic code, never an LLM, owns this read-only broker boundary."""

    surface: ReadOnlyToolSurface
    config: StrategyConfig
    _locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    _failures: int = 0
    _circuit_opened_at: float | None = None

    async def call(self, tool_name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        lock = self._locks.setdefault(tool_name, asyncio.Lock())
        async with lock:
            if self._circuit_opened_at is not None:
                elapsed = time.monotonic() - self._circuit_opened_at
                if elapsed < self.config.maximum_backoff_seconds:
                    raise CircuitOpenError("Robinhood read circuit breaker is open")
                self._circuit_opened_at = None
                self._failures = 0

            last_error: Exception | None = None
            for attempt in range(self.config.maximum_poll_retries):
                try:
                    result = await self.surface.call(tool_name, arguments)
                except Exception as exc:  # the adapter normalizes transport failures
                    last_error = exc
                    self._failures += 1
                    if self._failures >= self.config.circuit_breaker_failures:
                        self._circuit_opened_at = time.monotonic()
                        raise CircuitOpenError("Robinhood read circuit breaker opened") from exc
                    if attempt + 1 < self.config.maximum_poll_retries:
                        base = min(2**attempt, self.config.maximum_backoff_seconds)
                        await asyncio.sleep(base + random.uniform(0, min(0.25, base / 4)))
                    continue
                self._failures = 0
                return result
            raise ConnectionError("Robinhood read failed after bounded retries") from last_error

    async def connectivity_check(self) -> bool:
        await self.call("get_accounts", {})
        return True


@dataclass
class RobinhoodMarketDataProvider:
    """Normalize approved Robinhood market reads; no complete MCP client is retained."""

    adapter: RobinhoodReadAdapter
    account_number: str
    config: StrategyConfig
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    context_ttl_seconds: float = 15.0
    _context_cache: dict[str, tuple[float, int, Decimal]] = field(default_factory=dict)

    async def get_quote(self, symbol: str) -> Quote:
        normalized = symbol.strip().upper()
        quote_payload, context = await asyncio.gather(
            self.adapter.call("get_equity_quotes", {"symbols": [normalized]}),
            self._market_context(normalized),
        )
        row = _first_result(quote_payload, "results")
        raw_quote = _mapping(row.get("quote"), "quote")
        if not bool(raw_quote.get("has_traded")) or raw_quote.get("state") != "active":
            raise ValueError(f"Robinhood quote is not active for {normalized}")
        timestamp, last = _latest_trade(raw_quote)
        bid = _positive_decimal(raw_quote.get("bid_price"), "bid_price")
        ask = _positive_decimal(raw_quote.get("ask_price"), "ask_price")
        if ask < bid:
            raise ValueError("Robinhood quote has ask below bid")
        total_volume, session_origin = context
        received_at = self.clock()
        _require_not_future(timestamp, received_at)
        return Quote(
            provider_timestamp=timestamp,
            receipt_timestamp=received_at,
            age_seconds=(received_at - timestamp).total_seconds(),
            source="robinhood_mcp:get_equity_quotes",
            quality=DataQuality.GOOD,
            symbol=normalized,
            bid=bid,
            ask=ask,
            last=last,
            previous_close=_positive_decimal(raw_quote.get("adjusted_previous_close"), "adjusted_previous_close"),
            total_volume=total_volume,
            current_minute_volume=0,
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
        interval = {"1m": "minute", "5m": "5minute"}.get(timeframe)
        if interval is None:
            raise ValueError(f"unsupported Robinhood timeframe: {timeframe}")
        payload = await self.adapter.call(
            "get_equity_historicals",
            {
                "symbols": [symbol.strip().upper()],
                "start_time": start.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                "end_time": end.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                "interval": interval,
                "bounds": "extended" if self.config.vwap_session == "extended_hours" else "regular",
                "adjustment_type": "split",
            },
        )
        row = _first_result(payload, "results")
        bars = row.get("bars")
        if not isinstance(bars, list):
            return ()
        received_at = self.clock()
        output: list[OHLCVBar] = []
        for value in bars:
            if not isinstance(value, Mapping) or bool(value.get("interpolated")):
                continue
            timestamp = _timestamp(value.get("begins_at"), "begins_at")
            _require_not_future(timestamp, received_at)
            output.append(
                OHLCVBar(
                    provider_timestamp=timestamp,
                    receipt_timestamp=received_at,
                    age_seconds=(received_at - timestamp).total_seconds(),
                    source="robinhood_mcp:get_equity_historicals",
                    quality=DataQuality.GOOD,
                    symbol=str(row.get("symbol", symbol)).upper(),
                    timeframe=timeframe,
                    open=_positive_decimal(value.get("open_price"), "open_price"),
                    high=_positive_decimal(value.get("high_price"), "high_price"),
                    low=_positive_decimal(value.get("low_price"), "low_price"),
                    close=_positive_decimal(value.get("close_price"), "close_price"),
                    volume=int(value.get("volume", 0)),
                    extended_hours=value.get("session") != "reg",
                )
            )
        return tuple(sorted(output, key=lambda bar: bar.provider_timestamp))

    async def get_level2(self, symbol: str) -> Level2Snapshot:
        normalized = symbol.strip().upper()
        payload = await self.adapter.call("get_equity_price_book", {"symbols": [normalized]})
        row = _first_result(payload, "books")
        timestamp = _timestamp(row.get("updated_at"), "updated_at")
        received_at = self.clock()
        _require_not_future(timestamp, received_at)

        def levels(side: str) -> tuple[Level2Level, ...]:
            raw_levels = row.get(side)
            if not isinstance(raw_levels, list):
                return ()
            singular = "bid" if side == "bids" else "ask"
            return tuple(
                Level2Level(
                    provider_timestamp=timestamp,
                    receipt_timestamp=received_at,
                    age_seconds=(received_at - timestamp).total_seconds(),
                    source="robinhood_mcp:get_equity_price_book",
                    quality=DataQuality.GOOD,
                    side=singular,
                    price=_positive_decimal(item.get("price"), "price"),
                    size=int(item.get("quantity", 0)),
                )
                for item in raw_levels
                if isinstance(item, Mapping)
            )

        return Level2Snapshot(
            provider_timestamp=timestamp,
            receipt_timestamp=received_at,
            age_seconds=(received_at - timestamp).total_seconds(),
            source="robinhood_mcp:get_equity_price_book",
            quality=DataQuality.GOOD,
            symbol=str(row.get("symbol", normalized)).upper(),
            bids=levels("bids"),
            asks=levels("asks"),
        )

    async def is_tradable(self, symbol: str) -> bool:
        normalized = symbol.strip().upper()
        payload = await self.adapter.call(
            "get_equity_tradability",
            {"account_number": self.account_number, "symbols": [normalized]},
        )
        row = _first_result(payload, "results")
        return bool(row.get("tradeable")) and row.get("state") == "active" and not row.get("internal_halt_start_time")

    async def _market_context(self, symbol: str) -> tuple[int, Decimal]:
        now_monotonic = time.monotonic()
        cached = self._context_cache.get(symbol)
        if cached is not None and now_monotonic - cached[0] <= self.context_ttl_seconds:
            return cached[1], cached[2]
        fundamentals = await self.adapter.call("get_equity_fundamentals", {"symbols": [symbol], "bounds": "extended"})
        fundamental = _first_result(fundamentals, "results")
        total_volume = int(_decimal(fundamental.get("volume"), "volume"))
        origin = _positive_decimal(fundamental.get("open"), "open")
        self._context_cache[symbol] = (now_monotonic, total_volume, origin)
        return total_volume, origin


@dataclass
class RobinhoodBrokerReadProvider:
    """Normalize approved account reads for one explicitly supplied account number."""

    adapter: RobinhoodReadAdapter
    account_number: str
    account_alias: str
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    maximum_pages: int = 10

    async def get_positions(self) -> Sequence[PositionSnapshot]:
        received_at = self.clock()
        cursor: str | None = None
        output: list[PositionSnapshot] = []
        for _ in range(self.maximum_pages):
            arguments = {"account_number": self.account_number}
            if cursor is not None:
                arguments["cursor"] = cursor
            payload = await self.adapter.call("get_equity_positions", arguments)
            data = _data(payload)
            rows = data.get("positions")
            if isinstance(rows, list):
                for value in rows:
                    if not isinstance(value, Mapping):
                        continue
                    normalized = str(value.get("symbol", "")).strip().upper()
                    if not normalized:
                        continue
                    quantity = _decimal(value.get("quantity"), "quantity")
                    if quantity < 0:
                        raise ValueError("short equity positions are unsupported by the long-only strategy")
                    average = value.get("average_buy_price")
                    output.append(
                        PositionSnapshot(
                            provider_timestamp=received_at,
                            receipt_timestamp=received_at,
                            age_seconds=0,
                            source="robinhood_mcp:get_equity_positions",
                            quality=DataQuality.LIMITED,
                            symbol=normalized,
                            account_alias=self.account_alias,
                            quantity=quantity,
                            average_entry=(
                                _positive_decimal(average, "average_buy_price") if average is not None else None
                            ),
                            unrealized_pnl=None,
                            latest_higher_low=None,
                            mfe=None,
                            mae=None,
                        )
                    )
            cursor = _next_cursor(data.get("next"))
            if cursor is None:
                return tuple(output)
        raise ValueError("Robinhood position pagination exceeded the safety limit")

    async def get_position(self, symbol: str) -> PositionSnapshot | None:
        normalized = symbol.strip().upper()
        return next((position for position in await self.get_positions() if position.symbol == normalized), None)

    async def get_account_risk(self) -> AccountRiskSnapshot:
        now = self.clock()
        portfolio, realized, trades = await asyncio.gather(
            self.adapter.call("get_portfolio", {"account_number": self.account_number}),
            self.adapter.call(
                "get_realized_pnl",
                {
                    "account_number": self.account_number,
                    "span": "day",
                    "asset_classes": ["equity"],
                    "timezone": "America/New_York",
                },
            ),
            self._trade_rows(),
        )
        portfolio_data = _data(portfolio)
        buying_power = _mapping(portfolio_data.get("buying_power"), "buying_power")
        realized_data = _data(realized)
        today = now.astimezone(EASTERN).date()
        todays_trades = [
            row for row in trades if _timestamp(row.get("timestamp"), "timestamp").astimezone(EASTERN).date() == today
        ]
        chronological = sorted(todays_trades, key=lambda row: _timestamp(row.get("timestamp"), "timestamp"))
        cumulative = Decimal(0)
        peak = Decimal(0)
        for row in chronological:
            cumulative += _decimal(row.get("realized_gain"), "realized_gain")
            peak = max(peak, cumulative)
        consecutive_losses = 0
        for row in reversed(chronological):
            if _decimal(row.get("realized_gain"), "realized_gain") >= 0:
                break
            consecutive_losses += 1
        current_realized = _decimal(realized_data.get("total_returns"), "total_returns")
        peak = max(peak, current_realized, Decimal(0))
        return AccountRiskSnapshot(
            provider_timestamp=now,
            receipt_timestamp=now,
            age_seconds=0,
            source="robinhood_mcp:account_risk",
            quality=DataQuality.LIMITED,
            trading_date=today,
            buying_power=_decimal(buying_power.get("buying_power"), "buying_power"),
            realized_session_pnl=current_realized,
            peak_realized_session_pnl=peak,
            unrealized_pnl=Decimal(0),
            consecutive_losses=consecutive_losses,
            session_locked=False,
        )

    async def get_trade_history(self) -> Sequence[dict[str, object]]:
        return tuple(dict(row) for row in await self._trade_rows())

    async def get_open_orders(self, symbol: str) -> Sequence[dict[str, object]]:
        normalized = symbol.strip().upper()
        cursor: str | None = None
        output: list[dict[str, object]] = []
        open_states = {
            "new",
            "queued",
            "confirmed",
            "unconfirmed",
            "partially_filled",
            "pending_cancelled",
            "locating",
        }
        for _ in range(self.maximum_pages):
            arguments = {"account_number": self.account_number, "symbol": normalized}
            if cursor is not None:
                arguments["cursor"] = cursor
            payload = await self.adapter.call("get_equity_orders", arguments)
            data = _data(payload)
            rows = data.get("orders")
            if isinstance(rows, list):
                output.extend(dict(row) for row in rows if isinstance(row, Mapping) and row.get("state") in open_states)
            cursor = _next_cursor(data.get("next"))
            if cursor is None:
                return tuple(output)
        raise ValueError("Robinhood order pagination exceeded the safety limit")

    async def _trade_rows(self) -> list[Mapping[str, Any]]:
        cursor: str | None = None
        output: list[Mapping[str, Any]] = []
        for _ in range(self.maximum_pages):
            arguments = {"account_number": self.account_number, "span": "week"}
            if cursor is not None:
                arguments["cursor"] = cursor
            payload = await self.adapter.call("get_pnl_trade_history", arguments)
            data = _data(payload)
            output.extend(_trade_rows(payload))
            next_cursor = data.get("next_cursor")
            if not isinstance(next_cursor, str) or not next_cursor:
                return output
            cursor = next_cursor
        raise ValueError("Robinhood trade-history pagination exceeded the safety limit")


@dataclass
class RobinhoodScannerProvider:
    adapter: RobinhoodReadAdapter
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)

    async def candidates(self, scan_title: str | None = None) -> tuple[ScreenerCandidate, ...]:
        scans_payload = await self.adapter.call("get_scans", {})
        scans = _data(scans_payload).get("scans")
        if not isinstance(scans, list) or not scans:
            return ()
        usable = [item for item in scans if isinstance(item, Mapping)]
        if scan_title:
            matching = [item for item in usable if str(item.get("title", "")).casefold() == scan_title.casefold()]
            if not matching:
                return ()
            selected = matching[0]
        else:
            selected = usable[0]
        scan_id = str(selected.get("scan_id", ""))
        if not scan_id:
            return ()
        payload = await self.adapter.call("run_scan", {"scan_id": scan_id})
        result = _mapping(_data(payload).get("result"), "result")
        rows = result.get("results")
        if not isinstance(rows, list):
            return ()
        now = self.clock()
        title = str(result.get("scan_title") or selected.get("title") or "Robinhood scan")
        output: list[ScreenerCandidate] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            ticker = str(row.get("ticker", "")).strip().upper()
            columns = row.get("columns")
            if not ticker or not isinstance(columns, Mapping):
                continue
            output.append(
                ScreenerCandidate(
                    provider_timestamp=now,
                    receipt_timestamp=now,
                    age_seconds=0,
                    source="robinhood_mcp:run_scan",
                    quality=DataQuality.LIMITED,
                    symbol=ticker,
                    scan_id=scan_id,
                    scan_title=title,
                    columns={str(key): str(value) for key, value in columns.items()},
                )
            )
        return tuple(output)


async def resolve_robinhood_account(
    adapter: RobinhoodReadAdapter,
    requested_last4: str | None,
) -> tuple[str, str]:
    """Resolve an explicit account choice while returning only a masked display alias."""

    payload = await adapter.call("get_accounts", {})
    accounts = _data(payload).get("accounts")
    if not isinstance(accounts, list):
        raise ValueError("Robinhood returned no account list")
    rows = [item for item in accounts if isinstance(item, Mapping) and not bool(item.get("deactivated"))]
    if requested_last4 is not None:
        suffix = requested_last4.strip()
        if len(suffix) != 4 or not suffix.isdigit():
            raise ValueError("account selector must be exactly the last four digits")
        rows = [item for item in rows if str(item.get("account_number", "")).endswith(suffix)]
    if len(rows) != 1:
        raise ValueError("select exactly one Robinhood account with --account-last4")
    account_number = str(rows[0].get("account_number", ""))
    if len(account_number) < 4:
        raise ValueError("Robinhood account identifier is invalid")
    nickname = rows[0].get("nickname")
    alias = str(nickname).strip() if isinstance(nickname, str) and nickname.strip() else f"••••{account_number[-4:]}"
    return account_number, alias


def _data(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(payload.get("data"), "data")


def _first_result(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    values = _data(payload).get(key)
    if not isinstance(values, list):
        raise ValueError(f"Robinhood response has no {key} array")
    for value in values:
        if isinstance(value, Mapping):
            return cast(Mapping[str, Any], value)
    raise ValueError(f"Robinhood response has no usable {key} row")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Robinhood response is missing {label}")
    return cast(Mapping[str, Any], value)


def _decimal(value: object, label: str) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception as exc:
        raise ValueError(f"Robinhood response has invalid {label}") from exc


def _positive_decimal(value: object, label: str) -> Decimal:
    result = _decimal(value, label)
    if result <= 0:
        raise ValueError(f"Robinhood response has non-positive {label}")
    return result


def _timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"Robinhood response is missing {label}")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"Robinhood response has naive {label}")
    return result


def _latest_trade(raw_quote: Mapping[str, Any]) -> tuple[datetime, Decimal]:
    regular = (
        _timestamp(raw_quote.get("venue_last_trade_time"), "venue_last_trade_time"),
        _positive_decimal(raw_quote.get("last_trade_price"), "last_trade_price"),
    )
    non_regular_time = raw_quote.get("venue_last_non_reg_trade_time")
    non_regular_price = raw_quote.get("last_non_reg_trade_price")
    if non_regular_time is None or non_regular_price is None:
        return regular
    alternate = (
        _timestamp(non_regular_time, "venue_last_non_reg_trade_time"),
        _positive_decimal(non_regular_price, "last_non_reg_trade_price"),
    )
    return max(regular, alternate, key=lambda item: item[0])


def _trade_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = _data(payload).get("trades")
    if not isinstance(rows, list):
        return []
    return [cast(Mapping[str, Any], row) for row in rows if isinstance(row, Mapping)]


def _next_cursor(value: object) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError("Robinhood pagination cursor is invalid")
    cursor = parse_qs(urlparse(value).query).get("cursor", [None])[0]
    if not isinstance(cursor, str) or not cursor:
        raise ValueError("Robinhood pagination URL has no cursor")
    return cursor


def _require_not_future(provider_timestamp: datetime, receipt_timestamp: datetime) -> None:
    if provider_timestamp > receipt_timestamp:
        raise ValueError("Robinhood provider timestamp is in the future")
