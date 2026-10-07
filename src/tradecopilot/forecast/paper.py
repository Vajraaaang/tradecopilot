"""Bounded GET-only paper account, clock and completed historical candle reads."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Literal
from zoneinfo import ZoneInfo

import httpx

from tradecopilot.auth import AlpacaKeychainStorage
from tradecopilot.forecast.bars import HistoricalBar
from tradecopilot.forecast.contracts import utc
from tradecopilot.forecast.sessions import CALENDAR_VERSION, session_bounds

PAPER_ENDPOINT = "https://paper-api.alpaca.markets"
DATA_ENDPOINT = "https://data.alpaca.markets/v2/stocks/bars"
MAX_PAGES = 50
MAX_BAR_ROWS = 30_000
MAX_RESPONSE_BYTES = 20 * 1024 * 1024
MAX_TOTAL_RAW_BYTES = 64 * 1024 * 1024
_MINUTE = timedelta(minutes=1)
_NEW_YORK = ZoneInfo("America/New_York")
_FLAGS = ("account_blocked", "trading_blocked", "trade_suspended_by_user", "transfers_blocked")


@dataclass(frozen=True)
class PaperSnapshot:
    bars: tuple[HistoricalBar, ...]
    account: dict[str, Any]
    clock: dict[str, Any]
    metadata: dict[str, Any]
    raw_pages: tuple[bytes, ...] = field(repr=False)


async def _response_bytes(
    client: httpx.AsyncClient,
    endpoint: str,
    params: dict[str, str] | None = None,
    *,
    max_bytes: int | None = None,
) -> bytes:
    limit = MAX_RESPONSE_BYTES if max_bytes is None else min(MAX_RESPONSE_BYTES, max_bytes)
    if limit <= 0:
        raise ValueError
    async with client.stream("GET", endpoint, params=params) as response:
        if response.status_code != 200:
            raise ValueError
        raw = bytearray()
        async for chunk in response.aiter_bytes():
            if len(raw) + len(chunk) > limit:
                raise ValueError
            raw.extend(chunk)
        return bytes(raw)


def _invalid_constant(value: str) -> None:
    raise ValueError


def _json_object(raw: bytes) -> dict[str, Any]:
    value = json.loads(raw, parse_float=Decimal, parse_constant=_invalid_constant)
    if not isinstance(value, dict):
        raise ValueError
    return value


def _safe_account(account: dict[str, Any]) -> dict[str, Any]:
    if account.get("status") != "ACTIVE" or account.get("currency") != "USD":
        raise ValueError
    if any(flag not in account for flag in ("account_blocked", "trading_blocked")):
        raise ValueError
    safe: dict[str, Any] = {"status": "ACTIVE", "currency": "USD"}
    for flag in _FLAGS:
        if flag in account:
            if type(account[flag]) is not bool:
                raise ValueError
            safe[flag] = account[flag]
    return safe


def _safe_clock(clock: dict[str, Any]) -> dict[str, Any]:
    if type(clock.get("is_open")) is not bool:
        raise ValueError
    safe: dict[str, Any] = {"is_open": clock["is_open"]}
    for key in ("timestamp", "next_open", "next_close"):
        if not isinstance(clock[key], str):
            raise ValueError
        safe[key] = utc(datetime.fromisoformat(clock[key])).isoformat()
    return safe


def _bar(symbol: str, row: Any, feed: Literal["iex", "sip"]) -> HistoricalBar:
    if not isinstance(row, dict) or not isinstance(row.get("t"), str):
        raise ValueError
    start = utc(datetime.fromisoformat(row["t"]))
    if start.second or start.microsecond:
        raise ValueError
    values = []
    for key in ("o", "h", "l", "c", "v"):
        value = row[key]
        if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
            raise ValueError
        values.append(Decimal(str(value)))
    return HistoricalBar(
        symbol=symbol, start_time=start, end_time=start + _MINUTE, available_at=start + _MINUTE,
        opening=values[0], high=values[1], low=values[2], close=values[3], volume=values[4],
        source="alpaca_iex_1min_bar" if feed == "iex" else "alpaca_sip_1min_bar",
    )


async def read_paper_snapshot(
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    *,
    feed: Literal["iex", "sip"] = "iex",
    credentials: tuple[str, str] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> PaperSnapshot:
    """Read fixed paper/data hosts without retries, feed fallback, redirects or order access.

    Account identifiers and balances are discarded. Raw pages contain market data
    only and must stay private. Historical availability at bar end is an assumption;
    each actual response receipt is recorded independently.
    """
    try:
        if (
            isinstance(symbols, (str, bytes)) or not 1 <= len(symbols) <= 5
            or any(not isinstance(symbol, str) or not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", symbol)
                   for symbol in symbols)
            or len(set(symbols)) != len(symbols) or feed not in ("iex", "sip")
        ):
            raise ValueError
        start, end = utc(start), utc(end)
        if not start < end or end - start > timedelta(days=31):
            raise ValueError
        selected = tuple(sorted(symbols))
        if credentials is None:
            credentials = await AlpacaKeychainStorage().get_credentials()
        if (
            not isinstance(credentials, tuple) or len(credentials) != 2
            or any(not isinstance(value, str) or not value or not value.isascii()
                   or any(ord(char) < 33 or ord(char) == 127 for char in value) for value in credentials)
        ):
            raise ValueError
        params = {
            "symbols": ",".join(selected), "timeframe": "1Min", "feed": feed, "adjustment": "raw",
            "start": start.isoformat(), "end": (end - timedelta(microseconds=1)).isoformat(),
            "limit": "1000", "sort": "asc",
        }
        bars: list[HistoricalBar] = []
        seen: set[tuple[str, datetime]] = set()
        tokens: set[str] = set()
        raw_pages: list[bytes] = []
        records: list[dict[str, Any]] = []
        exclusions: Counter[str] = Counter()
        raw_count = 0
        raw_bytes = 0
        async with httpx.AsyncClient(
            headers={"APCA-API-KEY-ID": credentials[0], "APCA-API-SECRET-KEY": credentials[1]},
            timeout=20, trust_env=False, follow_redirects=False, transport=transport,
        ) as client:
            account = _safe_account(_json_object(await _response_bytes(client, f"{PAPER_ENDPOINT}/v2/account")))
            clock = _safe_clock(_json_object(await _response_bytes(client, f"{PAPER_ENDPOINT}/v2/clock")))
            for page_number in range(1, MAX_PAGES + 1):
                raw = await _response_bytes(client, DATA_ENDPOINT, params, max_bytes=MAX_TOTAL_RAW_BYTES - raw_bytes)
                raw_bytes += len(raw)
                if raw_bytes > MAX_TOTAL_RAW_BYTES:
                    raise ValueError
                receipt = datetime.now(UTC)
                page = _json_object(raw)
                rows = page.get("bars")
                if not isinstance(rows, dict) or any(
                    symbol not in selected or not isinstance(values, list) for symbol, values in rows.items()
                ):
                    raise ValueError
                page_count = sum(len(values) for values in rows.values())
                raw_count += page_count
                if raw_count > MAX_BAR_ROWS:
                    raise ValueError
                raw_pages.append(raw)
                records.append({
                    "page": page_number, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                    "bar_count": page_count, "receipt_at": receipt.isoformat(),
                })
                for symbol, values in rows.items():
                    for row in values:
                        item = _bar(symbol, row, feed)
                        key = (item.symbol, item.start_time)
                        if key in seen:
                            raise ValueError
                        seen.add(key)
                        if item.start_time < start or item.end_time > end:
                            exclusions["outside_requested_range"] += 1
                        elif item.end_time > receipt:
                            exclusions["incomplete_at_receipt"] += 1
                        else:
                            bounds = session_bounds(item.start_time.astimezone(_NEW_YORK).date())
                            if bounds is None or item.start_time < bounds[0] or item.end_time > bounds[1]:
                                exclusions["outside_regular_session"] += 1
                            else:
                                bars.append(item)
                token = page.get("next_page_token")
                if token is None:
                    break
                if (
                    not isinstance(token, str) or not token or len(token) > 4096 or token in tokens
                    or any(ord(char) < 33 or ord(char) == 127 for char in token)
                ):
                    raise ValueError
                tokens.add(token)
                params["page_token"] = token
            else:
                raise ValueError
        bars.sort(key=lambda item: (item.symbol, item.start_time))
        return PaperSnapshot(tuple(bars), account, clock, {
            "provider": "Alpaca", "feed": feed, "adjustment_policy": "raw",
            "source": f"alpaca_{feed}_1min_bar", "source_endpoint": DATA_ENDPOINT, "paper_endpoint": PAPER_ENDPOINT,
            "requested_symbols": list(selected), "requested_start_utc": start.isoformat(),
            "requested_end_utc": end.isoformat(), "receipt_at": receipt.isoformat(), "raw_pages": records,
            "downloaded_bar_count": raw_count, "bar_count": len(bars), "exclusions": dict(sorted(exclusions.items())),
            "calendar": "XNYS", "calendar_version": CALENDAR_VERSION, "timezone": "America/New_York",
            "interval_seconds": 60, "timestamp_semantics": "Alpaca t is the minute start; bar end is start plus 60s",
            "availability_assumption": "available_at=bar end is a historical assumption; actual request receipt "
            "is recorded separately",
            "raw_data_policy": "Private local market-data evidence only",
        }, tuple(raw_pages))
    except Exception:
        # HTTP, parsing and Keychain exceptions can embed credentials or response bodies.
        raise ValueError("Alpaca paper snapshot could not be read") from None
