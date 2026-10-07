"""Bounded, read-only Alpaca SIP minute history; retrieval is separate from replay availability."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from tradecopilot.forecast.bars import BAR_DATA_VERSION, HistoricalBar, write_bar_dataset
from tradecopilot.forecast.contracts import ForecastConfig, utc
from tradecopilot.forecast.sessions import CALENDAR_VERSION, session_bounds

ENDPOINT = "https://data.alpaca.markets/v2/stocks/bars"
MAX_PAGES = 100
MAX_RESPONSE_BYTES = 20 * 1024 * 1024
MAX_BAR_ROWS = 500_000
_MAX_TOKEN_LENGTH = 4096
_NEW_YORK = ZoneInfo("America/New_York")
_MINUTE = timedelta(minutes=1)


class _ImportError(ValueError):
    """Only deliberately safe error messages cross the importer boundary."""


def _invalid_constant(value: str) -> None:
    raise _ImportError("Alpaca response contains nonfinite values")


def _parse_bar(symbol: str, row: Any, start: datetime, end: datetime) -> HistoricalBar:
    try:
        if not isinstance(row, dict) or not isinstance(row.get("t"), str):
            raise ValueError
        timestamp = utc(datetime.fromisoformat(row["t"]))
        if timestamp.second or timestamp.microsecond or not start <= timestamp < end:
            raise ValueError
        values = []
        for field in ("o", "h", "l", "c", "v"):
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
                raise ValueError
            values.append(Decimal(str(value)))
        return HistoricalBar(
            symbol=symbol, start_time=timestamp, end_time=timestamp + _MINUTE,
            available_at=timestamp + _MINUTE, opening=values[0], high=values[1],
            low=values[2], close=values[3], volume=values[4], source="alpaca_sip_1min_bar",
        )
    except (ValueError, TypeError, KeyError, ArithmeticError):
        raise _ImportError("invalid Alpaca bar values, timestamp, or requested range") from None


async def _response_bytes(client: httpx.AsyncClient, params: dict[str, str]) -> bytes:
    try:
        async with client.stream("GET", ENDPOINT, params=params) as response:
            if response.status_code != 200:
                raise _ImportError(f"Alpaca historical data HTTP status {response.status_code}")
            chunks = bytearray()
            async for chunk in response.aiter_bytes():
                if len(chunks) + len(chunk) > MAX_RESPONSE_BYTES:
                    raise _ImportError("Alpaca response exceeds byte size limit")
                chunks.extend(chunk)
            return bytes(chunks)
    except _ImportError:
        raise
    except Exception:
        # Transport exceptions can contain URLs, headers, or echoed credentials.
        raise _ImportError("Alpaca historical data request failed") from None


def _parse_page(raw: bytes, symbols: tuple[str, ...]) -> tuple[dict[str, Any], str | None]:
    try:
        page = json.loads(raw, parse_float=Decimal, parse_constant=_invalid_constant)
    except (ValueError, UnicodeError, RecursionError):
        raise _ImportError("invalid Alpaca historical response JSON") from None
    if not isinstance(page, dict) or not isinstance(page.get("bars"), dict):
        raise _ImportError("invalid Alpaca historical response bars")
    rows = page["bars"]
    if any(symbol not in symbols or not isinstance(values, list) for symbol, values in rows.items()):
        raise _ImportError("invalid Alpaca historical response symbols or bar rows")
    token = page.get("next_page_token")
    if token is not None and (
        not isinstance(token, str) or not token or len(token) > _MAX_TOKEN_LENGTH
        or any(ord(character) < 32 or ord(character) == 127 for character in token)
    ):
        raise _ImportError("invalid Alpaca pagination token")
    return rows, token


def _validated_range(config: ForecastConfig, start: date, end: date) -> tuple[datetime, datetime]:
    try:
        validated = ForecastConfig.model_validate(config.model_dump(mode="json"))
    except (ValueError, TypeError, AttributeError):
        raise _ImportError("invalid forecast configuration") from None
    if len(validated.symbols) > 5:
        raise _ImportError("Alpaca historical import is limited to five symbols")
    if (
        type(start) is not date or type(end) is not date or not start < end
        or (end - start).days > 366 or end >= datetime.now(UTC).date()
    ):
        raise _ImportError("historical date range must span 1-366 days and end before the current UTC day")
    return (
        datetime.combine(start, time.min, tzinfo=_NEW_YORK).astimezone(UTC),
        datetime.combine(end, time.min, tzinfo=_NEW_YORK).astimezone(UTC),
    )


async def download_history(
    config: ForecastConfig,
    start: date,
    end: date,
    output_dir: Path,
    api_key: str,
    secret_key: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Path:
    """Download a historical [start, end) NY-date range into a new private local directory.

    SIP entitlement errors stop the import. There are no retries, alternate feeds,
    current-day requests, trading endpoints, or inference calls. Raw response pages
    are private research evidence; available_at=bar end is a replay assumption.
    """
    start_utc, end_utc = _validated_range(config, start, end)
    if any(
        not isinstance(value, str) or not value or not value.isascii()
        or any(ord(character) < 33 or ord(character) == 127 for character in value)
        for value in (api_key, secret_key)
    ):
        raise _ImportError("Alpaca credentials are required; run tradecopilot auth alpaca")
    if output_dir.exists() or output_dir.is_symlink():
        raise _ImportError("Alpaca history output is immutable; choose a directory that does not exist")
    symbols = tuple(sorted(config.symbols))
    base_params = {
        "symbols": ",".join(symbols), "timeframe": "1Min", "feed": "sip", "adjustment": "raw",
        "start": start_utc.astimezone(_NEW_YORK).isoformat(),
        # Alpaca includes its end timestamp; our requested date boundary is exclusive.
        "end": (end_utc - timedelta(microseconds=1)).astimezone(_NEW_YORK).isoformat(), "limit": "10000",
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    exclusions: Counter[str] = Counter()
    bars: dict[tuple[str, datetime], HistoricalBar] = {}
    downloads: list[dict[str, Any]] = []
    downloaded_count = 0
    tokens: set[str] = set()
    token = None
    # Stage private evidence on disk, avoiding accumulation of up to 100 response pages in RAM.
    with TemporaryDirectory(prefix=".alpaca-history-", dir=output_dir.parent) as temporary:
        stage = Path(temporary)
        (stage / "raw").mkdir(mode=0o700)
        async with httpx.AsyncClient(
            headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key},
            timeout=30, trust_env=False, follow_redirects=False, transport=transport,
        ) as client:
            for page_number in range(1, MAX_PAGES + 1):
                params = dict(base_params)
                if token is not None:
                    params["page_token"] = token
                raw = await _response_bytes(client, params)
                retrieved_at = datetime.now(UTC).isoformat()
                rows, next_token = _parse_page(raw, symbols)
                downloaded_count += sum(len(values) for values in rows.values())
                if downloaded_count > MAX_BAR_ROWS:
                    raise _ImportError("Alpaca historical bar row count exceeds limit")
                for symbol, values in rows.items():
                    for row in values:
                        bar = _parse_bar(symbol, row, start_utc, end_utc)
                        key = (symbol, bar.start_time)
                        existing = bars.get(key)
                        if existing is not None:
                            if existing != bar:
                                raise _ImportError("conflicting duplicate Alpaca historical bar minute")
                            exclusions["duplicate_identical"] += 1
                            continue
                        bars[key] = bar
                raw_path = Path("raw") / f"page-{page_number:04d}.json"
                with (stage / raw_path).open("xb") as file:
                    file.write(raw)
                (stage / raw_path).chmod(0o600)
                downloads.append({
                    "path": raw_path.as_posix(), "url": ENDPOINT, "params": params,
                    "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "retrieved_at": retrieved_at,
                })
                if next_token is None:
                    break
                if next_token in tokens:
                    raise _ImportError("Alpaca pagination token loop")
                tokens.add(next_token)
                token = next_token
            else:
                raise _ImportError("Alpaca historical page count exceeds limit")
        regular_bars = []
        for bar in bars.values():
            day = bar.start_time.astimezone(_NEW_YORK).date()
            bounds = session_bounds(day)
            if bounds is None or bar.start_time < bounds[0] or bar.end_time > bounds[1]:
                exclusions["outside_regular_session"] += 1
                continue
            regular_bars.append(bar)
        metadata = {
            "schema_version": BAR_DATA_VERSION, "provenance": "historical", "provider": "Alpaca",
            "provider_url": "https://alpaca.markets", "endpoint": ENDPOINT,
            "documentation_url": "https://docs.alpaca.markets/us/reference/stockbars",
            "feed": "sip", "adjustment_policy": "raw", "source_params": base_params,
            "selected_symbols": list(symbols), "start_date": start.isoformat(), "end_date_exclusive": end.isoformat(),
            "downloads": downloads, "downloaded_bar_count": downloaded_count,
            "bar_count": len(regular_bars), "exclusions": dict(sorted(exclusions.items())),
            "calendar": "XNYS", "calendar_version": CALENDAR_VERSION, "timezone": "America/New_York",
            "interval_seconds": 60, "observed_sessions": sorted({
                bar.start_time.astimezone(_NEW_YORK).date().isoformat() for bar in regular_bars
            }),
            "first_bar_start_utc": min(bar.start_time for bar in regular_bars).isoformat() if regular_bars else None,
            "last_bar_end_utc": max(bar.end_time for bar in regular_bars).isoformat() if regular_bars else None,
            "timestamp_semantics": "Alpaca timestamp marks bar start; OHLCV is assumed available at bar end",
            "availability_assumption": "Replay only: available_at equals bar end; actual delivery times are "
            "unavailable; retrieved_at records historical retrieval, not minute availability",
            "retrieved_at": downloads[-1]["retrieved_at"],
            "raw_data_policy": "Private local research evidence; subject to Alpaca market-data entitlement and terms",
        }
        write_bar_dataset(stage / "bars", regular_bars, metadata)
        try:
            output_dir.mkdir(mode=0o700)
        except FileExistsError:
            raise _ImportError("Alpaca history output is immutable; directory already exists") from None
        (stage / "raw").rename(output_dir / "raw")
        (stage / "bars").rename(output_dir / "bars")
    return output_dir / "bars" / "manifest.json"
