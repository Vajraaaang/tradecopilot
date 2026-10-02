"""Import local FirstRate Data samples without claiming live quote availability.

Raw archives stay local. Provider and receipt timestamps in returned observations
are replay assumptions at bar end, not actual trade or data-delivery timestamps.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile
from zoneinfo import ZoneInfo

from tradecopilot.forecast.contracts import Observation, utc
from tradecopilot.forecast.sessions import CALENDAR_VERSION, session_bounds
from tradecopilot.models import DataQuality

MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_MEMBER_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_ROWS = 250_000
MAX_LINE_BYTES = 1024
_NEW_YORK = ZoneInfo("America/New_York")
_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
_MINUTE = timedelta(minutes=1)


def _bounded_read(path: Path, limit: int) -> bytes:
    if path.stat().st_size > limit:
        raise ValueError("historical input exceeds size limit")
    with path.open("rb") as file:
        raw = file.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("historical input exceeds size limit")
    return raw


def _start_timestamp(value: str) -> datetime:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:00", value):
        raise ValueError("bar timestamp must mark a local minute start")
    local = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    first = local.replace(tzinfo=_NEW_YORK, fold=0)
    second = local.replace(tzinfo=_NEW_YORK, fold=1)
    if (
        first.utcoffset() != second.utcoffset()
        or first.astimezone(UTC).astimezone(_NEW_YORK).replace(tzinfo=None) != local
    ):
        raise ValueError("ambiguous or nonexistent local bar timestamp")
    return first.astimezone(UTC)


def _read_bars(
    raw: bytes, symbol: str, exclusions: Counter[str],
) -> tuple[dict[datetime, tuple[Decimal, ...]], dict[str, Any]]:
    try:
        with ZipFile(io.BytesIO(raw)) as archive:
            members = archive.infolist()
            if (
                len(members) > 20 or sum(member.file_size for member in members) > MAX_TOTAL_BYTES
                or any(member.file_size > MAX_MEMBER_BYTES for member in members)
            ):
                raise ValueError("historical ZIP member size limit exceeded")
            filename = f"{symbol}_1min_sample.csv"
            matches = [member for member in members if member.filename == filename]
            if len(matches) != 1:
                raise ValueError("exact unique minute CSV member filename is required")
            with archive.open(matches[0]) as file:
                data = file.read(MAX_MEMBER_BYTES + 1)
            if len(data) > MAX_MEMBER_BYTES:
                raise ValueError("historical CSV size limit exceeded")
    except (BadZipFile, RuntimeError, OSError) as error:
        raise ValueError("invalid historical ZIP archive") from error
    try:
        text = data.decode("utf-8")
        if any(len(line.encode("utf-8")) > MAX_LINE_BYTES for line in text.splitlines()):
            raise ValueError("historical CSV line size limit exceeded")
        reader = csv.reader(io.StringIO(text), strict=True)
        if next(reader, None) != _COLUMNS:
            raise ValueError("historical CSV columns must be timestamp,open,high,low,close,volume")
        bars: dict[datetime, tuple[Decimal, ...]] = {}
        rows = 0
        for row in reader:
            rows += 1
            if rows > MAX_ROWS:
                raise ValueError("historical CSV row limit exceeded")
            if len(row) != len(_COLUMNS):
                raise ValueError("malformed historical bar row")
            start = _start_timestamp(row[0])
            prices = tuple(Decimal(value) for value in row[1:5])
            if any(not price.is_finite() or price <= 0 for price in prices):
                raise ValueError("OHLC prices must be finite and positive")
            opening, high, low, close = prices
            if high < max(opening, close) or low > min(opening, close) or high < low:
                raise ValueError("OHLC prices are inconsistent")
            if not re.fullmatch(r"\d+", row[5]):
                raise ValueError("bar volume must be a nonnegative integer")
            values = (*prices, Decimal(row[5]))
            if start in bars:
                if bars[start] != values:
                    raise ValueError("conflicting duplicate historical bar")
                exclusions["duplicate_identical"] += 1
            else:
                bars[start] = values
    except (UnicodeError, csv.Error, InvalidOperation) as error:
        raise ValueError("malformed historical bar data") from error
    bounds = sorted(bars)
    return bars, {
        "symbol": symbol, "filename": filename, "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data), "rows": rows, "unique_rows": len(bars),
        "first_bar_start_utc": bounds[0].isoformat() if bounds else None,
        "last_bar_start_utc": bounds[-1].isoformat() if bounds else None,
        "first_local_date": bounds[0].astimezone(_NEW_YORK).date().isoformat() if bounds else None,
        "last_local_date": bounds[-1].astimezone(_NEW_YORK).date().isoformat() if bounds else None,
    }


def _download_records(directory: Path, symbols: list[str]) -> list[dict[str, Any]]:
    try:
        manifest = json.loads(_bounded_read(directory / "downloads.json", 1024 * 1024))
    except (ValueError, OSError) as error:
        raise ValueError("invalid historical download manifest") from error
    if not isinstance(manifest, list) or not all(isinstance(record, dict) for record in manifest):
        raise ValueError("historical download manifest must contain records")
    records = []
    for symbol in symbols:
        matches = [record for record in manifest if record.get("symbol") == symbol]
        if len(matches) != 1:
            raise ValueError("download manifest requires one record for each selected symbol")
        record = matches[0]
        if (
            not isinstance(record.get("bytes"), int) or isinstance(record["bytes"], bool)
            or not 0 < record["bytes"] <= MAX_ARCHIVE_BYTES
            or not isinstance(record.get("sha256"), str)
            or not re.fullmatch(r"[a-f0-9]{64}", record["sha256"])
            or record.get("url") != f"https://frd001.s3.us-east-2.amazonaws.com/frd_sample_stock_{symbol}.zip"
        ):
            raise ValueError("invalid historical download manifest record")
        try:
            utc(datetime.fromisoformat(record["retrieved_at"]))
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError("manifest retrieval time must be timezone-aware") from error
        records.append(dict(record))
    return records


def import_frd_samples(directory: Path, symbols: Sequence[str]) -> tuple[list[Observation], dict[str, Any]]:
    """Validate local downloads and replay regular-session minute closes.

    Previous close is the last *observed* regular minute close of the immediately
    preceding XNYS session. Missing minutes remain missing, including at session
    end; this proxy is not an official daily close or closing-auction price.
    """
    if (
        not symbols or len(symbols) > 20 or len(set(symbols)) != len(symbols)
        or any(not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", symbol) for symbol in symbols)
    ):
        raise ValueError("provide 1-20 unique normalized stock symbols")
    selected = sorted(symbols)
    downloads = _download_records(directory, selected)
    observations: list[Observation] = []
    csv_files = []
    exclusions: Counter[str] = Counter()
    for record in downloads:
        symbol = record["symbol"]
        raw = _bounded_read(directory / f"{symbol}_sample.zip", MAX_ARCHIVE_BYTES)
        if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("historical archive does not match download manifest")
        bars, csv_metadata = _read_bars(raw, symbol, exclusions)
        sessions: dict[date, list[tuple[datetime, Decimal]]] = defaultdict(list)
        for start, prices in sorted(bars.items()):
            day = start.astimezone(_NEW_YORK).date()
            bounds = session_bounds(day)
            if bounds is None or start < bounds[0] or start + _MINUTE > bounds[1]:
                exclusions["outside_regular_session"] += 1
                continue
            sessions[day].append((start + _MINUTE, prices[3]))
        before = len(observations)
        for day, rows in sorted(sessions.items()):
            prior = day - timedelta(days=1)
            while session_bounds(prior) is None:
                prior -= timedelta(days=1)
            previous = sessions.get(prior)
            if not previous:
                exclusions["previous_session_unavailable"] += len(rows)
                continue
            previous_close = previous[-1][1]
            observations.extend(Observation(
                symbol=symbol, last=close, previous_close=previous_close,
                provider_timestamp=end, receipt_timestamp=end, age_seconds=0,
                source="firstratedata_1min_close", quality=DataQuality.LIMITED, provenance="historical",
            ) for end, close in rows)
        csv_metadata.update({
            "regular_session_rows": sum(len(rows) for rows in sessions.values()),
            "observation_count": len(observations) - before,
            "observed_sessions": [day.isoformat() for day in sorted(sessions)],
        })
        csv_files.append(csv_metadata)
    observations.sort(key=lambda row: (row.receipt_timestamp, row.provider_timestamp, row.symbol))
    return observations, {
        "schema_version": "historical-frd-sample-v1", "provenance": "historical",
        "provider": "FirstRate Data", "provider_url": "https://firstratedata.com",
        "page_url": "https://firstratedata.com/free-tick-data",
        "product_page_urls": {symbol: f"https://firstratedata.com/i/stock/{symbol}" for symbol in selected},
        "license_url": "https://firstratedata.com/about/license",
        "documentation_url": "https://firstratedata.com/about/FAQ",
        "downloads": downloads, "selected_symbols": selected, "csv_files": csv_files,
        "exclusions": dict(sorted(exclusions.items())), "observation_count": len(observations),
        "calendar": "XNYS", "calendar_version": CALENDAR_VERSION,
        "timezone": "America/New_York", "interval_seconds": 60,
        "timestamp_semantics": "CSV timestamp marks bar start; close is available at start plus one minute",
        "adjustment_policy": "unspecified_in_free_sample",
        "previous_close_policy": "Last observed regular-session minute close of immediately preceding XNYS session; "
        "proxy only, not official daily or auction close; unavailable previous sessions skipped",
        "availability_assumption": "Replay only: provider_timestamp and receipt_timestamp both equal bar end; "
        "age_seconds=0 is assumed, not measured latency; actual trade and receipt times are unavailable",
        "retrieved_at": max(downloads, key=lambda record: utc(datetime.fromisoformat(record["retrieved_at"])))[
            "retrieved_at"
        ],
        "raw_data_policy": "Local only; publish derived research with FirstRate Data attribution",
    }
