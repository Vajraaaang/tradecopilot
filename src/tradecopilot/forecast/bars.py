"""Preserve validated local historical OHLCV; replay availability is assumed at bar end."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, Self
from zoneinfo import ZoneInfo

from pydantic import Field, field_validator, model_validator

from tradecopilot.forecast.contracts import content_hash, utc
from tradecopilot.forecast.historical import MAX_ARCHIVE_BYTES, _bounded_read, _download_records, _read_bars
from tradecopilot.forecast.sessions import CALENDAR_VERSION, session_bounds
from tradecopilot.models import FrozenModel

BAR_DATA_VERSION = "historical-ohlcv-bars-v2"
_MINUTE = timedelta(minutes=1)
_NEW_YORK = ZoneInfo("America/New_York")
_MAX_DATASET_BYTES = 256 * 1024 * 1024
_MAX_MANIFEST_BYTES = 1024 * 1024


class HistoricalBar(FrozenModel):
    symbol: str = Field(pattern=r"^[A-Z][A-Z0-9.-]{0,9}$")
    start_time: datetime
    end_time: datetime
    available_at: datetime
    opening: Decimal = Field(gt=0, allow_inf_nan=False)
    high: Decimal = Field(gt=0, allow_inf_nan=False)
    low: Decimal = Field(gt=0, allow_inf_nan=False)
    close: Decimal = Field(gt=0, allow_inf_nan=False)
    volume: Decimal = Field(ge=0, allow_inf_nan=False)
    provenance: Literal["historical"] = "historical"
    source: Literal["firstratedata_1min_bar", "alpaca_sip_1min_bar"] = "firstratedata_1min_bar"

    @field_validator("start_time", "end_time", "available_at")
    @classmethod
    def normalize_utc(cls, value: datetime) -> datetime:
        return utc(value)

    @model_validator(mode="after")
    def validate_bar(self) -> Self:
        if self.end_time - self.start_time != _MINUTE:
            raise ValueError("historical bar interval must be one minute")
        if self.available_at < self.end_time:
            raise ValueError("bar cannot be available before its end")
        if (
            self.high < max(self.opening, self.close)
            or self.low > min(self.opening, self.close) or self.high < self.low
        ):
            raise ValueError("OHLC prices are inconsistent")
        return self

    @property
    def bar_id(self) -> str:
        return content_hash(self.model_dump(mode="json"))


def import_frd_bars(directory: Path, symbols: Sequence[str]) -> tuple[list[HistoricalBar], dict[str, Any]]:
    """Reuse bounded archive checks and retain every regular XNYS minute, including the primer session."""
    if (
        not symbols or len(symbols) > 20 or len(set(symbols)) != len(symbols)
        or any(not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", symbol) for symbol in symbols)
    ):
        raise ValueError("provide 1-20 unique normalized stock symbols")
    selected = sorted(symbols)
    downloads = _download_records(directory, selected)
    result: list[HistoricalBar] = []
    csv_files = []
    exclusions: Counter[str] = Counter()
    for record in downloads:
        symbol = record["symbol"]
        raw = _bounded_read(directory / f"{symbol}_sample.zip", MAX_ARCHIVE_BYTES)
        if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("historical archive does not match download manifest")
        rows, csv_metadata = _read_bars(raw, symbol, exclusions)
        observed_sessions = set()
        before = len(result)
        for start, prices in sorted(rows.items()):
            day = start.astimezone(_NEW_YORK).date()
            bounds = session_bounds(day)
            end = start + _MINUTE
            if bounds is None or start < bounds[0] or end > bounds[1]:
                exclusions["outside_regular_session"] += 1
                continue
            observed_sessions.add(day.isoformat())
            result.append(HistoricalBar(
                symbol=symbol, start_time=start, end_time=end, available_at=end,
                opening=prices[0], high=prices[1], low=prices[2], close=prices[3], volume=prices[4],
            ))
        csv_metadata.update({
            "regular_session_rows": len(result) - before,
            "bar_count": len(result) - before, "observed_sessions": sorted(observed_sessions),
        })
        csv_files.append(csv_metadata)
    result.sort(key=lambda bar: (bar.available_at, bar.symbol, bar.start_time))
    return result, {
        "schema_version": BAR_DATA_VERSION, "provenance": "historical",
        "provider": "FirstRate Data", "provider_url": "https://firstratedata.com",
        "page_url": "https://firstratedata.com/free-tick-data",
        "product_page_urls": {symbol: f"https://firstratedata.com/i/stock/{symbol}" for symbol in selected},
        "license_url": "https://firstratedata.com/about/license",
        "documentation_url": "https://firstratedata.com/about/FAQ",
        "downloads": downloads, "selected_symbols": selected, "csv_files": csv_files,
        "exclusions": dict(sorted(exclusions.items())), "bar_count": len(result),
        "calendar": "XNYS", "calendar_version": CALENDAR_VERSION,
        "timezone": "America/New_York", "interval_seconds": 60,
        "first_bar_start_utc": min(bar.start_time for bar in result).isoformat() if result else None,
        "last_bar_end_utc": max(bar.end_time for bar in result).isoformat() if result else None,
        "timestamp_semantics": "CSV timestamp marks bar start; OHLCV is available at start plus one minute",
        "adjustment_policy": "unspecified_in_free_sample",
        "availability_assumption": "Replay only: available_at equals bar end; actual trade and delivery times "
        "are unavailable; retrieved_at records the actual archive retrieval, not minute availability",
        "retrieved_at": max(downloads, key=lambda record: utc(datetime.fromisoformat(record["retrieved_at"])))[
            "retrieved_at"
        ],
        "raw_data_policy": "Local only; publish derived research with FirstRate Data attribution",
    }


class _BarManifest(FrozenModel):
    schema_version: Literal["historical-ohlcv-bars-v2"] = "historical-ohlcv-bars-v2"
    data_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    bytes: int = Field(ge=0, le=_MAX_DATASET_BYTES, strict=True)
    bar_count: int = Field(ge=0, strict=True)
    source_metadata: dict[str, Any]


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _ordered_bars(bars: Sequence[HistoricalBar]) -> list[HistoricalBar]:
    validated = [HistoricalBar.model_validate(bar.model_dump(mode="json")) for bar in bars]
    validated.sort(key=lambda bar: (bar.available_at, bar.symbol, bar.start_time))
    keys = {(bar.symbol, bar.start_time) for bar in validated}
    if len(keys) != len(validated):
        raise ValueError("duplicate historical bar minute")
    return validated


def write_bar_dataset(directory: Path, bars: Sequence[HistoricalBar], metadata: dict[str, Any]) -> Path:
    """Write a local immutable dataset, or verify an identical existing one without rewriting it."""
    ordered = _ordered_bars(bars)
    payload = b"".join(_json_bytes(bar.model_dump(mode="json")) + b"\n" for bar in ordered)
    if len(payload) > _MAX_DATASET_BYTES:
        raise ValueError("bar dataset exceeds size limit")
    identity = {
        "schema_version": BAR_DATA_VERSION, "content_hash": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload), "bar_count": len(ordered), "source_metadata": metadata,
    }
    manifest = _BarManifest.model_validate({**identity, "data_id": content_hash(identity)}).model_dump(mode="json")
    encoded_manifest = _json_bytes(manifest) + b"\n"
    if len(encoded_manifest) > _MAX_MANIFEST_BYTES:
        raise ValueError("bar manifest exceeds size limit")
    manifest_path = directory / "manifest.json"
    bars_path = directory / "bars.jsonl"
    if manifest_path.exists() or bars_path.exists():
        _, existing = load_bar_dataset(directory)
        if existing != manifest:
            raise ValueError("bar dataset is immutable; select a different directory")
        return manifest_path
    directory.mkdir(parents=True, exist_ok=True)
    with bars_path.open("xb") as file:
        file.write(payload)
    with manifest_path.open("xb") as file:
        file.write(encoded_manifest)
    return manifest_path


def load_bar_dataset(directory: Path) -> tuple[list[HistoricalBar], dict[str, Any]]:
    """Verify schema, source-bound identity, file hash, row validation, ordering, and count."""
    try:
        manifest = _BarManifest.model_validate_json(
            _bounded_read(directory / "manifest.json", _MAX_MANIFEST_BYTES),
        ).model_dump(mode="json")
        identity = {key: value for key, value in manifest.items() if key != "data_id"}
        if content_hash(identity) != manifest["data_id"]:
            raise ValueError("bar dataset manifest identity mismatch")
        payload = _bounded_read(directory / "bars.jsonl", _MAX_DATASET_BYTES)
        if len(payload) != manifest["bytes"] or hashlib.sha256(payload).hexdigest() != manifest["content_hash"]:
            raise ValueError("bar dataset content hash mismatch")
        bars = [HistoricalBar.model_validate_json(line) for line in payload.splitlines()]
        if len(bars) != manifest["bar_count"]:
            raise ValueError("bar dataset count mismatch")
        if _ordered_bars(bars) != bars:
            raise ValueError("bar dataset rows are not deterministically ordered")
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"invalid bar dataset: {error}") from error
    return bars, manifest
