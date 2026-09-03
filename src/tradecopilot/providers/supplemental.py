from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from tradecopilot.models import (
    CatalystEvidence,
    DataQuality,
    FloatEvidence,
    HistoricalContextEvidence,
    TimeAndSalesPrint,
)

MAX_FEED_BYTES = 1_000_000


class JsonSupplementalProvider:
    """Read sourced float, news, and tape evidence from a locally managed JSON feed."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._mtime_ns: int | None = None
        self._payload: Mapping[str, Any] = {}

    async def get_float(self, symbol: str) -> FloatEvidence | None:
        raw = self._symbol(symbol).get("float")
        if not isinstance(raw, Mapping):
            return None
        provider_timestamp, receipt_timestamp = _timestamps(raw)
        source = _required_text(raw, "source")
        return FloatEvidence(
            provider_timestamp=provider_timestamp,
            receipt_timestamp=receipt_timestamp,
            age_seconds=(receipt_timestamp - provider_timestamp).total_seconds(),
            source=f"supplemental:{source}",
            quality=DataQuality.GOOD,
            shares=int(raw["shares"]),
            verified=True,
            reference=source,
        )

    async def get_catalyst(self, symbol: str) -> CatalystEvidence | None:
        raw = self._symbol(symbol).get("news")
        if not isinstance(raw, Mapping):
            return None
        provider_timestamp, receipt_timestamp = _timestamps(raw)
        source = _required_text(raw, "source")
        return CatalystEvidence(
            provider_timestamp=provider_timestamp,
            receipt_timestamp=receipt_timestamp,
            age_seconds=(receipt_timestamp - provider_timestamp).total_seconds(),
            source=f"supplemental:{source}",
            quality=DataQuality.GOOD,
            description=_required_text(raw, "headline"),
            verified=True,
            reference=source,
        )

    async def get_historical_context(self, symbol: str) -> HistoricalContextEvidence | None:
        del symbol
        return None

    async def get_prints(self, symbol: str) -> Sequence[TimeAndSalesPrint] | None:
        rows = self._symbol(symbol).get("time_and_sales")
        if rows is None:
            return None
        if not isinstance(rows, list):
            raise ValueError("time_and_sales must be a JSON array")
        receipt_timestamp = datetime.now(UTC)
        output: list[TimeAndSalesPrint] = []
        for raw in rows:
            if not isinstance(raw, Mapping):
                raise ValueError("each time_and_sales row must be an object")
            timestamp = _timestamp(raw.get("timestamp"))
            if timestamp > receipt_timestamp:
                raise ValueError("supplemental tape timestamp cannot be in the future")
            side = _required_text(raw, "side").lower()
            if side not in {"buy", "sell", "unknown"}:
                raise ValueError("time_and_sales side must be buy, sell, or unknown")
            source = _required_text(raw, "source")
            output.append(
                TimeAndSalesPrint(
                    provider_timestamp=timestamp,
                    receipt_timestamp=receipt_timestamp,
                    age_seconds=(receipt_timestamp - timestamp).total_seconds(),
                    source=f"supplemental:{source}",
                    quality=DataQuality.GOOD,
                    symbol=symbol.upper(),
                    price=Decimal(str(raw["price"])),
                    size=int(raw["size"]),
                    side=side,
                )
            )
        return tuple(sorted(output, key=lambda item: item.provider_timestamp))

    def _symbol(self, symbol: str) -> Mapping[str, Any]:
        payload = self._load()
        symbols = payload.get("symbols")
        if not isinstance(symbols, Mapping):
            raise ValueError("supplemental feed must contain a symbols object")
        raw = symbols.get(symbol.upper(), {})
        if not isinstance(raw, Mapping):
            raise ValueError(f"supplemental feed entry for {symbol.upper()} must be an object")
        return raw

    def _load(self) -> Mapping[str, Any]:
        stat = self.path.stat()
        if stat.st_size > MAX_FEED_BYTES:
            raise ValueError("supplemental feed exceeds 1 MB")
        if self._mtime_ns == stat.st_mtime_ns:
            return self._payload
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("supplemental feed must be a JSON object")
        self._payload = raw
        self._mtime_ns = stat.st_mtime_ns
        return self._payload


def _timestamps(raw: Mapping[str, Any]) -> tuple[datetime, datetime]:
    provider_timestamp = _timestamp(raw.get("timestamp"))
    receipt_timestamp = datetime.now(UTC)
    if provider_timestamp > receipt_timestamp:
        raise ValueError("supplemental evidence timestamp cannot be in the future")
    return provider_timestamp, receipt_timestamp


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("supplemental evidence requires an ISO-8601 timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("supplemental evidence timestamps must include a timezone")
    return parsed


def _required_text(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"supplemental evidence requires {key}")
    return value.strip()
