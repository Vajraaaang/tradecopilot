from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, Protocol, cast

from tradecopilot.mcp_client import OfficialMcpSession
from tradecopilot.models import CatalystEvidence, DataQuality, FloatEvidence, HistoricalContextEvidence

DEFAULT_SHIBUI_MCP_URL = "https://mcp.shibui.finance/mcp"
SHIBUI_READ_ONLY_ALLOWLIST = frozenset(
    {
        "get_database_schema",
        "get_query_patterns",
        "stock_data_query",
    }
)
SYMBOL_PATTERN = re.compile(r"[A-Z][A-Z0-9.-]{0,9}\Z")


class ShibuiToolDeniedError(PermissionError):
    pass


class ShibuiToolTransport(Protocol):
    async def call(self, tool_name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass
class StreamableHttpShibuiTransport:
    """Official-SDK Streamable HTTP transport for Shibui read-only queries."""

    endpoint: str = DEFAULT_SHIBUI_MCP_URL

    def __post_init__(self) -> None:
        if not self.endpoint.startswith("https://"):
            raise ValueError("Shibui MCP endpoint must use HTTPS")

    async def call(self, tool_name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        async with OfficialMcpSession(self.endpoint) as session:
            return await session.call(tool_name, arguments)


@dataclass
class ShibuiReadOnlyClient:
    """Deny-by-default surface; schema and SQL patterns are loaded before queries."""

    transport: ShibuiToolTransport
    _ready: bool = False
    _ready_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def visible_tools(self) -> tuple[str, ...]:
        return tuple(sorted(SHIBUI_READ_ONLY_ALLOWLIST))

    async def call(self, tool_name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        if tool_name not in SHIBUI_READ_ONLY_ALLOWLIST:
            raise ShibuiToolDeniedError(f"Shibui tool denied: {tool_name}")
        return await self.transport.call(tool_name, arguments)

    async def query(self, user_prompt: str, query: str) -> Sequence[Mapping[str, Any]]:
        await self._ensure_ready()
        payload = await self.call(
            "stock_data_query",
            {"user_prompt": user_prompt, "query": query},
        )
        structured = payload.get("structuredContent")
        if not isinstance(structured, Mapping):
            raise ValueError("Shibui query response has no structured content")
        rows = structured.get("result")
        if not isinstance(rows, list):
            raise ValueError("Shibui query response has no result array")
        return tuple(cast(Mapping[str, Any], row) for row in rows if isinstance(row, Mapping))

    async def _ensure_ready(self) -> None:
        if self._ready:
            return
        async with self._ready_lock:
            if self._ready:
                return
            await self.call("get_database_schema", {})
            await self.call("get_query_patterns", {})
            self._ready = True


@dataclass
class ShibuiSupplementalProvider:
    """Use Shibui only for sourced float; it is not an intraday news/tape feed."""

    client: ShibuiReadOnlyClient
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    monotonic_clock: Callable[[], float] = time.monotonic
    cache_seconds: float = 28_800.0
    _float_cache: dict[str, tuple[float, FloatEvidence]] = field(default_factory=dict)
    _context_cache: dict[str, tuple[float, HistoricalContextEvidence]] = field(default_factory=dict)

    async def get_float(self, symbol: str) -> FloatEvidence | None:
        normalized = _normalized_symbol(symbol)
        retrieved_at = self.clock()
        cached = self._float_cache.get(normalized)
        if cached is not None and self.monotonic_clock() - cached[0] <= self.cache_seconds:
            evidence = cached[1]
            return evidence.model_copy(
                update={
                    "receipt_timestamp": retrieved_at,
                    "age_seconds": (retrieved_at - evidence.provider_timestamp).total_seconds(),
                }
            )
        rows = await self.client.query(
            f"Read verified public float for {normalized} for a manual-only trading simulator.",
            (
                "SELECT ticker, shares_float "
                "FROM shibui.ownership_stats "
                f"WHERE ticker = '{normalized}' AND shares_float IS NOT NULL "
                "LIMIT 1"
            ),
        )
        if not rows:
            return None
        shares = int(rows[0].get("shares_float", 0))
        if shares <= 0:
            return None
        if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
            raise ValueError("Shibui provider clock must be timezone-aware")
        evidence = FloatEvidence(
            provider_timestamp=retrieved_at,
            receipt_timestamp=retrieved_at,
            age_seconds=0,
            source="shibui_finance:ownership_stats",
            quality=DataQuality.LIMITED,
            shares=shares,
            verified=True,
            reference=f"Shibui Finance snapshot retrieved {retrieved_at.isoformat()}",
        )
        self._float_cache[normalized] = (self.monotonic_clock(), evidence)
        return evidence

    async def get_catalyst(self, symbol: str) -> CatalystEvidence | None:
        del symbol
        return None

    async def get_historical_context(self, symbol: str) -> HistoricalContextEvidence | None:
        normalized = _normalized_symbol(symbol)
        retrieved_at = self.clock()
        cached = self._context_cache.get(normalized)
        if cached is not None and self.monotonic_clock() - cached[0] <= self.cache_seconds:
            evidence = cached[1]
            return evidence.model_copy(
                update={
                    "receipt_timestamp": retrieved_at,
                    "age_seconds": (retrieved_at - evidence.provider_timestamp).total_seconds(),
                }
            )
        rows = await self.client.query(
            f"Read the latest daily context and recent SEC filing metadata for {normalized}.",
            (
                "WITH ranked_prices AS ("
                "SELECT sq.date, sq.volume, ROW_NUMBER() OVER (ORDER BY sq.date DESC) AS rn "
                "FROM shibui.stock_quotes sq "
                f"WHERE sq.ticker = '{normalized}' AND sq.date >= CURRENT_DATE - INTERVAL '120 days'"
                "), daily AS ("
                "SELECT MAX(rp.date) AS latest_daily_date, CAST(AVG(rp.volume) AS BIGINT) AS average_volume_50d "
                "FROM ranked_prices rp WHERE rp.rn <= 50 HAVING COUNT(*) = 50"
                "), latest_filing AS ("
                "SELECT sf.acceptance_datetime, sf.raw_form_type, sf.filing_url "
                "FROM shibui.sec_filings sf "
                f"WHERE sf.cik = (SELECT g.cik FROM shibui.general_info g WHERE g.ticker = '{normalized}') "
                "AND sf.filing_date >= CURRENT_DATE - INTERVAL '2 years' "
                "ORDER BY sf.acceptance_datetime DESC LIMIT 1"
                ") "
                "SELECT d.latest_daily_date, d.average_volume_50d, lf.acceptance_datetime, "
                "lf.raw_form_type, lf.filing_url FROM daily d LEFT JOIN latest_filing lf ON TRUE LIMIT 1"
            ),
        )
        if not rows:
            return None
        row = rows[0]
        latest_daily = _date(row.get("latest_daily_date"))
        average_volume = int(row.get("average_volume_50d", 0))
        if average_volume <= 0:
            return None
        filing_timestamp = _optional_timestamp(row.get("acceptance_datetime"))
        evidence = HistoricalContextEvidence(
            provider_timestamp=retrieved_at,
            receipt_timestamp=retrieved_at,
            age_seconds=0,
            source="shibui_finance:daily_context",
            quality=DataQuality.LIMITED,
            symbol=normalized,
            average_daily_volume_50d=average_volume,
            latest_daily_date=latest_daily,
            ownership_snapshot_timestamp=retrieved_at,
            latest_filing_timestamp=filing_timestamp,
            latest_filing_type=_optional_text(row.get("raw_form_type")),
            latest_filing_url=_optional_text(row.get("filing_url")),
        )
        self._context_cache[normalized] = (self.monotonic_clock(), evidence)
        return evidence


def shibui_registration_config() -> dict[str, object]:
    """Non-secret Codex MCP registration material for diagnostics/documentation."""

    return {
        "name": "shibui-finance",
        "url": DEFAULT_SHIBUI_MCP_URL,
        "read_only_tools": sorted(SHIBUI_READ_ONLY_ALLOWLIST),
    }


def _normalized_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if not SYMBOL_PATTERN.fullmatch(normalized):
        raise ValueError("unsupported stock symbol format")
    return normalized


def _date(value: object) -> date:
    if not isinstance(value, str):
        raise ValueError("Shibui daily context has no latest daily date")
    return date.fromisoformat(value[:10])


def _optional_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
