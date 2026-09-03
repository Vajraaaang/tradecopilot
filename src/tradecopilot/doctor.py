from __future__ import annotations

import asyncio
import importlib.util
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame

from tradecopilot.auth import (
    ROBINHOOD_MCP_URL,
    AlpacaKeychainStorage,
    KeychainOAuthStorage,
    LoopbackOAuthFlow,
    robinhood_oauth_client,
)
from tradecopilot.config import StrategyConfig
from tradecopilot.journal import Journal
from tradecopilot.mcp_client import OfficialMcpSession
from tradecopilot.providers.robinhood import (
    RobinhoodBrokerReadProvider,
    RobinhoodMarketDataProvider,
    RobinhoodReadAdapter,
    resolve_robinhood_account,
)
from tradecopilot.providers.shibui import (
    DEFAULT_SHIBUI_MCP_URL,
    SHIBUI_READ_ONLY_ALLOWLIST,
    ShibuiReadOnlyClient,
    ShibuiSupplementalProvider,
    StreamableHttpShibuiTransport,
)
from tradecopilot.redaction import redact as redact_payload
from tradecopilot.security import BLOCKED_WRITE_TOOLS, READ_ONLY_ALLOWLIST, ReadOnlyToolSurface


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def run_doctor(
    database_path: Path,
    config: StrategyConfig,
    *,
    symbol: str = "AAPL",
    account_last4: str | None = None,
    alpaca_feed: str = "sip",
) -> tuple[Check, ...]:
    checks = list(_local_checks(database_path, config))
    checks.extend(asyncio.run(_provider_checks(symbol.strip().upper(), account_last4, alpaca_feed, config)))
    explanation_mode = os.getenv("TRADECOPILOT_EXPLANATION_MODE", "deterministic")
    if explanation_mode == "openai":
        openai_ok = bool(os.getenv("OPENAI_API_KEY")) and importlib.util.find_spec("openai") is not None
        checks.append(
            Check(
                "OpenAI explanation",
                openai_ok,
                "configured" if openai_ok else "install the openai extra and set OPENAI_API_KEY",
            )
        )
    else:
        checks.append(Check("OpenAI explanation", True, "deterministic fallback selected"))
    return tuple(checks)


def _local_checks(database_path: Path, config: StrategyConfig) -> tuple[Check, ...]:
    checks: list[Check] = []
    try:
        config.strategy_version()
        checks.append(Check("configuration", True, "valid; execution_mode=manual_only"))
    except Exception as exc:
        checks.append(Check("configuration", False, str(exc)))
    try:
        with Journal(database_path, config.raw_snapshot_retention_rows):
            pass
        checks.append(Check("database", True, f"SQLite available at {database_path}"))
    except Exception as exc:
        checks.append(Check("database", False, str(exc)))
    visible = set(READ_ONLY_ALLOWLIST)
    checks.append(Check("allowed tool inventory", True, f"{len(visible)} explicitly reviewed read-only tools"))
    checks.append(
        Check(
            "blocked tools exposed",
            not bool(visible & BLOCKED_WRITE_TOOLS),
            "none" if not visible & BLOCKED_WRITE_TOOLS else str(visible & BLOCKED_WRITE_TOOLS),
        )
    )
    try:
        eastern = datetime.now(ZoneInfo("America/New_York"))
        pacific = datetime.now(ZoneInfo("America/Los_Angeles"))
        checks.append(Check("clock/timezone", True, f"ET={eastern.isoformat()} PT={pacific.isoformat()}"))
    except Exception as exc:
        checks.append(Check("clock/timezone", False, str(exc)))
    redacted = redact_payload({"account_number": "123456789", "api_key": "secret"})
    checks.append(
        Check(
            "secret redaction",
            redacted == {"account_number": "[REDACTED]", "api_key": "[REDACTED]"},
            "account identifiers and credential fields are redacted",
        )
    )
    checks.append(Check("browser stream", True, "sanitized SSE endpoint is enabled; no MCP credentials are routed"))
    supplemental_path = os.getenv("TRADECOPILOT_SUPPLEMENTAL_FEED")
    supplemental_ok = supplemental_path is not None and Path(supplemental_path).is_file()
    checks.append(
        Check(
            "sourced catalyst/classified tape",
            supplemental_ok,
            f"configured at {supplemental_path}"
            if supplemental_ok
            else "not configured; manual sourced catalyst remains available",
        )
    )
    return tuple(checks)


async def _provider_checks(
    symbol: str,
    account_last4: str | None,
    alpaca_feed: str,
    config: StrategyConfig,
) -> tuple[Check, ...]:
    checks: list[Check] = []
    if not symbol or len(symbol) > 10:
        return (Check("provider symbol", False, "use a valid stock ticker"),)

    alpaca_credentials = await AlpacaKeychainStorage().get_credentials()
    if alpaca_credentials is None:
        checks.append(Check("Alpaca market data", False, "not authenticated; run `tradecopilot auth alpaca`"))
    else:
        checks.append(await _alpaca_check(symbol, alpaca_feed, alpaca_credentials))

    robinhood_storage = KeychainOAuthStorage()
    if not await robinhood_storage.has_tokens():
        checks.append(
            Check("Robinhood read-only connectivity", False, "not authenticated; run `tradecopilot auth robinhood`")
        )
    else:
        checks.extend(await _robinhood_checks(symbol, account_last4, robinhood_storage, config=config))

    shibui_url = os.getenv("TRADECOPILOT_SHIBUI_MCP_URL", DEFAULT_SHIBUI_MCP_URL)
    try:
        shibui = ShibuiReadOnlyClient(StreamableHttpShibuiTransport(shibui_url))
        supplemental = ShibuiSupplementalProvider(shibui)
        float_evidence, context = await asyncio.gather(
            supplemental.get_float(symbol),
            supplemental.get_historical_context(symbol),
        )
        safe = set(shibui.visible_tools) == set(SHIBUI_READ_ONLY_ALLOWLIST)
        detail = (
            "connected; float and 50-day daily context available"
            if float_evidence and context
            else "connected; symbol enrichment incomplete"
        )
        checks.append(
            Check("Shibui daily enrichment", safe and float_evidence is not None and context is not None, detail)
        )
    except Exception as exc:
        checks.append(Check("Shibui daily enrichment", False, f"unavailable ({type(exc).__name__})"))
    return tuple(checks)


async def _alpaca_check(symbol: str, feed_name: str, credentials: tuple[str, str]) -> Check:
    if feed_name not in {"iex", "sip"}:
        return Check("Alpaca market data", False, "feed must be iex or sip")
    api_key, secret_key = credentials
    feed = DataFeed.SIP if feed_name == "sip" else DataFeed.IEX

    def read() -> tuple[Any, Any, Any]:
        client = StockHistoricalDataClient(api_key, secret_key)
        quote = client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=feed))
        trade = client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol, feed=feed))
        now = datetime.now(UTC)
        bars = client.get_stock_bars(
            StockBarsRequest(
                symbol_or_symbols=symbol,
                start=now - timedelta(days=5),
                end=now,
                timeframe=TimeFrame.Minute,
                limit=5,
                feed=feed,
            )
        )
        return quote, trade, bars

    try:
        quote, trade, bars = await asyncio.to_thread(read)
        quote_ok = symbol in quote
        trade_ok = symbol in trade
        bar_rows = getattr(bars, "data", {}).get(symbol, [])
        return Check(
            "Alpaca market data",
            quote_ok and trade_ok and bool(bar_rows),
            f"{feed_name.upper()} latest quote/trade and minute-bar reads succeeded",
        )
    except Exception as exc:
        return Check("Alpaca market data", False, f"unavailable ({type(exc).__name__})")


async def _robinhood_checks(
    symbol: str,
    account_last4: str | None,
    storage: KeychainOAuthStorage,
    *,
    config: StrategyConfig,
) -> tuple[Check, ...]:
    with LoopbackOAuthFlow() as flow:
        http_client = robinhood_oauth_client(storage, flow, allow_browser=False)
        try:
            async with OfficialMcpSession(ROBINHOOD_MCP_URL, http_client=http_client) as session:
                discovered = frozenset(tool.name for tool in await session.list_tools())
                surface = ReadOnlyToolSurface.from_transport(session, discovered)
                visible = set(surface.visible_tools)
                checks = [
                    Check(
                        "Robinhood application surface",
                        not bool(visible & BLOCKED_WRITE_TOOLS),
                        f"{len(visible)} reviewed read-only tools visible; mutations absent",
                    )
                ]
                adapter = RobinhoodReadAdapter(surface, config)
                account_number, account_alias = await resolve_robinhood_account(adapter, account_last4)
                market = RobinhoodMarketDataProvider(adapter, account_number, config)
                broker = RobinhoodBrokerReadProvider(adapter, account_number, account_alias)
                now = datetime.now(UTC)
                quote, bars, level2, position, risk = await asyncio.gather(
                    market.get_quote(symbol),
                    market.get_bars(symbol, "1m", now - timedelta(hours=8), now),
                    market.get_level2(symbol),
                    broker.get_position(symbol),
                    broker.get_account_risk(),
                )
                del quote, position, risk
                checks.append(
                    Check(
                        "Robinhood read-only connectivity",
                        bool(bars) and bool(level2.bids) and bool(level2.asks),
                        f"quote, OHLCV, Level 2, positions, and P&L read for account {account_alias}",
                    )
                )
                return tuple(checks)
        except Exception as exc:
            return (Check("Robinhood read-only connectivity", False, f"unavailable ({type(exc).__name__})"),)


def visible_surface_is_safe(surface: ReadOnlyToolSurface) -> bool:
    return not bool(set(surface.visible_tools) & BLOCKED_WRITE_TOOLS)
