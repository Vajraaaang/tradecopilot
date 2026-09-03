from __future__ import annotations

import asyncio
import os
import threading
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime
from pathlib import Path

from tradecopilot.auth import (
    ROBINHOOD_MCP_URL,
    SHIBUI_MCP_URL,
    AlpacaKeychainStorage,
    KeychainOAuthStorage,
    LoopbackOAuthFlow,
    robinhood_oauth_client,
)
from tradecopilot.config import StrategyConfig
from tradecopilot.mcp_client import OfficialMcpSession
from tradecopilot.models import (
    CatalystEvidence,
    FloatEvidence,
    HistoricalContextEvidence,
    Level2Snapshot,
    MarketFrame,
    OHLCVBar,
    Quote,
    RunMode,
    ScreenerCandidate,
    TimeAndSalesPrint,
)
from tradecopilot.providers.alpaca import AlpacaStreamingProvider
from tradecopilot.providers.base import SupplementalDataProvider
from tradecopilot.providers.cadence import CadencedBrokerReadProvider, CadencedMarketDataProvider
from tradecopilot.providers.polling import PollingFrameProvider
from tradecopilot.providers.robinhood import (
    RobinhoodBrokerReadProvider,
    RobinhoodMarketDataProvider,
    RobinhoodReadAdapter,
    RobinhoodScannerProvider,
    resolve_robinhood_account,
)
from tradecopilot.providers.shibui import ShibuiReadOnlyClient, ShibuiSupplementalProvider
from tradecopilot.providers.supplemental import JsonSupplementalProvider
from tradecopilot.security import BLOCKED_WRITE_TOOLS, ReadOnlyToolSurface


@dataclass
class HybridLiveMarketProvider:
    """Alpaca owns intraday price/bars; Robinhood owns Level 2/tradability."""

    alpaca: AlpacaStreamingProvider
    robinhood: RobinhoodMarketDataProvider

    async def get_quote(self, symbol: str) -> Quote:
        return await self.alpaca.get_quote(symbol)

    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> Sequence[OHLCVBar]:
        return await self.alpaca.get_bars(symbol, timeframe, start, end)

    async def get_level2(self, symbol: str) -> Level2Snapshot:
        return await self.robinhood.get_level2(symbol)

    async def is_tradable(self, symbol: str) -> bool:
        return await self.robinhood.is_tradable(symbol)


@dataclass
class LiveMcpFrameProvider:
    """Own both MCP sessions and the Alpaca stream for the full monitor lifetime."""

    symbol: str
    account_last4: str | None
    config: StrategyConfig
    alpaca_feed: str = "sip"
    scan_title: str | None = None
    screener_limit: int = 10
    manual_float: FloatEvidence | None = None
    manual_catalyst: CatalystEvidence | None = None
    _symbol_lock: threading.RLock = dataclass_field(default_factory=threading.RLock, init=False)

    def __post_init__(self) -> None:
        self.symbol = _normalize_symbol(self.symbol)

    def select_symbol(self, symbol: str) -> bool:
        normalized = _normalize_symbol(symbol)
        with self._symbol_lock:
            self.symbol = normalized
        return True

    def _current_symbol(self) -> str:
        with self._symbol_lock:
            return self.symbol

    async def frames(self) -> AsyncIterator[MarketFrame]:
        credentials = await AlpacaKeychainStorage().get_credentials()
        if credentials is None:
            raise ConnectionError("Alpaca credentials are missing; run `tradecopilot auth alpaca`")
        api_key, secret_key = credentials
        alpaca = AlpacaStreamingProvider(api_key, secret_key, self.alpaca_feed)
        storage = KeychainOAuthStorage()
        with LoopbackOAuthFlow() as oauth_flow:
            robinhood_http = robinhood_oauth_client(storage, oauth_flow)
            try:
                async with (
                    OfficialMcpSession(ROBINHOOD_MCP_URL, http_client=robinhood_http) as robinhood_session,
                    OfficialMcpSession(SHIBUI_MCP_URL) as shibui_session,
                ):
                    discovered = frozenset(tool.name for tool in await robinhood_session.list_tools())
                    surface = ReadOnlyToolSurface.from_transport(robinhood_session, discovered)
                    if set(surface.visible_tools) & BLOCKED_WRITE_TOOLS:
                        raise PermissionError("Robinhood write tool reached the application surface")
                    adapter = RobinhoodReadAdapter(surface, self.config)
                    account_number, account_alias = await resolve_robinhood_account(adapter, self.account_last4)
                    robinhood_market = RobinhoodMarketDataProvider(adapter, account_number, self.config)
                    hybrid = HybridLiveMarketProvider(alpaca, robinhood_market)
                    market = CadencedMarketDataProvider(hybrid, self.config)
                    broker = CadencedBrokerReadProvider(
                        RobinhoodBrokerReadProvider(adapter, account_number, account_alias), self.config
                    )
                    supplemental_path = os.getenv("TRADECOPILOT_SUPPLEMENTAL_FEED")
                    file_supplemental = JsonSupplementalProvider(Path(supplemental_path)) if supplemental_path else None
                    supplemental = LiveSupplementalProvider(
                        ShibuiSupplementalProvider(ShibuiReadOnlyClient(shibui_session)),
                        self._current_symbol(),
                        self.manual_float,
                        self.manual_catalyst,
                        file_supplemental,
                    )
                    polling = PollingFrameProvider(
                        self._current_symbol(),
                        market,
                        broker,
                        supplemental,
                        LiveTapeProvider(alpaca, file_supplemental),
                        self.config,
                        mode=RunMode.LIVE,
                    )
                    scanner = RobinhoodScannerProvider(adapter)
                    candidates: tuple[ScreenerCandidate, ...] = ()
                    last_scan = float("-inf")
                    while True:
                        started = time.monotonic()
                        selected_symbol = self._current_symbol()
                        if selected_symbol != polling.symbol:
                            polling.select_symbol(selected_symbol)
                        frame = await polling.snapshot()
                        now = time.monotonic()
                        if now - last_scan >= self.config.screener_interval_seconds:
                            refreshed = await _safe_candidates(scanner, self.scan_title)
                            if refreshed is not None:
                                candidates = await _enrich_candidates(refreshed[: self.screener_limit], supplemental)
                            last_scan = now
                        yield frame.model_copy(update={"screener_candidates": candidates})
                        elapsed = time.monotonic() - started
                        await asyncio.sleep(max(0.0, self.config.poll_interval_seconds - elapsed))
            finally:
                await alpaca.close()


async def authenticate_robinhood() -> tuple[str, ...]:
    """Complete OAuth and list metadata only; never call any Robinhood tool."""

    storage = KeychainOAuthStorage()
    with LoopbackOAuthFlow() as flow:
        http_client = robinhood_oauth_client(storage, flow)
        async with OfficialMcpSession(ROBINHOOD_MCP_URL, http_client=http_client) as session:
            tools = await session.list_tools()
    return tuple(sorted(tool.name for tool in tools))


async def _safe_candidates(
    scanner: RobinhoodScannerProvider,
    scan_title: str | None,
) -> tuple[ScreenerCandidate, ...] | None:
    try:
        return await scanner.candidates(scan_title)
    except Exception:
        return None


async def _enrich_candidates(
    candidates: Sequence[ScreenerCandidate],
    supplemental: SupplementalDataProvider,
) -> tuple[ScreenerCandidate, ...]:
    async def enrich(candidate: ScreenerCandidate) -> ScreenerCandidate:
        float_evidence, context = await asyncio.gather(
            supplemental.get_float(candidate.symbol),
            supplemental.get_historical_context(candidate.symbol),
        )
        return candidate.model_copy(
            update={
                "float_shares": float_evidence.shares if float_evidence else None,
                "average_daily_volume_50d": context.average_daily_volume_50d if context else None,
            }
        )

    return tuple(await asyncio.gather(*(enrich(candidate) for candidate in candidates)))


@dataclass
class LiveSupplementalProvider:
    shibui: ShibuiSupplementalProvider
    manual_symbol: str
    manual_float: FloatEvidence | None = None
    manual_catalyst: CatalystEvidence | None = None
    file: JsonSupplementalProvider | None = None

    async def get_float(self, symbol: str) -> FloatEvidence | None:
        if self.manual_float is not None and symbol.strip().upper() == self.manual_symbol:
            return self.manual_float
        if self.file is not None:
            file_evidence = await self.file.get_float(symbol)
            if file_evidence is not None:
                return file_evidence
        return await self.shibui.get_float(symbol)

    async def get_catalyst(self, symbol: str) -> CatalystEvidence | None:
        if symbol.strip().upper() == self.manual_symbol and self.manual_catalyst is not None:
            return self.manual_catalyst
        if self.file is not None:
            return await self.file.get_catalyst(symbol)
        return None

    async def get_historical_context(self, symbol: str) -> HistoricalContextEvidence | None:
        return await self.shibui.get_historical_context(symbol)


@dataclass
class LiveTapeProvider:
    alpaca: AlpacaStreamingProvider
    file: JsonSupplementalProvider | None = None

    async def get_prints(self, symbol: str) -> Sequence[TimeAndSalesPrint] | None:
        if self.file is not None:
            file_prints = await self.file.get_prints(symbol)
            if file_prints is not None:
                return file_prints
        return await self.alpaca.get_prints(symbol)


def _normalize_symbol(value: str) -> str:
    normalized = value.strip().upper()
    valid = (
        normalized
        and len(normalized) <= 10
        and all(character.isalnum() or character in ".-" for character in normalized)
    )
    if not valid:
        raise ValueError("unsupported stock symbol format")
    return normalized
