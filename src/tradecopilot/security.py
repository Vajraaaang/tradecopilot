from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

READ_ONLY_ALLOWLIST = frozenset(
    {
        "get_accounts",
        "get_portfolio",
        "get_realized_pnl",
        "get_pnl_trade_history",
        "search",
        "get_equity_historicals",
        "get_equity_fundamentals",
        "get_equity_tax_lots",
        "get_equity_technical_indicators",
        "get_equity_price_book",
        "get_equity_positions",
        "get_equity_quotes",
        "get_equity_orders",
        "get_equity_tradability",
        "get_earnings_calendar",
        "get_earnings_results",
        "get_financials",
        "get_indexes",
        "get_index_quotes",
        "get_scanner_filter_specs",
        "get_scans",
        "run_scan",
    }
)


BLOCKED_WRITE_TOOLS = frozenset(
    {
        "add_option_to_watchlist",
        "add_to_watchlist",
        "cancel_equity_order",
        "cancel_option_exercise",
        "cancel_option_order",
        "create_scan",
        "create_watchlist",
        "exercise_option",
        "follow_watchlist",
        "place_equity_order",
        "place_option_order",
        "remove_from_watchlist",
        "remove_option_from_watchlist",
        "review_equity_order",
        "review_option_order",
        "unfollow_watchlist",
        "update_scan_config",
        "update_scan_filters",
        "update_watchlist",
    }
)


DISCOVERED_READ_TOOLS = frozenset(
    {
        "get_accounts",
        "get_earnings_calendar",
        "get_earnings_results",
        "get_equity_fundamentals",
        "get_equity_historicals",
        "get_equity_orders",
        "get_equity_positions",
        "get_equity_price_book",
        "get_equity_quotes",
        "get_equity_tax_lots",
        "get_equity_technical_indicators",
        "get_equity_tradability",
        "get_financials",
        "get_index_historicals",
        "get_index_quotes",
        "get_indexes",
        "get_option_chains",
        "get_option_historicals",
        "get_option_instruments",
        "get_option_level_upgrade_info",
        "get_option_orders",
        "get_option_positions",
        "get_option_quotes",
        "get_option_watchlist",
        "get_pnl_trade_history",
        "get_popular_watchlists",
        "get_portfolio",
        "get_realized_pnl",
        "get_scanner_filter_specs",
        "get_scans",
        "get_watchlist_items",
        "get_watchlists",
        "run_scan",
        "search",
    }
)


DISCOVERED_ROBINHOOD_TOOLS = DISCOVERED_READ_TOOLS | BLOCKED_WRITE_TOOLS


class ToolDeniedError(PermissionError):
    """Raised before transport invocation when a tool is not explicitly allowed."""


class RobinhoodTransport(Protocol):
    async def call(self, tool_name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class ReadOnlyToolSurface:
    """The only Robinhood surface application code may receive.

    The surface is deny-by-default. It cannot list or call mutation tools, and it
    does not provide a handle to the underlying complete MCP client.
    """

    _call: Callable[[str, Mapping[str, Any]], Awaitable[Mapping[str, Any]]]
    discovered_tools: frozenset[str]

    @classmethod
    def from_transport(cls, transport: RobinhoodTransport, discovered_tools: frozenset[str]) -> ReadOnlyToolSurface:
        return cls(transport.call, discovered_tools)

    @property
    def visible_tools(self) -> tuple[str, ...]:
        return tuple(sorted(self.discovered_tools & READ_ONLY_ALLOWLIST))

    async def call(self, tool_name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        if tool_name not in READ_ONLY_ALLOWLIST:
            raise ToolDeniedError(f"Robinhood tool denied: {tool_name}")
        if tool_name not in self.discovered_tools:
            raise ToolDeniedError(f"Robinhood tool unavailable or unreviewed: {tool_name}")
        return await self._call(tool_name, arguments)


class UnavailableRobinhoodTransport:
    def __init__(self, reason: str) -> None:
        self.reason = reason

    async def call(self, tool_name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        del tool_name, arguments
        raise ConnectionError(self.reason)
