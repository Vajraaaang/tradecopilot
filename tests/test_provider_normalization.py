from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from tradecopilot.config import StrategyConfig
from tradecopilot.providers.robinhood import (
    RobinhoodBrokerReadProvider,
    RobinhoodMarketDataProvider,
    RobinhoodReadAdapter,
)
from tradecopilot.providers.shibui import (
    SHIBUI_READ_ONLY_ALLOWLIST,
    ShibuiReadOnlyClient,
    ShibuiSupplementalProvider,
    ShibuiToolDeniedError,
    StreamableHttpShibuiTransport,
)
from tradecopilot.security import BLOCKED_WRITE_TOOLS, DISCOVERED_ROBINHOOD_TOOLS, ReadOnlyToolSurface

NOW = datetime(2026, 8, 10, 13, 35, tzinfo=UTC)


class RobinhoodFixtureTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call(self, tool_name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        args = dict(arguments)
        self.calls.append((tool_name, args))
        if tool_name == "get_equity_quotes":
            return {
                "data": {
                    "results": [
                        {
                            "quote": {
                                "adjusted_previous_close": "7.00",
                                "ask_price": "9.61",
                                "bid_price": "9.59",
                                "has_traded": True,
                                "last_non_reg_trade_price": "9.40",
                                "last_trade_price": "9.60",
                                "state": "active",
                                "symbol": "YXT",
                                "venue_last_non_reg_trade_time": "2026-08-10T13:00:00Z",
                                "venue_last_trade_time": "2026-08-10T13:34:59Z",
                            }
                        }
                    ]
                }
            }
        if tool_name == "get_equity_fundamentals":
            return {"data": {"results": [{"symbol": "YXT", "volume": "2400000", "open": "7.00"}]}}
        if tool_name == "get_equity_historicals":
            if args["interval"] == "day":
                bars = [
                    {
                        "begins_at": (NOW - timedelta(days=52 - index)).isoformat(),
                        "volume": 1000 + index,
                    }
                    for index in range(1, 52)
                ]
                bars.append({"begins_at": NOW.isoformat(), "volume": 99_000_000})
                return {"data": {"results": [{"symbol": "YXT", "bars": bars}]}}
            begins_at = "2026-08-10T13:34:00Z" if args["interval"] == "minute" else "2026-08-10T13:30:00Z"
            return {
                "data": {
                    "results": [
                        {
                            "symbol": "YXT",
                            "bars": [
                                {
                                    "begins_at": begins_at,
                                    "open_price": "9.50",
                                    "high_price": "9.62",
                                    "low_price": "9.48",
                                    "close_price": "9.60",
                                    "volume": 12_000,
                                    "session": "reg",
                                },
                                {
                                    "begins_at": begins_at,
                                    "open_price": "9.60",
                                    "high_price": "9.60",
                                    "low_price": "9.60",
                                    "close_price": "9.60",
                                    "volume": 0,
                                    "session": "reg",
                                    "interpolated": True,
                                },
                            ],
                        }
                    ]
                }
            }
        if tool_name == "get_equity_price_book":
            return {
                "data": {
                    "books": [
                        {
                            "symbol": "YXT",
                            "updated_at": "2026-08-10T13:34:59Z",
                            "bids": [{"price": "9.59", "quantity": 2_500}],
                            "asks": [{"price": "9.61", "quantity": 3_000}],
                        }
                    ]
                }
            }
        if tool_name == "get_equity_tradability":
            return {
                "data": {
                    "results": [
                        {
                            "symbol": "YXT",
                            "tradeable": True,
                            "state": "active",
                            "internal_halt_start_time": None,
                        }
                    ]
                }
            }
        if tool_name == "get_equity_positions":
            if args.get("cursor") == "second-page":
                return {
                    "data": {
                        "positions": [{"symbol": "YXT", "quantity": "100", "average_buy_price": "9.15"}],
                        "next": None,
                    }
                }
            return {
                "data": {
                    "positions": [{"symbol": "OTHER", "quantity": "1", "average_buy_price": "1"}],
                    "next": "https://broker.invalid/positions?cursor=second-page",
                }
            }
        if tool_name == "get_portfolio":
            return {"data": {"buying_power": {"buying_power": "1250.50"}}}
        if tool_name == "get_realized_pnl":
            return {"data": {"total_returns": "5.00"}}
        if tool_name == "get_pnl_trade_history":
            if args.get("cursor") == "older-page":
                return {
                    "data": {
                        "trades": [
                            {
                                "symbol": "YXT",
                                "timestamp": "2026-08-10T12:30:00Z",
                                "realized_gain": "20.00",
                            }
                        ],
                        "next_cursor": "",
                    }
                }
            return {
                "data": {
                    "trades": [
                        {
                            "symbol": "YXT",
                            "timestamp": "2026-08-10T13:34:00Z",
                            "realized_gain": "-5.00",
                        },
                        {
                            "symbol": "YXT",
                            "timestamp": "2026-08-10T13:00:00Z",
                            "realized_gain": "-10.00",
                        },
                    ],
                    "next_cursor": "older-page",
                }
            }
        if tool_name == "get_equity_orders":
            if args.get("cursor") == "orders-page-two":
                return {
                    "data": {
                        "orders": [{"id": "two", "symbol": "YXT", "state": "pending_cancelled"}],
                        "next": None,
                    }
                }
            return {
                "data": {
                    "orders": [
                        {"id": "one", "symbol": "YXT", "state": "new"},
                        {"id": "filled", "symbol": "YXT", "state": "filled"},
                    ],
                    "next": "https://broker.invalid/orders?cursor=orders-page-two",
                }
            }
        raise AssertionError(f"unexpected fixture tool: {tool_name}")


def _robinhood_providers():
    transport = RobinhoodFixtureTransport()
    surface = ReadOnlyToolSurface.from_transport(transport, DISCOVERED_ROBINHOOD_TOOLS)
    adapter = RobinhoodReadAdapter(surface, StrategyConfig())
    market = RobinhoodMarketDataProvider(adapter, "account-from-explicit-config", StrategyConfig(), clock=lambda: NOW)
    broker = RobinhoodBrokerReadProvider(adapter, "account-from-explicit-config", "••••0000", clock=lambda: NOW)
    return transport, market, broker


def test_robinhood_market_data_normalizes_exact_read_schemas() -> None:
    transport, market, _ = _robinhood_providers()

    async def read_all():
        return await asyncio.gather(
            market.get_quote("yxt"),
            market.get_bars("YXT", "1m", NOW - timedelta(hours=1), NOW),
            market.get_bars("YXT", "5m", NOW - timedelta(days=1), NOW),
            market.get_level2("YXT"),
            market.is_tradable("YXT"),
        )

    quote, bars_1m, bars_5m, level2, tradable = asyncio.run(read_all())
    assert quote.last == Decimal("9.60")
    assert quote.total_volume == 2_400_000
    assert quote.average_daily_volume_50d is None
    assert len(bars_1m) == len(bars_5m) == 1
    assert level2.bids[0].size == 2_500
    assert tradable is True
    assert not ({name for name, _ in transport.calls} & BLOCKED_WRITE_TOOLS)


def test_robinhood_broker_reads_paginate_and_never_mutate() -> None:
    transport, _, broker = _robinhood_providers()

    async def read_all():
        return await asyncio.gather(
            broker.get_position("YXT"),
            broker.get_positions(),
            broker.get_account_risk(),
            broker.get_trade_history(),
            broker.get_open_orders("YXT"),
        )

    position, positions, risk, history, orders = asyncio.run(read_all())
    assert position is not None and position.quantity == Decimal("100")
    assert {item.symbol for item in positions} == {"OTHER", "YXT"}
    assert {item.account_alias for item in positions} == {"••••0000"}
    assert risk.buying_power == Decimal("1250.50")
    assert risk.realized_session_pnl == Decimal("5.00")
    assert risk.peak_realized_session_pnl == Decimal("20.00")
    assert risk.consecutive_losses == 2
    assert len(history) == 3
    assert {order["id"] for order in orders} == {"one", "two"}
    assert not ({name for name, _ in transport.calls} & BLOCKED_WRITE_TOOLS)


class ShibuiFixtureTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call(self, tool_name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        self.calls.append((tool_name, dict(arguments)))
        if tool_name == "stock_data_query":
            return {"structuredContent": {"result": [{"ticker": "YXT", "shares_float": 2_900_000}]}}
        return {"structuredContent": {"result": "loaded"}}


def test_shibui_float_provider_loads_required_metadata_then_uses_fixed_read_query() -> None:
    transport = ShibuiFixtureTransport()
    client = ShibuiReadOnlyClient(transport)
    provider = ShibuiSupplementalProvider(client, clock=lambda: NOW)
    evidence = asyncio.run(provider.get_float("YXT"))
    cached = asyncio.run(provider.get_float("YXT"))
    assert evidence is not None and evidence.verified
    assert cached is not None and cached.shares == evidence.shares
    assert evidence.shares == 2_900_000
    assert evidence.quality.value == "LIMITED"
    assert [name for name, _ in transport.calls] == [
        "get_database_schema",
        "get_query_patterns",
        "stock_data_query",
    ]
    query = transport.calls[-1][1]["query"]
    assert query == (
        "SELECT ticker, shares_float FROM shibui.ownership_stats "
        "WHERE ticker = 'YXT' AND shares_float IS NOT NULL LIMIT 1"
    )
    assert set(client.visible_tools) == set(SHIBUI_READ_ONLY_ALLOWLIST)


def test_shibui_unknown_tools_fail_closed_before_transport() -> None:
    transport = ShibuiFixtureTransport()
    client = ShibuiReadOnlyClient(transport)
    with pytest.raises(ShibuiToolDeniedError):
        asyncio.run(client.call("export_to_excel", {}))
    assert transport.calls == []


def test_shibui_transport_rejects_non_https_endpoint() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        StreamableHttpShibuiTransport("http://example.com/mcp")
