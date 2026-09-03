from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from tradecopilot.redaction import redact, redact_text
from tradecopilot.security import (
    BLOCKED_WRITE_TOOLS,
    DISCOVERED_ROBINHOOD_TOOLS,
    READ_ONLY_ALLOWLIST,
    ReadOnlyToolSurface,
    ToolDeniedError,
)


class SpyTransport:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def call(self, tool_name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        del arguments
        self.calls.append(tool_name)
        return {"data": {}}


def test_every_write_capable_robinhood_tool_is_absent_from_visible_surface() -> None:
    transport = SpyTransport()
    surface = ReadOnlyToolSurface.from_transport(transport, DISCOVERED_ROBINHOOD_TOOLS)
    assert not (set(surface.visible_tools) & BLOCKED_WRITE_TOOLS)
    assert set(surface.visible_tools) == set(READ_ONLY_ALLOWLIST)


def test_unknown_mcp_tool_fails_closed() -> None:
    transport = SpyTransport()
    surface = ReadOnlyToolSurface.from_transport(transport, DISCOVERED_ROBINHOOD_TOOLS)
    with pytest.raises(ToolDeniedError, match="denied"):
        asyncio.run(surface.call("new_unreviewed_tool", {}))
    assert transport.calls == []


@pytest.mark.parametrize("tool_name", sorted(BLOCKED_WRITE_TOOLS))
def test_no_write_or_order_tool_is_ever_called(tool_name: str) -> None:
    transport = SpyTransport()
    surface = ReadOnlyToolSurface.from_transport(transport, DISCOVERED_ROBINHOOD_TOOLS)
    with pytest.raises(ToolDeniedError):
        asyncio.run(surface.call(tool_name, {}))
    assert transport.calls == []


def test_allowed_read_call_reaches_transport_once() -> None:
    transport = SpyTransport()
    surface = ReadOnlyToolSurface.from_transport(transport, DISCOVERED_ROBINHOOD_TOOLS)
    asyncio.run(surface.call("get_accounts", {}))
    assert transport.calls == ["get_accounts"]


def test_inventory_matches_live_discovery_count() -> None:
    assert len(DISCOVERED_ROBINHOOD_TOOLS) == 53
    assert len(BLOCKED_WRITE_TOOLS) == 19


def test_secrets_and_account_identifiers_are_redacted() -> None:
    payload = {
        "account_number": "sensitive-value",
        "nested": {"api_key": "also-sensitive", "message": "Bearer abc.def"},
    }
    result = redact(payload)
    assert result["account_number"] == "[REDACTED]"
    assert result["nested"]["api_key"] == "[REDACTED]"
    assert "abc.def" not in result["nested"]["message"]
    assert "123456789" not in redact_text("identifier 123456789")
