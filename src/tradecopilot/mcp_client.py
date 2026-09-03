from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, cast
from urllib.parse import urlparse

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent, Tool


class McpProtocolError(ConnectionError):
    """Raised when an MCP server returns an unusable or interactive result."""


@dataclass
class OfficialMcpSession:
    """Small persistent Streamable HTTP client built on the official MCP SDK."""

    endpoint: str
    http_client: Any | None = None
    terminate_on_close: bool = True
    _stack: AsyncExitStack | None = field(default=None, init=False)
    _session: ClientSession | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        parsed = urlparse(self.endpoint)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("MCP endpoints must use an absolute HTTPS URL")

    async def __aenter__(self) -> OfficialMcpSession:
        if self._session is not None:
            raise RuntimeError("MCP session is already open")
        stack = AsyncExitStack()
        try:
            if self.http_client is not None:
                await stack.enter_async_context(self.http_client)
            streams = await stack.enter_async_context(
                streamable_http_client(
                    self.endpoint,
                    http_client=self.http_client,
                    terminate_on_close=self.terminate_on_close,
                )
            )
            read_stream, write_stream = streams
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
        except BaseException:
            await stack.aclose()
            raise
        self._stack = stack
        self._session = session
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        stack = self._stack
        self._stack = None
        self._session = None
        if stack is not None:
            await stack.aclose()

    async def list_tools(self) -> tuple[Tool, ...]:
        session = self._require_session()
        result = await session.list_tools()
        return tuple(result.tools)

    async def call(self, tool_name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        session = self._require_session()
        result = await session.call_tool(tool_name, dict(arguments), allow_input_required=False)
        if not isinstance(result, CallToolResult):
            raise McpProtocolError("MCP tool requested unsupported interactive input")
        if result.is_error:
            raise McpProtocolError(f"MCP tool {tool_name} returned an error")
        structured = result.structured_content
        if structured is None:
            structured = _structured_from_text(result.content)
        payload: dict[str, Any] = {"structuredContent": structured}
        if isinstance(structured, Mapping):
            payload.update(cast(Mapping[str, Any], structured))
        return payload

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("MCP session is not open")
        return self._session


def _structured_from_text(content: list[Any]) -> object:
    for item in content:
        if not isinstance(item, TextContent):
            continue
        try:
            parsed = json.loads(item.text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, (Mapping, list)):
            return parsed
    raise McpProtocolError("MCP tool returned no structured JSON data")
