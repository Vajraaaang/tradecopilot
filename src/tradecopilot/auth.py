from __future__ import annotations

import asyncio
import getpass
import json
import queue
import threading
import webbrowser
from collections.abc import Awaitable, Callable
from contextlib import suppress
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import keyring
from keyring.errors import PasswordDeleteError
from mcp.client.auth import OAuthClientProvider
from mcp.client.streamable_http import create_mcp_http_client  # type: ignore[attr-defined]
from mcp.shared.auth import (
    AuthorizationCodeResult,
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthToken,
)
from pydantic import AnyUrl

ROBINHOOD_MCP_URL = "https://agent.robinhood.com/mcp/trading"
SHIBUI_MCP_URL = "https://mcp.shibui.finance/mcp"
KEYCHAIN_SERVICE = "tradecopilot.robinhood-mcp"
_TOKEN_KEY = "oauth-token"
_CLIENT_KEY = "oauth-client"
ALPACA_KEYCHAIN_SERVICE = "tradecopilot.alpaca-market-data"
_ALPACA_CREDENTIALS_KEY = "api-credentials"


class KeychainOAuthStorage:
    """Store Robinhood OAuth material only in the operating-system keychain."""

    def __init__(self, service_name: str = KEYCHAIN_SERVICE) -> None:
        self.service_name = service_name

    async def get_tokens(self) -> OAuthToken | None:
        raw = await asyncio.to_thread(keyring.get_password, self.service_name, _TOKEN_KEY)
        return OAuthToken.model_validate_json(raw) if raw else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        await asyncio.to_thread(
            keyring.set_password,
            self.service_name,
            _TOKEN_KEY,
            tokens.model_dump_json(exclude_none=True),
        )

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        raw = await asyncio.to_thread(keyring.get_password, self.service_name, _CLIENT_KEY)
        return OAuthClientInformationFull.model_validate_json(raw) if raw else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        await asyncio.to_thread(
            keyring.set_password,
            self.service_name,
            _CLIENT_KEY,
            client_info.model_dump_json(exclude_none=True),
        )

    async def clear(self) -> None:
        for key in (_TOKEN_KEY, _CLIENT_KEY):
            with suppress(PasswordDeleteError):
                await asyncio.to_thread(keyring.delete_password, self.service_name, key)

    async def has_tokens(self) -> bool:
        return await self.get_tokens() is not None


class AlpacaKeychainStorage:
    """Store Alpaca market-data keys outside files, logs, prompts, and browser state."""

    def __init__(self, service_name: str = ALPACA_KEYCHAIN_SERVICE) -> None:
        self.service_name = service_name

    async def set_credentials(self, api_key: str, secret_key: str) -> None:
        if not api_key.strip() or not secret_key.strip():
            raise ValueError("Alpaca API key and secret are required")
        payload = json.dumps({"api_key": api_key.strip(), "secret_key": secret_key.strip()}, separators=(",", ":"))
        await asyncio.to_thread(
            keyring.set_password,
            self.service_name,
            _ALPACA_CREDENTIALS_KEY,
            payload,
        )

    async def get_credentials(self) -> tuple[str, str] | None:
        raw = await asyncio.to_thread(keyring.get_password, self.service_name, _ALPACA_CREDENTIALS_KEY)
        if not raw:
            return None
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Alpaca keychain entry is invalid")
        api_key = parsed.get("api_key")
        secret_key = parsed.get("secret_key")
        if not isinstance(api_key, str) or not isinstance(secret_key, str):
            raise ValueError("Alpaca keychain entry is incomplete")
        return api_key, secret_key

    async def clear(self) -> None:
        with suppress(PasswordDeleteError):
            await asyncio.to_thread(keyring.delete_password, self.service_name, _ALPACA_CREDENTIALS_KEY)


def prompt_for_alpaca_credentials() -> tuple[str, str]:
    api_key = getpass.getpass("Alpaca API key (stored in macOS Keychain): ").strip()
    secret_key = getpass.getpass("Alpaca secret key (stored in macOS Keychain): ").strip()
    if not api_key or not secret_key:
        raise ValueError("Alpaca API key and secret are required")
    return api_key, secret_key


class LoopbackOAuthFlow:
    """One-use loopback OAuth callback with no browser or disk token storage."""

    def __init__(self, timeout_seconds: float = 300.0) -> None:
        self.timeout_seconds = timeout_seconds
        self._result: queue.Queue[AuthorizationCodeResult | Exception] = queue.Queue(maxsize=1)
        result_queue = self._result

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path != "/oauth/callback":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                values = parse_qs(parsed.query)
                if values.get("error"):
                    item: AuthorizationCodeResult | Exception = PermissionError("Robinhood authorization was denied")
                elif not values.get("code"):
                    item = ValueError("OAuth callback contained no authorization code")
                else:
                    item = AuthorizationCodeResult(
                        code=values["code"][0],
                        state=values.get("state", [None])[0],
                        iss=values.get("iss", [None])[0],
                    )
                with suppress(queue.Full):
                    result_queue.put_nowait(item)
                body = b"Authorization received. You can close this tab and return to Tradecopilot."
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        port = self._server.server_address[1]
        self.redirect_uri = f"http://127.0.0.1:{port}/oauth/callback"
        self._thread = threading.Thread(target=self._server.serve_forever, name="tradecopilot-oauth", daemon=True)

    def __enter__(self) -> LoopbackOAuthFlow:
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)

    async def open_authorization_url(self, url: str) -> None:
        opened = await asyncio.to_thread(webbrowser.open, url)
        if not opened:
            raise ConnectionError("Could not open the Robinhood authorization URL")

    async def wait_for_callback(self) -> AuthorizationCodeResult:
        try:
            result = await asyncio.to_thread(self._result.get, True, self.timeout_seconds)
        except queue.Empty as exc:
            raise TimeoutError("Robinhood OAuth callback timed out") from exc
        if isinstance(result, Exception):
            raise result
        return result


def robinhood_oauth_client(
    storage: KeychainOAuthStorage,
    flow: LoopbackOAuthFlow,
    *,
    allow_browser: bool = True,
) -> Any:
    metadata = OAuthClientMetadata(
        redirect_uris=[AnyUrl(flow.redirect_uri)],
        client_name="Tradecopilot analysis-only client",
        token_endpoint_auth_method="none",
        grant_types=["authorization_code", "refresh_token"],
    )
    provider = OAuthClientProvider(
        ROBINHOOD_MCP_URL,
        metadata,
        storage,
        redirect_handler=flow.open_authorization_url if allow_browser else _deny_interactive_oauth,
        callback_handler=flow.wait_for_callback if allow_browser else _deny_oauth_callback,
    )
    return create_mcp_http_client(auth=provider)


async def _deny_interactive_oauth(url: str) -> None:
    del url
    raise PermissionError("Robinhood credentials require refresh; run `tradecopilot auth robinhood`")


async def _deny_oauth_callback() -> AuthorizationCodeResult:
    raise PermissionError("Robinhood doctor cannot start an interactive OAuth callback")


def oauth_callbacks(
    flow: LoopbackOAuthFlow,
) -> tuple[Callable[[str], Awaitable[None]], Callable[[], Awaitable[AuthorizationCodeResult]]]:
    return flow.open_authorization_url, flow.wait_for_callback
