"""Read-only HTTP presentation of an existing, integrity-verified forecast bundle."""

from __future__ import annotations

import json
import secrets
import webbrowser
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from tradecopilot.forecast.experiment import load_report


def create_server(
    report_path: Path, host: str = "127.0.0.1", port: int = 8766,
    *, loader: Callable[[Path], dict[str, Any]] | None = None,
    dashboard_name: str = "dashboard.html",
) -> ThreadingHTTPServer:
    """Bind a report viewer; the caller owns its serving and shutdown lifecycle."""
    if host not in {"127.0.0.1", "0.0.0.0"}:
        raise ValueError("forecast service host must be 127.0.0.1 or 0.0.0.0")
    report_path = report_path.resolve()
    if dashboard_name not in {"dashboard.html", "kronos_dashboard.html"}:
        raise ValueError("unknown forecast dashboard")
    dashboard = files("tradecopilot.forecast").joinpath(dashboard_name).read_text(encoding="utf-8")

    class ReportHandler(BaseHTTPRequestHandler):
        server_version = "ForecastReport"
        sys_version = ""

        def do_GET(self) -> None:
            try:
                route = urlsplit(self.path).path
            except ValueError:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "Invalid request."})
                return
            if route == "/":
                nonce = secrets.token_urlsafe(24)
                body = dashboard.replace("__CSP_NONCE__", nonce).encode("utf-8")
                self._respond(HTTPStatus.OK, body, "text/html; charset=utf-8", nonce)
            elif route in {"/api/report", "/health"}:
                try:
                    report = (loader or load_report)(report_path)
                except (ValueError, OSError):
                    self._json(HTTPStatus.SERVICE_UNAVAILABLE, {
                        "error": "Report unavailable: bundle integrity verification failed.",
                    })
                    return
                payload = report if route == "/api/report" else {"ok": True, "write_capabilities": False}
                self._json(HTTPStatus.OK, payload)
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "Not found."})

        def do_HEAD(self) -> None:
            self.do_GET()

        def do_POST(self) -> None:
            self._json(HTTPStatus.METHOD_NOT_ALLOWED, {"error": "Method not allowed."})

        do_PUT = do_POST
        do_PATCH = do_POST
        do_DELETE = do_POST
        do_OPTIONS = do_POST

        def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
            self._json(code, {"error": "Invalid request."})

        def _json(self, status: int, value: dict[str, Any]) -> None:
            self._respond(status, json.dumps(value, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8")

        def _respond(self, status: int, body: bytes, content_type: str, nonce: str | None = None) -> None:
            policy = (
                "default-src 'none'; connect-src 'self'; base-uri 'none'; "
                "form-action 'none'; frame-ancestors 'none'; object-src 'none'"
            )
            if nonce:
                policy += f"; script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'"
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", policy)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
            if status == HTTPStatus.METHOD_NOT_ALLOWED:
                self.send_header("Allow", "GET, HEAD")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            # Keep report paths, query strings, and case contents out of access logs.
            return

    return ThreadingHTTPServer((host, port), ReportHandler)


def serve_report(
    report_path: Path, host: str = "127.0.0.1", port: int = 8766, open_browser: bool = False,
) -> None:
    """Serve saved evidence only; no inference, collection, or trading is performed."""
    load_report(report_path)
    server = create_server(report_path, host=host, port=port)
    try:
        if open_browser:
            webbrowser.open(f"http://127.0.0.1:{server.server_port}/")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
