import json
import re
import shutil
import threading
from contextlib import contextmanager
from http.client import HTTPConnection

import pytest

from tradecopilot.forecast.contracts import ForecastConfig
from tradecopilot.forecast.dataset import build_dataset
from tradecopilot.forecast.demo import synthetic_observations
from tradecopilot.forecast.experiment import load_report, run_experiment


@pytest.fixture(scope="module")
def report_bundle(tmp_path_factory):
    config = ForecastConfig(symbols=("AAPL",))
    manifest, examples = build_dataset(synthetic_observations(config), config)
    return run_experiment(manifest, examples, tmp_path_factory.mktemp("report") / "run")


@contextmanager
def running_server(report_path):
    from tradecopilot.forecast.service import create_server

    server = create_server(report_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def request(server, path, method="GET"):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        connection.request(method, path)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_serves_verified_report_and_read_only_health(report_bundle):
    with running_server(report_bundle) as server:
        status, headers, body = request(server, "/api/report")
        assert status == 200
        assert headers["Content-Type"] == "application/json; charset=utf-8"
        assert json.loads(body) == load_report(report_bundle)
        status, _, body = request(server, "/health")
        assert status == 200
        assert json.loads(body) == {"ok": True, "write_capabilities": False}


def test_dashboard_is_packaged_and_script_policy_matches_document(report_bundle):
    with running_server(report_bundle) as server:
        status, headers, body = request(server, "/")
        assert status == 200
        assert headers["Content-Type"] == "text/html; charset=utf-8"
        assert b"Forecast observatory" in body
        assert b"/api/report" in body
        assert b"<script src=" not in body
        assert b"<link" not in body
        nonces = re.findall(rb'<(?:script|style) nonce="([A-Za-z0-9_-]+)"', body)
        assert len(nonces) == 2
        assert nonces[0] == nonces[1]
        assert f"'nonce-{nonces[0].decode()}'" in headers["Content-Security-Policy"]
        assert "unsafe-inline" not in headers["Content-Security-Policy"]
        assert headers["Cache-Control"] == "no-store"
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["Referrer-Policy"] == "no-referrer"
        assert headers["X-Frame-Options"] == "DENY"
        assert "Access-Control-Allow-Origin" not in headers


def test_rechecks_bundle_integrity_on_every_request(tmp_path, report_bundle):
    copied = tmp_path / "copy"
    shutil.copytree(report_bundle.parent, copied)
    with running_server(copied / "report.json") as server:
        assert request(server, "/health")[0] == 200
        (copied / "predictions.jsonl").write_text("tampered SECRET_CONTENT", encoding="utf-8")
        for route in ("/api/report", "/health"):
            status, headers, body = request(server, route)
            assert status == 503
            assert json.loads(body) == {"error": "Report unavailable: bundle integrity verification failed."}
            assert str(tmp_path).encode() not in body
            assert b"SECRET_CONTENT" not in body
            assert headers["Cache-Control"] == "no-store"


@pytest.mark.parametrize("state", ["missing", "malformed", "incomplete"])
def test_unavailable_bundle_fails_closed(tmp_path, state):
    path = tmp_path / "private-report.json"
    if state == "malformed":
        path.write_text("not-json PRIVATE_INFORMATION", encoding="utf-8")
    elif state == "incomplete":
        path.write_text(json.dumps({"schema_version": "forecast-report-v1"}), encoding="utf-8")
    with running_server(path) as server:
        for route in ("/api/report", "/health"):
            status, _, body = request(server, route)
            assert status == 503
            assert b"private-report" not in body
            assert b"PRIVATE_INFORMATION" not in body
        assert request(server, "/")[0] == 200


@pytest.mark.parametrize("path", ["/report.json", "/predictions.jsonl", "/../secret", "/%2e%2e/secret", "/api/"])
def test_only_exact_routes_are_served(report_bundle, path):
    with running_server(report_bundle) as server:
        status, _, body = request(server, path)
        assert status == 404
        assert json.loads(body) == {"error": "Not found."}


def test_rejects_write_methods_and_returns_no_head_body(report_bundle):
    with running_server(report_bundle) as server:
        status, headers, body = request(server, "/api/report", "POST")
        assert status == 405
        assert headers["Allow"] == "GET, HEAD"
        assert json.loads(body) == {"error": "Method not allowed."}
        status, _, body = request(server, "/health", "HEAD")
        assert status == 200
        assert body == b""


@pytest.mark.parametrize("host", ["localhost", "::", "::1", "example.org", "192.168.1.1"])
def test_requires_explicit_supported_bind_address(tmp_path, host):
    from tradecopilot.forecast.service import create_server

    with pytest.raises(ValueError, match=r"127\.0\.0\.1 or 0\.0\.0\.0"):
        create_server(tmp_path / "report.json", host=host, port=0)


def test_serve_report_verifies_bundle_before_binding(monkeypatch, tmp_path):
    from tradecopilot.forecast import service

    def unexpected_bind(*args, **kwargs):
        raise AssertionError("An invalid report must be rejected before opening a listening socket")

    monkeypatch.setattr(service, "create_server", unexpected_bind)
    with pytest.raises(ValueError, match="integrity"):
        service.serve_report(tmp_path / "missing.json")


def test_serve_closes_server_and_opens_loopback_url_when_requested(monkeypatch, report_bundle):
    from tradecopilot.forecast import service

    events = []

    class FakeServer:
        server_port = 12345

        def serve_forever(self):
            events.append("serve")
            raise KeyboardInterrupt

        def server_close(self):
            events.append("close")

    server = FakeServer()
    monkeypatch.setattr(service, "create_server", lambda *args, **kwargs: server)
    monkeypatch.setattr(service.webbrowser, "open", lambda url: events.append(url))
    service.serve_report(report_bundle, host="0.0.0.0", port=0, open_browser=True)
    assert events == ["http://127.0.0.1:12345/", "serve", "close"]
