import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest


def test_kronos_viewer_uses_integrity_loader_and_exposes_no_write_route(tmp_path):
    from tradecopilot.forecast.kronos_report import load_report, write_report
    from tradecopilot.forecast.service import create_server

    report = write_report(
        tmp_path / "report",
        {
            "evidence_mode": "retrospective_development_pilot",
            "connection": {"mode": "paper"},
            "models": {},
            "cases": [],
        },
        {"inputs.json": b"[]"},
    )
    server = create_server(report, port=0, loader=load_report, dashboard_name="kronos_dashboard.html")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/api/report") as r:
            assert json.load(r)["schema_version"] == "kronos-paper-report-v1"
        with urlopen(base + "/") as r:
            assert b"TradeCopilot" in r.read()
        with pytest.raises(HTTPError) as failure:
            urlopen(Request(base + "/api/report", data=b"{}", method="POST"))
        assert failure.value.code == 405
        (report.parent / "inputs.json").write_bytes(b"[1]")
        with pytest.raises(HTTPError) as failure:
            urlopen(base + "/api/report")
        assert failure.value.code == 503
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
