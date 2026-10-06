import importlib.util
from pathlib import Path

import pytest


def _renderer():
    spec = importlib.util.spec_from_file_location(
        "render_kronos_results", Path(__file__).parents[1] / "scripts" / "render_kronos_results.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_summary_contains_aggregates_and_never_market_rows(tmp_path, monkeypatch):
    from tradecopilot.forecast.kronos_report import write_report

    metrics = {
        "eligible": 2,
        "scored": 1,
        "errors": 1,
        "direction_accuracy": 1.0,
        "direction_accuracy_errors_as_incorrect": 0.5,
        "terminal_mae_bps": 5.0,
        "path_mae_bps": 3.0,
        "raw_interval_coverage": 0.4,
        "class_order": ["DOWN", "FLAT", "UP"],
        "confusion_matrix": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    }
    report = {
        "evidence_mode": "retrospective_development_pilot",
        "connection": {"feed": "sip"},
        "source_data_id": "a" * 64,
        "protocol": {"dates": ["2026-10-06"], "symbols": ["AAPL"], "samples": 20},
        "catalog_counts": {"planned": 2, "eligible": 2, "excluded": 0},
        "models": {"mini": {"metadata": {"variant": "mini", "private_path": "DO_NOT_COPY"}, "metrics": metrics}},
        "cases": [{"group": "historical", "history_close": [123.456], "actual_close": [123.456]}],
        "broker_orders": 0,
        "jev_calls": 0,
        "limitations": ["development only"],
    }
    path = write_report(tmp_path / "report", report, {"paths.npz": b"private rows"})
    summary = _renderer().public_summary(path)
    assert summary["models"]["mini"]["metrics"]["direction_accuracy_errors_as_incorrect"] == 0.5
    assert "123.456" not in str(summary) and "DO_NOT_COPY" not in str(summary)
    assert "cases" not in summary and "inventory" not in summary
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    figure = pytest.importorskip("matplotlib.figure")
    titles = []
    original = figure.Figure.suptitle

    def capture(self, text, **kwargs):
        titles.append(text)
        return original(self, text, **kwargs)

    monkeypatch.setattr(figure.Figure, "suptitle", capture)
    _renderer().render(path, tmp_path / "public")
    assert "2 planned / 2 eligible" in titles[0]
    assert "1 stock" in titles[0] and "100-case" not in titles[0]
    assert (tmp_path / "public" / "comparison.png").stat().st_size > 1000
    report["evidence_mode"] = "synthetic_contract_fixture"
    fixture = write_report(tmp_path / "fixture", report, {})
    with pytest.raises(ValueError, match="real"):
        _renderer().public_summary(fixture)
