import importlib.util
import json
from datetime import UTC, datetime, timedelta
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


def _public_report():
    file = {"name": "config.json", "bytes": 225, "sha256": "a" * 64}
    weights = {"name": "model.safetensors", "bytes": 1234, "sha256": "b" * 64}
    metadata = {
        "variant": "mini", "schema_version": "kronos-local-engine-v1",
        "model": {"name": "Kronos-mini", "repository": "NeoQuasar/Kronos-mini", "revision": "c" * 40,
                  "files": [file, weights]},
        "tokenizer": {"name": "Kronos-Tokenizer-2k", "repository": "NeoQuasar/Kronos-Tokenizer-2k",
                      "revision": "d" * 40, "files": [dict(file), dict(weights)]},
        "source": {"repository": "https://github.com/shiyu-coder/Kronos", "commit": "e" * 40, "license": "MIT",
                   "files": {name: {"name": name, "bytes": 123, "sha256": "f" * 64}
                             for name in ("kronos.py", "module.py", "LICENSE")}},
        "device": "cpu", "dtype": "float32", "threads": 1, "max_context": 2048,
        "sampler": {"temperature": 1.0, "top_p": 0.9, "top_k": 0, "sample_count_per_series": 1,
                    "paths": "individual duplicated-input batch series", "seed_policy": "one CPU seed per call"},
        "model_parameter_count": 4108032, "tokenizer_parameter_count": 3958042,
        "dependencies": {"torch": "2.14.1", "pandas": "3.0.5", "huggingface-hub": "0.33.1",
                         "safetensors": "0.6.2", "einops": "0.8.1"},
        "amount_policy": "derived_proxy: volume * mean(open,high,low,close); not provider dollar volume",
        "fine_tuned": False,
    }
    metrics = {
        "eligible": 2, "scored": 1, "errors": 1, "forecast_availability": 0.5,
        "direction_accuracy": 1.0, "direction_accuracy_errors_as_incorrect": 0.5,
        "terminal_mae_bps": 5.0, "path_mae_bps": 3.0, "raw_interval_coverage": 0.4,
        "class_order": ["DOWN", "FLAT", "UP"], "confusion_matrix": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        "interval_nominal_mass": 0.8, "intervals_calibrated": False, "uncertainty_interval": None,
        "evidence": "retrospective_development_pilot; no confirmation or promotion",
    }
    protocol = {
        "schema_version": "kronos-pilot-protocol-v1", "registered_at": "2026-10-06T22:15:00+00:00",
        "dates": ["2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06"], "symbols": ["AAPL"],
        "lookback": 60, "horizon": 15, "offsets": [61, 121, 181, 241, 301], "samples": 20,
        "seed": 42, "flat_bps": 10, "max_seconds_per_model": 900, "source_data_id": "a" * 64,
        "case_ids_hash": "b" * 64, "protocol_id": "c" * 64, "dependency_lock_sha256": "d" * 64,
        "fine_tuning": False, "evidence": "retrospective_development_pilot", "broker_orders": 0, "jev_calls": 0,
        "prospectively_requested": True,
        "implementation": {"source_sha256": "e" * 64, "package_version": "0.1.0", "python": "3.13.5",
                           "dependencies": {"numpy": "2.5.2", "scikit-learn": "1.9.1",
                                            "exchange-calendars": "4.13.2", "pydantic": "2.13.4"}},
    }
    opening = datetime(2026, 10, 7, 13, 30, tzinfo=UTC)
    return {
        "evidence_mode": "retrospective_development_pilot", "connection": {"feed": "sip"},
        "source_data_id": "a" * 64, "protocol": protocol,
        "catalog_counts": {"planned": 2, "eligible": 2, "excluded": 0},
        "models": {"mini": {"metadata": metadata, "metrics": metrics,
                            "diagnostics": {"paths": 20, "rows": 300, "nonphysical_ohlc_rows": 5,
                                            "negative_volume_rows": 0, "negative_amount_rows": 0},
                            "elapsed_seconds": 12.5}},
        "cases": [{"group": "prospective", "symbol": "AAPL", "forecasts": {},
                   "future_times": [(opening + timedelta(minutes=i)).isoformat() for i in range(1, 16)]}],
        "prospective_unavailable": [], "broker_orders": 0, "jev_calls": 0,
        "limitations": ["Four-session development pilot is not confirmation.", "Raw sample bands are uncalibrated."],
    }


def test_public_summary_projects_nested_allowlists_without_private_sentinels(tmp_path):
    import copy

    from tradecopilot.forecast.kronos_report import write_report

    report = _public_report()
    expected = copy.deepcopy(report)
    sentinel = "PRIVATESENTINEL"
    model = report["models"]["mini"]
    containers = [report["protocol"], report["catalog_counts"], model["metrics"], model["diagnostics"],
                  report["protocol"]["implementation"], report["protocol"]["implementation"]["dependencies"],
                  model["metadata"]["model"], model["metadata"]["tokenizer"], model["metadata"]["source"],
                  model["metadata"]["sampler"], model["metadata"]["dependencies"],
                  model["metadata"]["source"]["files"], *model["metadata"]["model"]["files"],
                  *model["metadata"]["tokenizer"]["files"], *model["metadata"]["source"]["files"].values()]
    for container in containers:
        container["private_path"] = {"private_rows": [sentinel]}
    report["prospective_unavailable"] = [{"symbol": "AAPL", "reason": sentinel, "error": sentinel}]
    report["limitations"].append(sentinel)
    path = write_report(tmp_path / "report", report, {})
    summary = _renderer().public_summary(path)
    assert sentinel not in json.dumps(summary)
    assert summary["protocol"] == expected["protocol"]
    assert summary["catalog_counts"] == expected["catalog_counts"]
    assert summary["models"]["mini"]["metrics"] == expected["models"]["mini"]["metrics"]
    assert summary["models"]["mini"]["diagnostics"] == expected["models"]["mini"]["diagnostics"]
    for key in ("model", "tokenizer", "source", "sampler", "dependencies"):
        assert summary["models"]["mini"]["metadata"][key] == expected["models"]["mini"]["metadata"][key]
    assert summary["prospective"]["unavailable"] == [{"symbol": "AAPL", "reason": "unavailable_details_private"}]
    assert summary["limitations"][:2] == expected["limitations"]


@pytest.mark.parametrize("field", ["number", "version", "revision", "repository", "sampler", "dates", "classes",
                                  "missing_count"])
def test_public_summary_validates_known_fields_without_echoing_private_text(tmp_path, field):
    from tradecopilot.forecast.kronos_report import write_report

    report = _public_report()
    sentinel = "PRIVATESENTINEL"
    model = report["models"]["mini"]
    if field == "number":
        model["metrics"]["terminal_mae_bps"] = sentinel
    elif field == "version":
        model["metadata"]["dependencies"]["torch"] = sentinel
    elif field == "revision":
        model["metadata"]["model"]["revision"] = sentinel
    elif field == "repository":
        model["metadata"]["model"]["repository"] = sentinel
    elif field == "sampler":
        model["metadata"]["sampler"]["paths"] = sentinel
    elif field == "dates":
        report["protocol"]["dates"] = [sentinel]
    elif field == "classes":
        model["metrics"]["class_order"] = [sentinel]
    else:
        model["metrics"].pop("errors")
    path = write_report(tmp_path / "report", report, {})
    with pytest.raises(ValueError) as exc:
        _renderer().public_summary(path)
    assert sentinel not in str(exc.value)


def test_public_summary_preserves_initialization_failure_and_zero_scoring(tmp_path):
    from tradecopilot.forecast.kronos_report import write_report

    report = _public_report()
    model = report["models"]["mini"]
    model["metadata"] = {"variant": "mini", "initialization_status": "error"}
    model["initialization"] = {"status": "error", "phase": "initialization", "error_type": "RuntimeError"}
    model["metrics"] |= {"scored": 0, "errors": 2, "forecast_availability": 0,
                         "terminal_mae_bps": None, "path_mae_bps": None, "direction_accuracy": None,
                         "direction_accuracy_errors_as_incorrect": 0, "raw_interval_coverage": None,
                         "confusion_matrix": [[0] * 3 for _ in range(3)]}
    summary = _renderer().public_summary(write_report(tmp_path / "report", report, {}))
    public_model = summary["models"]["mini"]
    assert public_model["metrics"] == model["metrics"]
    assert public_model["initialization"] == model["initialization"]
    assert public_model["metadata"] == {"variant": "mini", "initialization_status": "error"}
