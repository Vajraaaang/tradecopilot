import json

import pytest

from tradecopilot.cli import main


def test_demo_runs_without_credentials_or_network_and_is_repeatable(tmp_path, monkeypatch, capsys):
    import tradecopilot.forecast.jev as jev
    from tradecopilot.forecast.experiment import load_report

    def forbidden(*args, **kwargs):
        raise AssertionError("offline demo cannot call a provider")

    monkeypatch.setattr(jev.JevForecaster, "predict", forbidden)
    out = tmp_path / "demo"
    for _ in range(2):
        with pytest.raises(SystemExit) as status:
            main(["forecast", "demo", "--output-dir", str(out)])
        assert status.value.code == 0
    report = load_report(out / "run" / "report.json")
    assert report["evaluation_kind"] == "synthetic_demo"
    assert report["models"][-1]["execution"] == "fixture"
    assert all(model["metrics"]["api_cost_estimate_usd"] == 0 for model in report["models"])
    assert "report.json" in capsys.readouterr().out


def test_predict_requires_explicit_live_flag_before_loading_key(tmp_path, monkeypatch):
    from tradecopilot import auth

    async def forbidden():
        raise AssertionError("key must not be loaded before explicit opt-in")

    monkeypatch.setattr(auth, "load_jev_api_key", forbidden)
    with pytest.raises(SystemExit) as result:
        main(["forecast", "predict", "AAPL", "--output-dir", str(tmp_path / "pilot")])
    assert result.value.code == 2
    assert not (tmp_path / "pilot").exists()


def test_config_validation_precedes_data_access(tmp_path):
    invalid = tmp_path / "bad.json"
    invalid.write_text(json.dumps({"symbols": ["AAPL", "AAPL"]}))
    with pytest.raises(SystemExit) as result:
        main(["forecast", "build", "--config", str(invalid), "--output-dir", str(tmp_path / "dataset")])
    assert result.value.code == 2
