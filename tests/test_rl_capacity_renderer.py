"""Synthetic aggregate-only renderer contracts; no real study artifacts."""

import importlib.util
import json
from pathlib import Path

import pytest


def renderer():
    path = Path(__file__).resolve().parents[1] / "scripts" / "render_rl_capacity_results.py"
    spec = importlib.util.spec_from_file_location("rl_capacity_renderer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(net):
    return {
        "valid": True,
        "mean_daily_return": net,
        "max_episode_drawdown": 0.001,
        "episodes": 25,
        "dates": 5,
        "invalid_episodes": 0,
        "trade_count": 10,
        "turnover": 2000,
        "fees": 0,
        "execution_drag": 2,
        "daily_returns": {f"private-day-{i}": net for i in range(5)},
        "secret": "/private/checkpoint/model.zip",
        "ledger": [{"price": 999}],
    }


def report_fixture():
    tune, runs = [], []
    for candidate, value in [("ppo-64", 0.001), ("ppo-256", 0.002), ("recurrent-256", 0.003)]:
        for seed in [42, 43, 44]:
            tune.append({"candidate": candidate, "seed": seed, "metrics": metrics(value)})
            runs.append(
                {
                    "candidate_key": candidate,
                    "seed": seed,
                    "n_parameters": 100,
                    "actual_timesteps": 10240,
                    "elapsed_seconds": 10,
                    "status": "complete",
                    "device": "cpu",
                }
            )
    return {
        "report_id": "report-hash",
        "registration_id": "reg-hash",
        "prepared_data_id": "data-hash",
        "budget_id": "budget-hash",
        "scope": "MARKET_ONLY",
        "status": "inconclusive_not_promoted",
        "selection": {
            "candidate": "recurrent-256",
            "seed": 42,
            "strongest_control": "rule",
            "selection_id": "selection-hash",
        },
        "training": runs,
        "tune": tune,
        "test_selected_architecture_seeds": {str(seed): metrics(-0.001) for seed in [42, 43, 44]},
        "test_controls": {"cash": metrics(0), "hold": metrics(0.001), "rule": metrics(0.002)},
        "cost_stress": {str(cost): {"neural": metrics(-0.002), "control": metrics(0.001)} for cost in [4, 8]},
        "improvement_gate": {"passed": False, "checks": {"positive_primary": False}, "intervals": {}},
        "inventory": {"study-seal.json": "hash"},
    }


def test_public_allowlist_honest_status_and_return_units():
    module = renderer()
    report = report_fixture()
    public = module.public_summary(report, {"splits": {"train": ["a", "b"], "tune": ["c", "d"], "test": ["e", "f"]}})
    serialized = json.dumps(public)
    for forbidden in ("daily_returns", "private-day", "/private/", "model.zip", "secret", "ledger", "price"):
        assert forbidden not in serialized
    assert public["status"] == "inconclusive_not_promoted"
    assert public["metric"] == "mean daily net policy return in basis points"
    assert public["primary_test"]["mean_daily_return_bps"] == -10
    assert len(public["tune_architectures"]) == 3
    assert all(len(row["seeds"]) == 3 for row in public["tune_architectures"])
    assert public["cost_stress"]["4"]["incremental_interval_bps"]["mean"] == pytest.approx(-30)


def test_render_fixture_png_and_json_only(tmp_path, monkeypatch):
    module = renderer()
    report_path = tmp_path / "report.json"
    report_path.write_text("{}")
    (tmp_path / "study-seal.json").write_text(
        json.dumps({"registration": {"splits": {"train": ["a", "b"], "tune": ["c", "d"], "test": ["e", "f"]}}})
    )
    monkeypatch.setattr(module, "load_report", lambda _: report_fixture())
    output = tmp_path / "public"
    result = module.render(report_path, output)
    assert result.name == "summary.json"
    assert {p.suffix for p in output.iterdir()} == {".json", ".png"}
    assert len(list(output.glob("*.png"))) == 4
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in output.glob("*.png"))
    with pytest.raises(ValueError, match="fresh"):
        module.render(report_path, output)
