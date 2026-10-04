import json

import pytest

from tradecopilot.forecast.contracts import content_hash
from tradecopilot.rl import study


def seal(path, value, key):
    value = dict(value)
    value[key] = content_hash(value)
    path.write_text(json.dumps(value))
    return value


def test_existing_output_refused(tmp_path):
    with pytest.raises(ValueError, match="immutable"):
        study.run_study(tmp_path, tmp_path / "reg", tmp_path / "budget", tmp_path)


def test_training_guard_rejects_tampered_checkpoint(tmp_path):
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"tampered")
    result = {"checkpoint_sha256": "incorrect"}
    with pytest.raises(ValueError, match="checkpoint"):
        study._verify_checkpoint(tmp_path, result)


def test_budget_caps_and_identity(tmp_path):
    reg = {"registration_id": "reg"}
    for steps in (True, 10241, 100096):
        with pytest.raises(ValueError):
            study._validate_budget(
                {"registration_id": "reg", "shared_steps": steps, "max_seconds": 600, "device": "cpu"}, reg
            )
    with pytest.raises(ValueError, match="identity"):
        study._validate_budget(
            {"registration_id": "other", "shared_steps": 10240, "max_seconds": 600, "device": "cpu"}, reg
        )


def test_fairness_requires_nine_complete_identical_runs():
    with pytest.raises(ValueError, match="complete"):
        study._verify_training([], {}, {}, {}, None)


def test_invalid_or_unresolved_blocks_improvement():
    primary = {"valid": False, "mean_daily_return": None}
    result = study._improvement(primary, {}, {}, [], 0.01)
    assert not result["passed"]


@pytest.fixture
def registered(tmp_path, monkeypatch):
    reg = {
        "candidates": list(study.training._CANDIDATES.values()),
        "seeds": [42, 43, 44],
        "optimizer": study.training.OPTIMIZER,
        "environment": {
            "initial_cash": "10000",
            "notional_cap": "1000",
            "loss_lock": "100",
            "base_cost_bps_per_side": 2,
            "fee_per_fill": "0",
            "capacity_fraction": "0.01",
            "quantity_quantum": "0.000001",
            "warmup_minutes": 60,
            "latency_minutes": 1,
            "forced_close_buffer_minutes": 5,
        },
        "rule": {"enter_when_return_5m_bps_above": 5},
    }
    reg_path = tmp_path / "registration.json"
    reg = seal(reg_path, reg, "registration_id")
    budget_path = tmp_path / "budget.json"
    seal(
        budget_path,
        {
            "registration_id": reg["registration_id"],
            "prepared_data_id": "prepared",
            "shared_steps": 10240,
            "max_seconds": 600,
            "device": "cpu",
        },
        "budget_id",
    )
    manifest = {
        "registration_id": reg["registration_id"],
        "data_id": "prepared",
        "feature_names": ["return_5m_bps", "missing_return_5m_bps"],
        "normalizer": {"feature_names": ["return_5m_bps"], "mean": [0], "scale": [1]},
    }
    events = []
    monkeypatch.setattr(study.training, "_source_provenance", lambda: {"study.py": "fixed"})
    monkeypatch.setattr(study, "_lock_hash", lambda: "lock")

    def load(directory, role, selection_path=None):
        events.append(role)
        if role == "test":
            selection = study.training._read_sealed(selection_path, "selection_id")
            assert selection["stage"] == "selection_frozen"
            assert events.count("worker") == 9
        return [], manifest

    monkeypatch.setattr(study, "load_prepared", load)

    def worker(prepared, registration, budget_path, candidate, seed, output, budget, progress):
        events.append("worker")
        assert events.count("tune") == 0
        output.mkdir()
        (output / "model.zip").write_bytes(f"{candidate}-{seed}".encode())
        result = {
            "candidate_key": candidate,
            "candidate": study.training._CANDIDATES[candidate],
            "seed": seed,
            "status": "complete",
            "actual_timesteps": budget["shared_steps"],
            "n_updates": 800,
            "registration_id": reg["registration_id"],
            "prepared_data_id": "prepared",
            "budget_id": budget["budget_id"],
            "config_hash": study.training.registered_config(reg).content_hash,
            "source_provenance": {"study.py": "fixed"},
            "lock_sha256": "lock",
            "versions": {"torch": "fake"},
            "checkpoint_sha256": study.sha256((output / "model.zip").read_bytes()).hexdigest(),
            "n_parameters": 5,
            "elapsed_seconds": 1,
        }
        return seal(output / "result.json", result, "training_id")

    monkeypatch.setattr(study, "_run_worker", worker)
    monkeypatch.setattr(study, "_neural", lambda directory, run, device: ("neural", run["candidate_key"], run["seed"]))
    test_return = [0.001]

    def score(episodes, config, policy, output):
        is_tune = output.name.startswith("tune-")
        if isinstance(policy, tuple):
            net = {"ppo-64": 0.001, "ppo-256": 0.003, "recurrent-256": 0.002}[policy[1]] if is_tune else test_return[0]
            if is_tune:
                net += (44 - policy[2]) * 0.00001
        else:
            net = 0 if policy.kind == "cash" else 0.0001
        metrics = {
            "valid": True,
            "mean_daily_return": net,
            "max_episode_drawdown": 0.001,
            "daily_returns": {f"day-{i}": net for i in range(5)},
        }
        output.write_text(json.dumps(metrics))
        return metrics

    monkeypatch.setattr(study, "_score", score)
    return reg_path, budget_path, events, worker, test_return


def test_freeze_before_test_and_selection_independent_of_test(tmp_path, registered):
    reg, budget, events, _, test_return = registered
    first = study.run_study(tmp_path, reg, budget, tmp_path / "first")
    a = json.loads(first.read_text())
    assert a["selection"]["candidate"] == "ppo-256"
    assert a["selection"]["seed"] == 42
    assert events.index("tune") > max(i for i, event in enumerate(events[: events.index("tune")]) if event == "worker")
    events.clear()
    test_return[0] = -0.005
    b = json.loads(study.run_study(tmp_path, reg, budget, tmp_path / "second").read_text())
    assert a["selection"] == b["selection"]
    assert b["status"] == "inconclusive_not_promoted"
    assert b["constraints"] == {"new_jev_calls": 0, "broker_orders": 0}


def test_incomplete_training_never_releases_tune(tmp_path, monkeypatch, registered):
    reg, budget, events, worker, _ = registered

    def stopped(*args):
        result = worker(*args)
        result["status"] = "stopped"
        return result

    monkeypatch.setattr(study, "_run_worker", stopped)
    with pytest.raises(ValueError, match="complete"):
        study.run_study(tmp_path, reg, budget, tmp_path / "incomplete")
    assert "tune" not in events and "test" not in events


@pytest.mark.parametrize("field", ["prepared_data_id", "budget_id", "source_provenance"])
def test_training_identity_mismatch_never_releases_tune(tmp_path, monkeypatch, registered, field):
    reg, budget, events, worker, _ = registered

    def mismatched(*args):
        result = worker(*args)
        result[field] = "other"
        return result

    monkeypatch.setattr(study, "_run_worker", mismatched)
    with pytest.raises(ValueError, match="identity"):
        study.run_study(tmp_path, reg, budget, tmp_path / "mismatched")
    assert "tune" not in events and "test" not in events


def test_checkpoint_tamper_never_releases_tune(tmp_path, monkeypatch, registered):
    reg, budget, events, worker, _ = registered

    def tampered(*args):
        result = worker(*args)
        (args[5] / "model.zip").write_bytes(b"changed")
        return result

    monkeypatch.setattr(study, "_run_worker", tampered)
    with pytest.raises(ValueError, match="checkpoint"):
        study.run_study(tmp_path, reg, budget, tmp_path / "tampered")
    assert "tune" not in events


def test_report_bundle_integrity(tmp_path, registered):
    reg, budget, _, _, _ = registered
    path = study.run_study(tmp_path, reg, budget, tmp_path / "bundle")
    report = study.load_report(path)
    assert report["selection"]["candidate"] == "ppo-256"
    (path.parent / "tune-cash.json").write_text("tampered")
    with pytest.raises(ValueError, match="artifact"):
        study.load_report(path)
