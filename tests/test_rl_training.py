import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")
from tradecopilot.forecast.contracts import content_hash
from tradecopilot.rl import training
from tradecopilot.rl.contracts import EpisodeData, SimConfig

CANDIDATES = [
    {"key": "ppo-64", "algorithm": "PPO", "pi": [64, 64], "vf": [64, 64]},
    {"key": "ppo-256", "algorithm": "PPO", "pi": [256, 256], "vf": [256, 256]},
    {
        "key": "recurrent-256",
        "algorithm": "RecurrentPPO",
        "pi": [256, 128],
        "vf": [256, 128],
        "lstm_hidden_size": 256,
        "n_lstm_layers": 1,
        "shared_lstm": False,
        "enable_critic_lstm": True,
    },
]


def episode():
    starts = np.arange(80, dtype=np.int64) * 60
    prices = np.full(80, 100.0)
    return EpisodeData(
        "SYNTHETIC",
        date(2026, 1, 2),
        0,
        4800,
        starts,
        starts + 60,
        starts + 60,
        prices,
        prices,
        np.full(80, 10000.0),
        np.zeros((80, 2), np.float32),
        ("x", "y"),
        "synthetic",
    )


@pytest.mark.parametrize("candidate", CANDIDATES)
def test_exact_architecture_predict_and_saved_parity(candidate, tmp_path):
    model = training.make_model(candidate, [episode()], SimConfig(), 42)
    policy = model.policy
    assert policy.net_arch == {"pi": candidate["pi"], "vf": candidate["vf"]}
    if candidate["algorithm"] == "RecurrentPPO":
        assert policy.lstm_actor.hidden_size == policy.lstm_critic.hidden_size == 256
        assert policy.lstm_actor.num_layers == policy.lstm_critic.num_layers == 1
        assert not policy.shared_lstm
    model.learn(total_timesteps=128)
    assert model.num_timesteps == 128 and model._n_updates == 10
    obs = model.get_env().reset()
    action, state = model.predict(obs, deterministic=True)
    assert int(action[0]) in (0, 1, 2)
    model.save(tmp_path / "model")
    restored = type(model).load(tmp_path / "model.zip", device="cpu")
    restored_action, restored_state = restored.predict(obs, deterministic=True)
    np.testing.assert_array_equal(action, restored_action)
    if state is not None:
        for before, after in zip(state, restored_state, strict=True):
            np.testing.assert_array_equal(before, after)


def test_unknown_architecture_and_auto_device_rejected():
    with pytest.raises(ValueError):
        training.make_model(dict(CANDIDATES[0], pi=[32]), [episode()], SimConfig(), 42)
    with pytest.raises(ValueError):
        training.make_model(CANDIDATES[0], [episode()], SimConfig(), 42, "auto")


def test_benchmark_and_time_guard_metadata_only():
    result = training.benchmark_model(CANDIDATES[0], [episode()], SimConfig(), 42, "cpu", steps=128)
    assert result["status"] == "complete" and result["n_updates"] == 10
    assert result["actual_timesteps"] == 128
    assert result["n_parameters"] > 0
    assert not any("reward" in key or "return" in key for key in result)
    model = training.make_model(CANDIDATES[0], [episode()], SimConfig(), 42)
    guard = training.make_budget_callback(128, 30, None)
    model.learn(total_timesteps=128, callback=guard)
    assert model._n_updates == 10  # final full rollout must reach its optimizer update
    stopped = training.benchmark_model(
        CANDIDATES[0], [episode()], SimConfig(), 42, "cpu", steps=1024, max_seconds=0.000001
    )
    assert stopped["status"] == "stopped" and stopped["actual_timesteps"] < 1024


def sealed(path: Path, value: dict, key: str):
    value[key] = content_hash(value)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize(
    "steps,seconds,seed,key",
    [(128, 30, 42, "ppo-64"), (10240, 601, 42, "ppo-64"), (10240, 30, 99, "ppo-64"), (10240, 30, 42, "other")],
)
def test_registry_and_caps_guard_before_data_load(tmp_path, steps, seconds, seed, key):
    reg = {"candidates": CANDIDATES, "seeds": [42], "environment": {}, "optimizer": training.OPTIMIZER}
    sealed(tmp_path / "reg.json", reg, "registration_id")
    budget = {
        "registration_id": reg["registration_id"],
        "prepared_data_id": "data",
        "shared_steps": steps,
        "max_seconds": seconds,
        "device": "cpu",
    }
    sealed(tmp_path / "budget.json", budget, "budget_id")
    with pytest.raises(ValueError):
        training.train_seed(
            tmp_path / "missing", tmp_path / "reg.json", tmp_path / "budget.json", key, seed, tmp_path / "output"
        )
    assert not (tmp_path / "output").exists()


def test_immutable_output_guard(tmp_path):
    with pytest.raises(ValueError, match="immutable"):
        training.train_seed(tmp_path, tmp_path / "missing", tmp_path / "missing", "ppo-64", 42, tmp_path)


def test_final_rollout_deadline_keeps_optimizer_update():
    model = training.make_model(CANDIDATES[0], [episode()], SimConfig(), 42)
    callback = training.make_budget_callback(128, 30, None)
    callback.init_callback(model)
    callback.num_timesteps = 128
    # A deadline that expires exactly at a complete rollout must retain the transition.
    monkeyclock = pytest.MonkeyPatch()
    monkeyclock.setattr(training.time, "monotonic", lambda: float("inf"))
    try:
        assert callback._on_step()
        callback.num_timesteps = 129
        assert not callback._on_step()
    finally:
        monkeyclock.undo()


def test_seal_tampering_rejected(tmp_path):
    value = {"seeds": [42]}
    sealed(tmp_path / "reg.json", value, "registration_id")
    value["seeds"] = [43]
    (tmp_path / "reg.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="registration_id"):
        training.train_seed(tmp_path, tmp_path / "reg.json", tmp_path / "unused", "ppo-64", 42, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_train_seed_binds_train_identity_and_checkpoint(tmp_path, monkeypatch):
    from tradecopilot.rl import data

    env = {
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
    }
    reg = {"candidates": CANDIDATES, "seeds": [42], "environment": env, "optimizer": training.OPTIMIZER}
    sealed(tmp_path / "reg.json", reg, "registration_id")
    budget = {
        "registration_id": reg["registration_id"],
        "prepared_data_id": "data",
        "shared_steps": 10240,
        "max_seconds": 30,
        "device": "cpu",
    }
    sealed(tmp_path / "budget.json", budget, "budget_id")
    roles = []

    def load(directory, role):
        roles.append(role)
        return [episode()], {"registration_id": reg["registration_id"], "data_id": "data"}

    monkeypatch.setattr(data, "load_prepared", load)
    original_make = training.make_model

    def make(candidate, episodes, config, seed, device):
        model = original_make(candidate, episodes, config, seed, device)

        def learn(total_timesteps, callback):
            # Unit-test the bounded orchestration without launching a real 10K training run.
            assert total_timesteps == 10240
            model.num_timesteps = total_timesteps
            model._n_updates = 800
            return model

        monkeypatch.setattr(model, "learn", learn)
        return model

    monkeypatch.setattr(training, "make_model", make)
    result = training.train_seed(
        tmp_path, tmp_path / "reg.json", tmp_path / "budget.json", "ppo-64", 42, tmp_path / "output"
    )
    assert roles == ["train"]
    assert result["status"] == "complete"
    assert result["registration_id"] == reg["registration_id"]
    assert result["budget_id"] == budget["budget_id"]
    assert result["source_provenance"]["rl/training.py"]
    assert len(result["checkpoint_sha256"]) == 64
    assert (tmp_path / "output/model.zip").exists()
    assert json.loads((tmp_path / "output/progress.json").read_text()) == {
        "actual_timesteps": 10240,
        "elapsed_seconds": result["elapsed_seconds"],
        "status": "complete",
    }
