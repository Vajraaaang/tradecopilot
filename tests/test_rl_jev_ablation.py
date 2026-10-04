import copy
import json
from datetime import date
from hashlib import sha256

import numpy as np
import pytest

from tradecopilot.forecast.contracts import content_hash
from tradecopilot.forecast.ohlcv_features import OHLCV_FEATURE_NAMES
from tradecopilot.rl import jev_ablation as ab
from tradecopilot.rl.contracts import EpisodeData
from tradecopilot.rl.training import _CANDIDATES, OPTIMIZER


def sealed(value, key):
    return {**value, key: content_hash(value)}


def write_sealed(path, value, key):
    result = sealed(value, key)
    path.write_text(json.dumps(result))
    return result


@pytest.fixture
def inputs():
    reg = sealed(
        {
            "symbols": ["A", "B", "C", "D", "E"],
            "seeds": [42, 43, 44],
            "profiles": ["CONTEXT_ONLY", "JEV_ASSISTED"],
            "candidates": [_CANDIDATES["recurrent-256"]],
            "optimizer": OPTIMIZER,
            "jev": {
                "model": "jev-1.13.0",
                "prompt_version": "masked-v1",
                "as_of_minutes_after_open": 61,
                "simulated_latency_seconds": 60,
                "horizons": ["15m", "60m", "close"],
            },
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
        },
        "registration_id",
    )
    names = (
        *OHLCV_FEATURE_NAMES,
        *(f"missing_{n}" for n in OHLCV_FEATURE_NAMES),
        *(f"symbol_{s}" for s in ["A", "B", "C", "D", "E"]),
    )
    opening = 1720000020
    starts = np.arange(opening, opening + 390 * 60, 60, dtype=np.int64)
    manifest = sealed(
        {
            "registration_id": reg["registration_id"],
            "source_data_id": "source",
            "feature_names": list(names),
            "episodes": [{"symbol": "A", "session_date": "2024-07-03"}],
            "normalizer": {"feature_names": list(OHLCV_FEATURE_NAMES), "mean": [0.0] * 55, "scale": [1.0] * 55},
        },
        "data_id",
    )
    episode = EpisodeData(
        "A",
        date(2024, 7, 3),
        opening,
        opening + 390 * 60,
        starts,
        starts + 60,
        starts + 60,
        np.ones(390) * 100,
        np.ones(390) * 100,
        np.ones(390) * 10000,
        np.zeros((390, 115), dtype=np.float32),
        names,
        manifest["data_id"],
    )
    as_of = opening + 61 * 60
    records = [
        {
            "symbol": "A",
            "session_date": "2024-07-03",
            "as_of": as_of,
            "target_time": episode.session_close if h == "close" else as_of + int(h[:-1]) * 60,
            "horizon_key": h,
            "anchor_price": "100",
            "replay_available_at": as_of + 60,
            "generated_at": "2026-10-04T00:00:00+00:00",
            "probabilities": {"DOWN": 0.1, "FLAT": 0.2, "UP": 0.7},
            "model_confidence": 0.5,
            "status": "abstained",
            "request_id": "r",
            "payload_id": "p",
            "input_id": "i",
        }
        for h in ["15m", "60m", "close"]
    ]
    cache = sealed(
        {
            "schema_version": "jev-rl-cache-v1",
            "registration_id": reg["registration_id"],
            "source_data_id": "source",
            "base_prepared_data_id": manifest["data_id"],
            "model": "jev-1.13.0",
            "prompt_version": "masked-v1",
            "availability_mode": "hypothetical_as_of_plus_60_seconds",
            "records": records,
            "usage": {},
            "inventory": {},
        },
        "cache_id",
    )
    return episode, manifest, reg, cache


def test_causal_availability_expiry_and_neutral_metadata(inputs):
    ep, manifest, reg, cache = inputs
    jev = ab.augment_episodes([ep], cache, "JEV_ASSISTED", manifest, reg)[0]
    neutral = ab.augment_episodes([ep], cache, "CONTEXT_ONLY", manifest, reg)[0]
    assert jev.features.shape == (390, 139)
    assert len(jev.feature_names) + 12 == 151
    for horizon, target in enumerate([76, 121, 390]):
        block = jev.features[:, 115 + horizon * 8 : 123 + horizon * 8]
        control = neutral.features[:, 115 + horizon * 8 : 123 + horizon * 8]
        assert not block[:61].any()
        assert not block[target:].any()
        np.testing.assert_allclose(block[61:target, :3], np.tile([0.1, 0.2, 0.7], (target - 61, 1)))
        np.testing.assert_allclose(control[61:target, :3], 1 / 3)
        assert not control[:, 3].any()
        np.testing.assert_array_equal(block[:, 4:], control[:, 4:])
        assert block[target - 1, -1] == 1  # Target-time equality remains available.
    assert cache["records"][0]["status"] == "abstained"  # Low confidence is a real forecast.


def test_future_probability_mutation_cannot_change_past_observations(inputs):
    ep, manifest, reg, cache = inputs
    baseline = ab.augment_episodes([ep], cache, "JEV_ASSISTED", manifest, reg)[0]
    changed = copy.deepcopy(cache)
    changed.pop("cache_id")
    changed["records"][0]["probabilities"] = {"DOWN": 0.8, "FLAT": 0.1, "UP": 0.1}
    changed = sealed(changed, "cache_id")
    updated = ab.augment_episodes([ep], changed, "JEV_ASSISTED", manifest, reg)[0]
    np.testing.assert_array_equal(baseline.features[:61], updated.features[:61])
    np.testing.assert_array_equal(baseline.features[76:], updated.features[76:])
    assert not np.array_equal(baseline.features[61:76], updated.features[61:76])


@pytest.mark.parametrize("field", ["cache_id", "registration_id", "source_data_id", "base_prepared_data_id", "model"])
def test_cache_identity_tamper_refused(inputs, field):
    ep, manifest, reg, cache = inputs
    changed = copy.deepcopy(cache)
    changed[field] = "tampered"
    if field != "cache_id":
        changed.pop("cache_id")
        changed = sealed(changed, "cache_id")
    with pytest.raises(ValueError, match=r"cache|identity"):
        ab.augment_episodes([ep], changed, "JEV_ASSISTED", manifest, reg)


def test_normalizer_tamper_refused(inputs):
    ep, manifest, reg, cache = inputs
    manifest["normalizer"]["mean"][0] = 99
    with pytest.raises(ValueError, match=r"manifest|data_id"):
        ab.augment_episodes([ep], cache, "CONTEXT_ONLY", manifest, reg)


@pytest.mark.parametrize("mutation", ["early", "target", "anchor", "probabilities", "duplicate", "missing"])
def test_invalid_cache_timing_coverage_and_values_refused(inputs, mutation):
    ep, manifest, reg, cache = inputs
    changed = copy.deepcopy(cache)
    changed.pop("cache_id")
    first = changed["records"][0]
    if mutation == "early":
        first["replay_available_at"] -= 60
    elif mutation == "target":
        first["target_time"] += 60
    elif mutation == "anchor":
        first["anchor_price"] = "101"
    elif mutation == "probabilities":
        first["probabilities"]["DOWN"] = float("nan")
    elif mutation == "duplicate":
        changed["records"].append(copy.deepcopy(first))
    else:
        changed["records"].pop()
    if mutation == "probabilities":
        with pytest.raises(ValueError):
            sealed(changed, "cache_id")
        return
    changed = sealed(changed, "cache_id")
    with pytest.raises(ValueError):
        ab.augment_episodes([ep], changed, "JEV_ASSISTED", manifest, reg)


@pytest.fixture
def runner(tmp_path, monkeypatch, inputs):
    ep, manifest, reg, cache = inputs
    reg_path, cache_path, budget_path = [tmp_path / n for n in ("registration.json", "cache.json", "budget.json")]
    reg_path.write_text(json.dumps(reg))
    cache_path.write_text(json.dumps(cache))
    allocation = write_sealed(
        budget_path,
        {
            "registration_id": reg["registration_id"],
            "base_prepared_data_id": manifest["data_id"],
            "cache_id": cache["cache_id"],
            "shared_steps": 10240,
            "max_seconds": 600,
            "device": "cpu",
        },
        "budget_id",
    )
    events = []
    monkeypatch.setattr(ab.training, "_source_provenance", lambda: {"fixed": "source"})
    monkeypatch.setattr(ab, "_lock_hash", lambda: "lock")
    monkeypatch.setattr(ab, "_execution_provenance", lambda: {"script": "fixed"})
    monkeypatch.setattr(ab, "_load_cache", lambda path: json.loads(path.read_text()))

    def load(directory, role, selection_path=None):
        events.append(role)
        if role == "test":
            selection = ab.training._read_sealed(selection_path, "selection_id")
            assert set(selection["profiles"]) == set(ab.PROFILES)
            assert selection["stage"] == "selection_frozen"
            assert events.count("worker") == 6
        return [ep], manifest

    monkeypatch.setattr(ab, "load_prepared", load)

    def worker(prepared, cache_path, registration, budget_path, profile, seed, output, budget, progress):
        events.append("worker")
        assert "tune" not in events
        output.mkdir()
        model = output / "model.zip"
        model.write_bytes(f"{profile}-{seed}".encode())
        result = {
            "profile": profile,
            "seed": seed,
            "candidate": _CANDIDATES["recurrent-256"],
            "status": "complete",
            "actual_timesteps": budget["shared_steps"],
            "n_updates": 800,
            "elapsed_seconds": 1.0,
            "registration_id": reg["registration_id"],
            "base_prepared_data_id": manifest["data_id"],
            "cache_id": cache["cache_id"],
            "budget_id": allocation["budget_id"],
            "config_hash": ab.training.registered_config(reg).content_hash,
            "source_provenance": {"fixed": "source"},
            "lock_sha256": "lock",
            "execution_provenance": {"script": "fixed"},
            "versions": {"torch": "fake"},
            "n_parameters": 123,
            "observation_dim": 151,
            "checkpoint_sha256": sha256(model.read_bytes()).hexdigest(),
        }
        return write_sealed(output / "result.json", result, "training_id")

    monkeypatch.setattr(ab, "_run_worker", worker)
    monkeypatch.setattr(ab, "_neural", lambda directory, run, device: (run["profile"], run["seed"]))
    test_return = [0.001]

    def score(episodes, config, policy, output):
        events.append("score")
        if isinstance(policy, tuple):
            net = 0.001 + (44 - policy[1]) * 0.00001 if output.name.startswith("tune-") else test_return[0]
            net += 0.0002 if policy[0] == "JEV_ASSISTED" else 0
        else:
            net = 0 if policy.kind == "cash" else 0.0001
        metrics = {
            "valid": True,
            "mean_daily_return": net,
            "max_episode_drawdown": 0.001,
            "daily_returns": {f"day-{i}": net for i in range(10)},
        }
        output.write_text(json.dumps(metrics))
        return metrics

    monkeypatch.setattr(ab, "_score", score)
    monkeypatch.setattr(ab, "grade_forecasts", lambda *args: {"evidence_mode": "mock"})
    return reg_path, cache_path, budget_path, events, worker, test_return


def test_both_selections_before_test_and_no_test_reselection(tmp_path, runner):
    reg, cache, budget, events, _, test_return = runner
    path = ab.run_ablation(tmp_path, cache, reg, budget, tmp_path / "first")
    first = ab.load_report(path)
    assert {p: r["seed"] for p, r in first["selection"]["profiles"].items()} == dict.fromkeys(ab.PROFILES, 42)
    assert events.index("tune") > max(i for i, v in enumerate(events) if v == "worker")
    assert set(first["test_seeds"]) == set(ab.PROFILES)
    assert all(set(v) == {"42", "43", "44"} for v in first["test_seeds"].values())
    assert all(set(s["profiles"]) == set(ab.PROFILES) for s in first["cost_stress"].values())
    events.clear()
    test_return[0] = -0.1
    second = ab.load_report(ab.run_ablation(tmp_path, cache, reg, budget, tmp_path / "second"))
    assert first["selection"] == second["selection"]
    assert second["status"] == "inconclusive_not_promoted"
    artifact = path.parent / "tune-CONTEXT_ONLY-42.json"
    artifact.write_text("tampered")
    with pytest.raises(ValueError, match="artifact"):
        ab.load_report(path)


@pytest.mark.parametrize(
    "field", ["status", "actual_timesteps", "n_updates", "source_provenance", "cache_id", "n_parameters"]
)
def test_unequal_or_incomplete_training_does_not_release_tune(tmp_path, runner, monkeypatch, field):
    reg, cache, budget, events, worker, _ = runner

    def changed(*args):
        result = worker(*args)
        if result["profile"] == "JEV_ASSISTED" and result["seed"] == 44:
            result[field] = "stopped" if field == "status" else 1
            result.pop("training_id")
            result = sealed(result, "training_id")
        return result

    monkeypatch.setattr(ab, "_run_worker", changed)
    with pytest.raises(ValueError):
        ab.run_ablation(tmp_path, cache, reg, budget, tmp_path / "bad")
    assert "tune" not in events and "test" not in events and "score" not in events
    assert (tmp_path / "bad" / "report.json").exists()


def test_checkpoint_tamper_blocks_tune(tmp_path, runner, monkeypatch):
    reg, cache, budget, events, worker, _ = runner

    def changed(*args):
        result = worker(*args)
        (args[6] / "model.zip").write_bytes(b"changed")
        return result

    monkeypatch.setattr(ab, "_run_worker", changed)
    with pytest.raises(ValueError, match="checkpoint"):
        ab.run_ablation(tmp_path, cache, reg, budget, tmp_path / "tampered")
    assert "tune" not in events


def test_mean_seed_comparison_uses_all_seeds():
    values = {
        str(seed): {"valid": True, "daily_returns": {"d": float(seed - 42)}, "max_episode_drawdown": 0.001}
        for seed in (42, 43, 44)
    }
    result = ab.mean_seed_metrics(values)
    assert result["daily_returns"] == {"d": 1.0}


def test_budget_has_strict_registered_cpu_cap(inputs):
    _, manifest, reg, cache = inputs
    for field, value in [
        ("shared_steps", True),
        ("shared_steps", 92288),
        ("shared_steps", 10241),
        ("max_seconds", 601),
        ("device", "mps"),
        ("cache_id", "other"),
    ]:
        budget = {
            "registration_id": reg["registration_id"],
            "base_prepared_data_id": manifest["data_id"],
            "cache_id": cache["cache_id"],
            "shared_steps": 10240,
            "max_seconds": 600,
            "device": "cpu",
        }
        budget[field] = value
        with pytest.raises(ValueError):
            ab.validate_budget(budget, reg, manifest, cache)


def test_forecast_grading_exact_targets_train_prior_and_abstention(inputs):
    ep, _, _, cache = inputs
    report = ab.grade_forecasts([ep], [ep], [ep], cache)
    horizon = report["horizons"]["15m"]
    assert horizon["train_prior"] == [0.0, 1.0, 0.0]
    assert horizon["test"]["jev"]["accuracy"] == 0
    assert horizon["test"]["jev"]["coverage_0_6"] == 0
    assert horizon["test"]["prior"]["accuracy"] == 1


def test_resealed_wrong_market_feature_order_refused(inputs):
    from dataclasses import replace

    ep, manifest, reg, cache = inputs
    changed = copy.deepcopy(manifest)
    changed.pop("data_id")
    changed["feature_names"][0], changed["feature_names"][1] = (
        changed["feature_names"][1],
        changed["feature_names"][0],
    )
    changed = sealed(changed, "data_id")
    cache = copy.deepcopy(cache)
    cache.pop("cache_id")
    cache["base_prepared_data_id"] = changed["data_id"]
    cache = sealed(cache, "cache_id")
    ep = replace(ep, feature_names=tuple(changed["feature_names"]), data_id=changed["data_id"], episode_id="")
    with pytest.raises(ValueError, match="feature order"):
        ab.augment_episodes([ep], cache, "JEV_ASSISTED", changed, reg)


def test_generation_timezone_and_boundary_distributions_refused(inputs):
    ep, manifest, reg, cache = inputs
    changed = copy.deepcopy(cache)
    changed.pop("cache_id")
    changed["records"][0]["generated_at"] = "2026-10-04T00:00:00"
    with pytest.raises(ValueError, match="aware"):
        ab.augment_episodes([ep], sealed(changed, "cache_id"), "JEV_ASSISTED", manifest, reg)
    changed["records"][0]["generated_at"] += "+00:00"
    changed["records"][0]["probabilities"]["DOWN"] = -0.1
    with pytest.raises(ValueError, match="probabilities"):
        ab.augment_episodes([ep], sealed(changed, "cache_id"), "JEV_ASSISTED", manifest, reg)


def test_delayed_market_bar_masks_forecast_too(inputs):
    from dataclasses import replace

    ep, manifest, reg, cache = inputs
    delayed = ep.available_at.copy()
    delayed[65] += 60
    ep = replace(ep, available_at=delayed, episode_id="")
    augmented = ab.augment_episodes([ep], cache, "JEV_ASSISTED", manifest, reg)[0]
    assert not augmented.features[65, 115:].any()


def test_rerun_existing_output_never_loads_inputs(tmp_path):
    with pytest.raises(ValueError, match="immutable"):
        ab.run_ablation(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path)


def test_exact_forecast_target_required(inputs):
    from dataclasses import replace

    ep, _, _, cache = inputs
    target_row = 75
    keep = np.arange(len(ep.starts)) != target_row
    reduced = replace(
        ep,
        **{
            name: getattr(ep, name)[keep]
            for name in ("starts", "ends", "available_at", "opens", "closes", "volumes", "features")
        },
        episode_id="",
    )
    with pytest.raises(ValueError, match="exact minute-close"):
        ab.grade_forecasts([ep], [ep], [reduced], cache)


@pytest.mark.parametrize("elapsed,status", [(1.0, "complete"), (600.1, "deadline_overrun")])
def test_train_profile_only_loads_train_and_uses_151_observations(tmp_path, runner, monkeypatch, elapsed, status):
    from types import SimpleNamespace

    reg, cache, budget, events, _, _ = runner
    observed = {}

    class Model:
        def learn(self, total_timesteps, callback):
            assert events == ["train"]
            observed["steps"] = total_timesteps

        def save(self, path):
            path.write_bytes(b"fake checkpoint")

        def get_env(self):
            return self

        def close(self):
            observed["closed"] = True

    def make(candidate, episodes, config, seed, device):
        observed["features"] = episodes[0].features.shape[1]
        observed["device"] = device
        observed["config"] = config.content_hash
        return Model()

    monkeypatch.setattr(ab.training, "make_model", make)
    monkeypatch.setattr(ab.training, "make_budget_callback", lambda *args: None)
    monkeypatch.setattr(
        ab.training,
        "_metrics",
        lambda model, elapsed, steps: {
            "status": "complete",
            "actual_timesteps": steps,
            "n_updates": 800,
            "n_parameters": 123,
            "elapsed_seconds": elapsed,
        },
    )
    measured_clock = iter([0.0, elapsed])
    monkeypatch.setattr(ab.time, "monotonic", lambda: next(measured_clock))
    version = SimpleNamespace(__version__="fake")
    monkeypatch.setattr(ab.training, "_extras", lambda: (version, version, version))
    result = ab.train_profile(tmp_path, cache, reg, budget, "JEV_ASSISTED", 42, tmp_path / "worker")
    assert result["observation_dim"] == 151
    assert result["status"] == status
    assert result["elapsed_seconds"] == elapsed
    assert result["actual_timesteps"] == 10240 and result["n_updates"] == 800
    assert (tmp_path / "worker" / "model.zip").exists()
    assert observed == {
        "features": 139,
        "device": "cpu",
        "config": result["config_hash"],
        "steps": 10240,
        "closed": True,
    }
    assert events == ["train", "train"]
    assert result["training_id"] == content_hash({k: v for k, v in result.items() if k != "training_id"})


def test_uniform_missing_capacity_blocks_tune(tmp_path, runner, monkeypatch):
    reg, cache, budget, events, worker, _ = runner

    def missing(*args):
        result = worker(*args)
        result.pop("training_id")
        result.pop("n_parameters")
        return sealed(result, "training_id")

    monkeypatch.setattr(ab, "_run_worker", missing)
    with pytest.raises(ValueError, match="capacity"):
        ab.run_ablation(tmp_path, cache, reg, budget, tmp_path / "missing-capacity")
    assert "tune" not in events


def test_forecast_selection_requires_both_distribution_and_model_confidence():
    metrics = ab._forecast_metrics([[0.34, 0.33, 0.33], [0.7, 0.2, 0.1]], [0, 0], [0.9, 0.5])
    assert metrics["coverage_0_6"] == 0
    assert metrics["selected_accuracy_0_6"] is None


@pytest.mark.parametrize("total", [0.9999995, 1.0000005])
def test_vendor_accepted_probability_rounding_is_accepted_by_rl(inputs, total):
    ep, manifest, reg, cache = inputs
    changed = copy.deepcopy(cache)
    changed.pop("cache_id")
    changed["records"][0]["probabilities"] = {"DOWN": 0.1, "FLAT": 0.2, "UP": total - 0.3}
    augmented = ab.augment_episodes([ep], sealed(changed, "cache_id"), "JEV_ASSISTED", manifest, reg)[0]
    np.testing.assert_allclose(augmented.features[61, 115:118], [0.1, 0.2, total - 0.3])


@pytest.mark.parametrize("total", [0.999998, 1.000002])
def test_probability_sum_beyond_vendor_tolerance_is_refused(inputs, total):
    ep, manifest, reg, cache = inputs
    changed = copy.deepcopy(cache)
    changed.pop("cache_id")
    changed["records"][0]["probabilities"] = {"DOWN": 0.1, "FLAT": 0.2, "UP": total - 0.3}
    with pytest.raises(ValueError, match="probabilities"):
        ab.augment_episodes([ep], sealed(changed, "cache_id"), "JEV_ASSISTED", manifest, reg)


@pytest.mark.parametrize("elapsed", [600.1, 0.0, -1.0, None, True])
def test_elapsed_deadline_or_invalid_evidence_blocks_tune(tmp_path, runner, monkeypatch, elapsed):
    reg, cache, budget, events, worker, _ = runner

    def overrun(*args):
        result = worker(*args)
        result.pop("training_id")
        result["elapsed_seconds"] = elapsed
        assert result["actual_timesteps"] == 10240 and result["n_updates"] == 800
        return sealed(result, "training_id")

    monkeypatch.setattr(ab, "_run_worker", overrun)
    with pytest.raises(ValueError, match="deadline"):
        ab.run_ablation(tmp_path, cache, reg, budget, tmp_path / "deadline")
    assert "tune" not in events and "test" not in events and "score" not in events
    report = ab.load_report(tmp_path / "deadline" / "report.json")
    assert report["status"] == "failed"
    assert all(r["actual_timesteps"] == 10240 and r["n_updates"] == 800 for r in report["training"])
    assert all(
        (tmp_path / "deadline" / f"{r['profile']}-{r['seed']}" / "model.zip").exists() for r in report["training"]
    )
