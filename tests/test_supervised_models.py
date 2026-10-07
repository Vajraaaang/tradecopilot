"""Synthetic model fixtures verify mechanics, never market forecasting quality."""

from __future__ import annotations

import copy
import importlib.util
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tradecopilot.forecast.contracts import FEATURE_NAMES, LABELS, ForecastConfig, ForecastExample, content_hash
from tradecopilot.forecast.ohlcv_features import OHLCV_FEATURE_NAMES, OhlcvFeatureRecord


def registration():
    config = ForecastConfig(symbols=("AAPL", "AMZN", "MSFT", "NVDA", "TSLA"), max_outcome_delay_seconds=0)
    value = {
        "schema_version": "supervised-forecast-registration-v1",
        "forecast_config": config.model_dump(mode="json"),
        "cpu_candidates": [
            {"key": "prior", "kind": "training_class_prior"},
            {"key": "lr-001", "kind": "ohlcv_logistic", "C": 0.01},
            {"key": "lr-1", "kind": "ohlcv_logistic", "C": 1.0},
            {"key": "hgb-31", "kind": "ohlcv_hist_gradient_boosting", "learning_rate": 0.05,
             "max_iter": 100, "max_leaf_nodes": 31, "max_depth": 6, "min_samples_leaf": 100,
             "l2_regularization": 10.0, "early_stopping": False, "class_weight": None, "seed": 42},
        ],
        "neural_candidates": [
            {"key": "tcn-32", "kind": "tcn", "input_channels": 6, "width": 32, "kernel_size": 3,
             "dilations": [1, 2, 4, 8, 16], "convolutions_per_block": 1, "input_projection": True,
             "causal_left_padding": True, "static_dim": 115, "head_hidden": 32, "dropout": 0.1,
             "readout": "last_timestep"},
            {"key": "lstm-64", "kind": "lstm", "input_channels": 6, "hidden_size": 64,
             "n_lstm_layers": 1, "bidirectional": False, "batch_first": True, "static_dim": 115,
             "head_hidden": 32, "dropout": 0.1, "lstm_dropout": 0.0, "readout": "last_hidden",
             "state_reset": "eachindependentwindow"},
        ],
        "neural_training": {
            "seeds": [42, 43, 44], "device": "cpu", "intraop_threads": 1, "batch_size": 256,
            "max_epochs": 20, "max_seconds_per_seed": 600, "optimizer": "AdamW",
            "learning_rate": 0.001, "weight_decay": 0.0001, "gradient_clip_norm": 1.0,
            "loss": "unweighted_cross_entropy",
            "shuffle": "TRAINonly; separatefixedgeneratorperseed soarchitecturesseeidenticalbatchorders",
            "drop_last": False, "num_workers": 0,
            "checkpoint_rule": "lowestcompleteTUNEmeanNLL; improvement>1e-4,patience4,earliesttie; noTRAIN+TUNErefit",
            "incomplete_seed": "familyineligible; nopartialensemble",
            "maximum_optimizer_updates": "20*ceil(N_train/256)", "class_order": list(LABELS),
        },
    }
    value["registration_id"] = content_hash(value)
    return value


def fixture_stage(role="TRAIN", count=36, offset=0):
    reg = registration()
    config = ForecastConfig.model_validate(reg["forecast_config"])
    rng = np.random.default_rng(123 + offset)
    examples, features = [], []
    for index in range(count):
        at = datetime(2025, 1, 3 if role == "TRAIN" else 6, 16, tzinfo=UTC) + timedelta(minutes=index)
        label = LABELS[index % 3]
        row = ForecastExample(
            config_id=config.config_id, symbol=config.symbols[index % 5], as_of=at,
            target_time=at + timedelta(minutes=15), session_date=at.date(), anchor_price="100",
            features=dict.fromkeys(FEATURE_NAMES, 0.0), observation_ids=(f"synthetic-model-{offset + index}",),
            provenance="historical", label=label, target_price=str(99 + index % 3),
            target_return_bps=(index % 3 - 1) * 100, label_observed_at=at + timedelta(minutes=15),
        )
        values = dict.fromkeys(OHLCV_FEATURE_NAMES, None if index % 3 == 2 else float(index % 3))
        examples.append(row)
        features.append(OhlcvFeatureRecord(
            base_example_id=row.example_id, config_id=config.config_id, symbol=row.symbol, as_of=at,
            values=values, input_bars_hash=content_hash(["synthetic-model", offset + index]),
        ))
    stage = SimpleNamespace(
        role=role, sequence=rng.normal(size=(count, 60, 6)).astype(np.float32),
        static=rng.normal(size=(count, 115)).astype(np.float32), examples=tuple(examples), features=tuple(features),
        targets=np.arange(count, dtype=np.int64) % 3, case_ids=tuple(row.example_id for row in examples),
    )
    normalizers = {
        "sequence": {"mean": [0.0] * 6, "scale": [1.0] * 6},
        "static": {"mean": [0.0] * 55, "scale": [1.0] * 55,
                   "binary_indices": [i for i, key in enumerate(OHLCV_FEATURE_NAMES)
                                      if key.startswith("missing_window_")]},
        "fit_role": "TRAIN", "train_case_ids_hash": content_hash(stage.case_ids),
        "fit_case_count": count, "fit_time_positions_count": count * 60,
        "train_raw_inputs_hash": content_hash(["synthetic-model-input", count]),
    }
    manifest = {
        "prepared_data_id": content_hash(["synthetic-prepared", count]), "registration_id": reg["registration_id"],
        "registration": reg, "forecast_config": reg["forecast_config"], "config_id": config.config_id,
        "normalizers": normalizers, "normalizer_id": content_hash(normalizers),
    }
    return stage, manifest, reg


@pytest.mark.parametrize("candidate", ["prior", "lr-001", "lr-1", "hgb-31"])
def test_cpu_safe_roundtrip_matches_native_missing_branch_fixture(candidate, tmp_path):
    from tradecopilot.forecast.supervised_models import (
        fit_cpu,
        load_cpu_artifact,
        native_cpu_probabilities,
        predict_cpu,
        save_cpu_artifact,
    )

    stage, manifest, reg = fixture_stage(count=900 if candidate == "hgb-31" else 36)
    artifact, native = fit_cpu(candidate, stage, manifest, reg)
    path = save_cpu_artifact(artifact, tmp_path / "cpu.json")
    restored = load_cpu_artifact(path)
    got = predict_cpu(restored, stage.features)
    expected = native_cpu_probabilities(artifact, native, stage.features)
    np.testing.assert_allclose(got, expected, atol=1e-10, rtol=0)
    np.testing.assert_allclose(got.sum(axis=1), 1, atol=1e-12)
    assert got.dtype == np.float64 and got.shape == (len(stage.examples), 3)
    assert artifact["training_ids"] == list(stage.case_ids)
    assert "NaN" not in path.read_text() and "Infinity" not in path.read_text()
    if candidate == "hgb-31":
        model = artifact["model_artifact"]
        assert model["parameters"]["fit_settings"]["max_leaf_nodes"] == 31
        assert model["parameters"]["fit_settings"]["max_depth"] == 6
        assert any(node["missing_only"] for tree in model["parameters"]["trees"] for node in tree)
    if candidate.startswith("lr"):
        assert artifact["model_artifact"]["parameters"]["fit_settings"]["class_weight"] is None


def test_cpu_role_alignment_and_future_outcomes_do_not_enter_prediction():
    from tradecopilot.forecast.supervised_models import fit_cpu, predict_cpu

    stage, manifest, reg = fixture_stage()
    artifact, _ = fit_cpu("prior", stage, manifest, reg)
    before = predict_cpu(artifact, stage.features)
    stage.examples = tuple(row.model_copy(update={"label": "UP", "target_return_bps": 999}) for row in stage.examples)
    np.testing.assert_array_equal(before, predict_cpu(artifact, stage.features))
    stage.role = "TUNE"
    with pytest.raises(ValueError, match="TRAIN"):
        fit_cpu("prior", stage, manifest, reg)


def test_cpu_accepts_sealed_normalizer_fit_counts_and_rejects_tampering(tmp_path):
    from tradecopilot.forecast.supervised_models import fit_cpu, load_cpu_artifact, save_cpu_artifact

    stage, manifest, reg = fixture_stage()
    manifest["normalizers"].update(
        fit_case_count=len(stage.case_ids), fit_time_positions_count=len(stage.case_ids) * 60,
    )
    manifest["normalizer_id"] = content_hash(manifest["normalizers"])
    artifact, _ = fit_cpu("lr-001", stage, manifest, reg)
    path = save_cpu_artifact(artifact, tmp_path / "cpu.json")
    for mutate in (lambda d: d.update(schema_version="unknown"),
                   lambda d: d.update(model_id="0" * 64),
                   lambda d: d.update(normalizer_hash="0" * 64),
                   lambda d: d["model_artifact"]["parameters"].update(coefficients=[[0]]),
                   lambda d: d["normalizers"].update(fit_case_count=100)):
        changed = copy.deepcopy(artifact)
        mutate(changed)
        if changed["model_id"] != "0" * 64:
            changed["model_id"] = content_hash({k: v for k, v in changed.items() if k != "model_id"})
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            load_cpu_artifact(path)


@pytest.mark.parametrize("candidate,count", [("tcn-32", 20579), ("lstm-64", 24291)])
def test_neural_exact_architecture_and_finite_float64_probabilities(candidate, count):
    from tradecopilot.forecast.supervised_models import make_neural_model, neural_probabilities

    torch = pytest.importorskip("torch")
    model = make_neural_model(candidate, 42)
    assert sum(p.numel() for p in model.parameters()) == count
    model.eval()
    rng = np.random.default_rng(9)
    sequence = rng.normal(size=(7, 60, 6)).astype(np.float32)
    static = rng.normal(size=(7, 115)).astype(np.float32)
    before = neural_probabilities(model, sequence, static)
    neural_probabilities(model, sequence[::-1].copy() * 10, static[::-1].copy())
    after = neural_probabilities(model, sequence, static)
    np.testing.assert_array_equal(before, after)
    np.testing.assert_allclose(before.sum(axis=1), 1, atol=1e-12, rtol=0)
    assert before.dtype == np.float64 and before.shape == (7, 3) and np.isfinite(before).all()
    assert torch.get_num_threads() == 1
    for seq, st in ((sequence[:, :-1], static), (sequence, static[:, :-1]),
                    (sequence.astype(np.float64) * np.nan, static)):
        with pytest.raises(ValueError, match="input"):
            neural_probabilities(model, seq, st)


def test_tcn_is_prefix_causal_and_has_receptive_field_63():
    from tradecopilot.forecast.supervised_models import make_neural_model

    torch = pytest.importorskip("torch")
    model = make_neural_model("tcn-32", 42).eval()
    original = torch.ones(2, 60, 6)
    changed = original.clone()
    changed[:, 23:] *= 91
    with torch.no_grad():
        a = model.encode_sequence(original)
        b = model.encode_sequence(changed)
    torch.testing.assert_close(a[:, :23], b[:, :23], rtol=0, atol=0)
    assert 1 + sum((block.kernel_size[0] - 1) * block.dilation[0] for block in model.blocks) == 63
    assert all(block.padding == (0,) for block in model.blocks)


def test_lstm_windows_are_stateless_and_batch_order_independent():
    from tradecopilot.forecast.supervised_models import make_neural_model, neural_probabilities

    pytest.importorskip("torch")
    model = make_neural_model("lstm-64", 42).eval()
    rng = np.random.default_rng(55)
    seq = rng.normal(size=(9, 60, 6)).astype(np.float32)
    static = rng.normal(size=(9, 115)).astype(np.float32)
    expected = neural_probabilities(model, seq, static)
    perm = np.array([4, 8, 1, 6, 3, 2, 0, 5, 7])
    reordered = neural_probabilities(model, seq[perm], static[perm])[np.argsort(perm)]
    independent = np.concatenate([neural_probabilities(model, seq[i:i + 1], static[i:i + 1]) for i in range(9)])
    np.testing.assert_allclose(expected, reordered, atol=1e-7, rtol=0)
    np.testing.assert_allclose(expected, independent, atol=1e-7, rtol=0)


def install_synthetic_training_fixture(tmp_path, monkeypatch):
    from tradecopilot.forecast import supervised_models as models

    train, manifest, reg = fixture_stage()
    tune, _, _ = fixture_stage("TUNE", count=18, offset=1000)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text(json.dumps(manifest))
    reg_path = tmp_path / "registration.json"
    reg_path.write_text(json.dumps(reg))
    calls = []
    # Other workers can still edit source during fixture-only verification. Pin this
    # fixture's source identity; the dedicated guard test changes the pinned hashes.
    fixture_sources = models.source_hashes()
    monkeypatch.setattr(models, "source_hashes", lambda: fixture_sources.copy())

    def loader(directory, role, *, selection_path=None):
        assert directory == prepared
        assert selection_path is None
        calls.append(role)
        if role not in ("TRAIN", "TUNE"):
            pytest.fail("training attempted to load a later stage")
        return (train if role == "TRAIN" else tune), manifest

    monkeypatch.setattr(models, "_load_stage", loader)
    return prepared, reg_path, train, tune, manifest, calls


@pytest.mark.parametrize("candidate", ["tcn-32", "lstm-64"])
def test_synthetic_optimizer_only_train_tune_and_json_tensor_roundtrip(candidate, tmp_path, monkeypatch):
    from tradecopilot.forecast import supervised_models as models

    pytest.importorskip("torch")
    prepared, reg_path, _, tune, manifest, calls = install_synthetic_training_fixture(tmp_path, monkeypatch)
    native_epoch_predictions = []
    tune_loss = models._tune_loss

    def record_native_checkpoint(model, stage, started):
        native_epoch_predictions.append(models.neural_probabilities(model, stage.sequence, stage.static))
        return tune_loss(model, stage, started)

    monkeypatch.setattr(models, "_tune_loss", record_native_checkpoint)
    result = models.train_neural_seed(prepared, reg_path, candidate, 42, tmp_path / "seed")
    assert result["status"] in {"complete", "early_stopped"} and result["eligible"] is True
    assert calls == ["TRAIN", "TUNE"]
    assert result["optimizer_updates"] <= 20 and result["epochs_completed"] <= 20
    assert result["checkpoint_hash"] and result["prepared_data_id"] == manifest["prepared_data_id"]
    predictor = models.load_neural_artifact(Path(result["model_path"]))
    expected = predictor.probabilities(tune.sequence, tune.static)
    np.testing.assert_array_equal(expected, native_epoch_predictions[result["best_epoch"] - 1])
    fresh = models.load_neural_artifact(Path(result["model_path"]))
    np.testing.assert_array_equal(expected, fresh.probabilities(tune.sequence, tune.static))
    assert predictor.model_id == result["checkpoint_hash"]
    assert json.loads((tmp_path / "seed" / "result.json").read_text()) == result
    assert predictor.artifact["checkpoint"]["epoch"] == result["best_epoch"]
    assert result["result_id"] == content_hash({k: v for k, v in result.items() if k != "result_id"})
    assert predictor.artifact["parameter_count"] == result["parameter_count"] == (
        20579 if candidate == "tcn-32" else 24291
    )
    assert predictor.artifact["input_dimensions"] == {"sequence": [60, 6], "static": 115}


def test_architectures_share_shuffle_order_from_separate_seed_generator(tmp_path, monkeypatch):
    from tradecopilot.forecast import supervised_models as models

    pytest.importorskip("torch")
    prepared, reg_path, train, *_ = install_synthetic_training_fixture(tmp_path, monkeypatch)
    train.sequence[:, 0, 0] = np.arange(len(train.case_ids), dtype=np.float32)
    orders = {"tcn-32": [], "lstm-64": []}
    maker = models.make_neural_model

    def record_inputs(key, seed):
        model = maker(key, seed)

        def record(module, inputs):
            if module.training:
                orders[key].append(inputs[0][:, 0, 0].tolist())

        model.register_forward_pre_hook(record)
        return model

    monkeypatch.setattr(models, "make_neural_model", record_inputs)
    for key in orders:
        result = models.train_neural_seed(prepared, reg_path, key, 43, tmp_path / key)
        assert result["eligible"]
    count = min(len(values) for values in orders.values())
    assert count >= 1
    assert orders["tcn-32"][:count] == orders["lstm-64"][:count]
    assert orders["tcn-32"][0] != list(range(len(train.case_ids)))


def test_neural_artifact_rejects_shapes_versions_hashes_and_executable_names(tmp_path, monkeypatch):
    from tradecopilot.forecast.supervised_models import load_neural_artifact, train_neural_seed

    pytest.importorskip("torch")
    prepared, reg_path, *_ = install_synthetic_training_fixture(tmp_path, monkeypatch)
    result = train_neural_seed(prepared, reg_path, "tcn-32", 42, tmp_path / "seed")
    base = json.loads(Path(result["model_path"]).read_text())
    for field, value in (("schema_version", "executable-v1"), ("model_id", "0" * 64),
                         ("class_order", ["UP", "FLAT", "DOWN"]), ("prepared_data_id", "wrong"),
                         ("normalizer_hash", "0" * 64)):
        changed = copy.deepcopy(base)
        changed[field] = value
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            load_neural_artifact(path)
    for mutate in (lambda d: d["tensors"].update({"python.callback": [1]}),
                   lambda d: d["tensors"].update({"projection.bias": [0]}),
                   lambda d: d["tensors"]["projection.bias"].__setitem__(0, float("nan"))):
        changed = copy.deepcopy(base)
        mutate(changed)
        changed["model_id"] = (
            content_hash({k: v for k, v in changed.items() if k != "model_id"})
            if "NaN" not in json.dumps(changed) else "0" * 64
        )
        path = tmp_path / "bad-tensors.json"
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            load_neural_artifact(path)


def test_time_budget_preserves_only_complete_tune_checkpoint_and_marks_ineligible(tmp_path, monkeypatch):
    from tradecopilot.forecast import supervised_models as models

    pytest.importorskip("torch")
    prepared, reg_path, *_ = install_synthetic_training_fixture(tmp_path, monkeypatch)
    origin = models.time.monotonic()
    expired = False
    tune_calls = 0
    tune_loss = models._tune_loss

    def synthetic_clock():
        return origin + (601 if expired else 0.1)

    def expire_during_second_tune(model, tune, started):
        nonlocal expired, tune_calls
        tune_calls += 1
        expired = tune_calls == 2
        return tune_loss(model, tune, started)

    monkeypatch.setattr(models.time, "monotonic", synthetic_clock)
    monkeypatch.setattr(models, "_tune_loss", expire_during_second_tune)
    result = models.train_neural_seed(prepared, reg_path, "tcn-32", 42, tmp_path / "capped")
    assert result["status"] == "time_cap" and result["eligible"] is False
    assert result["optimizer_updates"] == 2 and result["epochs_completed"] == 1
    assert result["model_path"] is not None
    artifact = models.load_neural_artifact(Path(result["model_path"])).artifact
    assert artifact["checkpoint"]["epoch"] == result["best_epoch"] == 1
    assert len(result["checkpoints"]) == 1


def test_patience_requires_four_complete_tune_evaluations_and_keeps_earliest_tie(tmp_path, monkeypatch):
    from tradecopilot.forecast import supervised_models as models

    pytest.importorskip("torch")
    prepared, reg_path, *_ = install_synthetic_training_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(models, "_tune_loss", lambda model, tune, started: 0.75)
    result = models.train_neural_seed(prepared, reg_path, "lstm-64", 44, tmp_path / "tied")
    assert result["status"] == "early_stopped" and result["eligible"]
    assert result["epochs_completed"] == result["optimizer_updates"] == 5
    assert result["best_epoch"] == 1 and len(result["checkpoints"]) == 1


def test_budget_before_first_epoch_retains_failure_without_checkpoint(tmp_path, monkeypatch):
    from tradecopilot.forecast import supervised_models as models

    pytest.importorskip("torch")
    prepared, reg_path, *_ = install_synthetic_training_fixture(tmp_path, monkeypatch)
    calls = 0

    def expired_clock():
        nonlocal calls
        calls += 1
        return 0.0 if calls == 1 else 601.0

    monkeypatch.setattr(models.time, "monotonic", expired_clock)
    result = models.train_neural_seed(prepared, reg_path, "tcn-32", 42, tmp_path / "never-started")
    assert result["status"] == "time_cap" and not result["eligible"]
    assert result["optimizer_updates"] == result["epochs_completed"] == 0
    assert result["model_path"] is None and result["checkpoints"] == []
    assert json.loads((tmp_path / "never-started" / "result.json").read_text()) == result


def test_source_change_guard_invalidates_seed_and_rejects_changed_export(tmp_path, monkeypatch):
    from tradecopilot.forecast import supervised_models as models

    pytest.importorskip("torch")
    prepared, reg_path, *_ = install_synthetic_training_fixture(tmp_path, monkeypatch)
    original = models.source_hashes
    calls = 0

    def changed_sources():
        nonlocal calls
        calls += 1
        return original() | {"fixture.py": content_hash("before" if calls == 1 else "after")}

    monkeypatch.setattr(models, "source_hashes", changed_sources)
    result = models.train_neural_seed(prepared, reg_path, "lstm-64", 42, tmp_path / "changed")
    assert result["status"] == "source_changed" and result["eligible"] is False
    assert result["source_hashes_before"] != result["source_hashes_after"]
    if result["model_path"] is not None:
        with pytest.raises(ValueError, match="source"):
            models.load_neural_artifact(Path(result["model_path"]))


def test_seed_script_uses_registered_arguments_without_provider_calls():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_supervised_seed.py"
    assert path.is_file()
    spec = importlib.util.spec_from_file_location("supervised_seed_script", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
