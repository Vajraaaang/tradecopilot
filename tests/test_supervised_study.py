"""Synthetic control-flow tests; these never evaluate market prediction quality."""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from tradecopilot.forecast.contracts import FEATURE_NAMES, LABELS, ForecastConfig, ForecastExample


def examples(days=20, per_day=9):
    config = ForecastConfig(symbols=("AAPL",))
    rows = []
    for day in range(days):
        at = datetime(2025, 5, 15, 15, tzinfo=UTC) + timedelta(days=day)
        for i in range(per_day):
            rows.append(
                ForecastExample(
                    config_id=config.config_id,
                    symbol="AAPL",
                    as_of=at + timedelta(minutes=5 * i),
                    target_time=at + timedelta(minutes=5 * i + 15),
                    session_date=at.date(),
                    anchor_price="100",
                    features=dict.fromkeys(FEATURE_NAMES, 0.0),
                    observation_ids=(f"{day}-{i}",),
                    provenance="historical",
                    label=LABELS[i % 3],
                    target_price="100",
                    target_return_bps=0,
                    label_observed_at=at + timedelta(minutes=5 * i + 15),
                )
            )
    return rows


def probabilities(rows, confidence=0.8):
    p = np.full((len(rows), 3), (1 - confidence) / 2)
    p[np.arange(len(rows)), [LABELS.index(row.label) for row in rows]] = confidence
    return p


def test_prediction_cohorts_reject_missing_duplicate_and_reordered_cases():
    from tradecopilot.forecast.supervised_study import _validate_probability_cohort

    ids = ("a", "b", "c")
    p = np.tile([0.2, 0.3, 0.5], (3, 1))
    np.testing.assert_array_equal(_validate_probability_cohort(ids, ids, p), p)
    for actual in (("a", "b"), ("a", "a", "c"), ("c", "a", "b")):
        with pytest.raises(ValueError, match="cohort"):
            _validate_probability_cohort(ids, actual, p[: len(actual)])
    for bad in (p[:-1], np.full((3, 3), np.nan), np.full((3, 3), 0.5), p[:, :2]):
        with pytest.raises(ValueError, match="probabilit"):
            _validate_probability_cohort(ids, ids, bad)


def test_primary_neural_distribution_requires_all_registered_seeds():
    from tradecopilot.forecast.supervised_study import _family_probabilities

    ids = ("a", "b")
    seed42 = np.tile([0.9, 0.05, 0.05], (2, 1))
    seed43 = np.tile([0.05, 0.9, 0.05], (2, 1))
    seed44 = np.tile([0.05, 0.05, 0.9], (2, 1))
    seeds = {42: (ids, seed42), 43: (ids, seed43), 44: (ids, seed44)}
    np.testing.assert_allclose(_family_probabilities(ids, seeds, (42, 43, 44)), 1 / 3)
    with pytest.raises(ValueError, match="all registered seeds"):
        _family_probabilities(ids, {42: seeds[42], 44: seeds[44]}, (42, 43, 44))


def test_all_abstain_retains_all_case_metrics_and_cannot_meet_target():
    from tradecopilot.forecast.supervised_study import _metrics

    rows = examples()
    metrics = _metrics(rows, probabilities(rows), None)
    assert metrics["accuracy"] == metrics["balanced_accuracy"] == metrics["macro_f1"] == 1
    assert metrics["missing_predictions"] == metrics["errors"] == 0
    assert metrics["coverage"] == metrics["selected"] == 0
    assert metrics["selective_accuracy"] is None
    assert metrics["target_requirements_met"] is False
    assert metrics["confusion_matrix"] == [[60, 0, 0], [0, 60, 0], [0, 0, 60]]
    assert len(metrics["by_session"]) == 20
    assert metrics["by_symbol"]["AAPL"]["accuracy"] == 1


def test_date_and_five_date_block_intervals_preserve_paired_case_weights():
    from tradecopilot.forecast.supervised_study import _paired_uncertainty, _uncertainty

    rows = examples()
    selected = probabilities(rows, 0.8)
    control = probabilities(rows, 0.4)
    settings = {"bootstrap_resamples": 1000, "seed": 42, "block_sensitivity_days": 5}
    paired = _paired_uncertainty(rows, control, selected, settings)
    assert paired["date_cluster"]["resamples"] == 1000
    assert paired["date_cluster"]["sessions"] == 20
    assert paired["five_date_block_sensitivity"]["clusters"] == 4
    assert paired["date_cluster"]["low"] == pytest.approx(np.log(2))
    assert paired["date_cluster"]["high"] == pytest.approx(np.log(2))
    intervals = _uncertainty(rows, selected, None, settings)
    assert intervals["date_cluster"]["accuracy"]["low"] == 1
    assert intervals["date_cluster"]["selective_accuracy"] is None


def test_metrics_present_absent_classes_explicitly():
    from tradecopilot.forecast.supervised_study import _metrics

    rows = [r.model_copy(update={"label": "FLAT", "session_date": date(2025, 5, 15)}) for r in examples(1)]
    metrics = _metrics(rows, np.tile([0.05, 0.9, 0.05], (len(rows), 1)), 0.5)
    assert metrics["accuracy"] == metrics["balanced_accuracy"] == 1
    assert metrics["macro_f1"] == pytest.approx(1 / 3)
    assert metrics["per_class"]["DOWN"]["support"] == 0
    assert metrics["per_class"]["DOWN"]["recall"] is None
    assert metrics["target_requirements_met"] is False


def registration():
    from tradecopilot.forecast.contracts import content_hash
    from tradecopilot.forecast.selective import GATE_REQUIREMENTS

    config = ForecastConfig(symbols=("AAPL",))
    reg = {
        "schema_version": "supervised-forecast-registration-v1",
        "experiment_type": "DIRECT_SUPERVISED_FORECAST",
        "evidence_mode": "historical_project_holdout",
        "forecast_config": config.model_dump(mode="json"),
        "cpu_candidates": [
            {"key": "prior", "kind": "training_class_prior"},
            {"key": "lr-001", "kind": "ohlcv_logistic", "C": 0.01},
            {"key": "lr-1", "kind": "ohlcv_logistic", "C": 1.0},
            {"key": "hgb-31", "kind": "ohlcv_hist_gradient_boosting"},
        ],
        "neural_candidates": [{"key": "tcn-32", "kind": "tcn"}, {"key": "lstm-64", "kind": "lstm"}],
        "neural_training": {
            "seeds": [42, 43, 44],
            "device": "cpu",
            "max_epochs": 20,
            "max_seconds_per_seed": 600,
            "batch_size": 256,
            "class_order": list(LABELS),
        },
        "uncertainty": {"bootstrap_resamples": 1000, "seed": 42, "block_sensitivity_days": 5},
        "gate_requirements": dict(GATE_REQUIREMENTS),
        "selection": {"production_promotion": False},
        "constraints": {"new_jev_calls": 0, "broker_orders": 0, "no_test_retuning": True},
    }
    return {**reg, "registration_id": content_hash(reg)}


def seed_result(tmp_path, reg, manifest, source):
    from tradecopilot.forecast.contracts import content_hash

    artifact = {
        "candidate_key": "lstm-64",
        "architecture": reg["neural_candidates"][1],
        "class_order": list(LABELS),
        "seed": 42,
        "registration_id": reg["registration_id"],
        "prepared_data_id": manifest["prepared_data_id"],
        "config_id": manifest["config_id"],
        "normalizer_hash": manifest["normalizer_id"],
        "source_hashes": source,
        "normalizers": manifest.get("normalizers", {}),
        "training_ids_hash": manifest["roles"]["TRAIN"]["case_ids_hash"],
        "tune_ids_hash": manifest["roles"]["TUNE"]["case_ids_hash"],
        "checkpoint": {"epoch": 1, "optimizer_updates": 1, "tune_nll": 0.2},
        "budget": {"max_epochs": 20, "max_seconds": 600, "max_optimizer_updates": 20, "batch_size": 256},
    }
    artifact["model_id"] = content_hash(artifact)
    import hashlib
    import json

    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "model.json"
    path.write_text(json.dumps(artifact))
    return {
        "status": "early_stopped",
        "eligible": True,
        "candidate_key": "lstm-64",
        "seed": 42,
        "model_path": str(path.resolve()),
        "checkpoint_hash": artifact["model_id"],
        "checkpoint_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "prepared_data_id": manifest["prepared_data_id"],
        "registration_id": reg["registration_id"],
        "config_id": manifest["config_id"],
        "normalizer_hash": manifest["normalizer_id"],
        "train_case_ids_hash": manifest["roles"]["TRAIN"]["case_ids_hash"],
        "tune_case_ids_hash": manifest["roles"]["TUNE"]["case_ids_hash"],
        "epochs_completed": 5,
        "optimizer_updates": 5,
        "elapsed_seconds": 1.0,
        "best_epoch": 1,
        "best_tune_nll": 0.2,
        "budget": artifact["budget"],
        "source_hashes_before": source,
        "source_hashes_after": source,
        "epoch_history": [{"epoch": epoch, "optimizer_updates": epoch, "tune_nll": 0.2} for epoch in range(1, 6)],
    }


def test_seed_validation_requires_completed_updates_budget_and_unchanged_sources(tmp_path):
    from tradecopilot.forecast.supervised_study import _validate_seed_result

    reg = registration()
    manifest = {
        "prepared_data_id": "prepared",
        "config_id": "config",
        "normalizer_id": "normalizer",
        "roles": {"TRAIN": {"count": 90, "case_ids_hash": "train"}, "TUNE": {"count": 90, "case_ids_hash": "tune"}},
    }
    source = {"module": "digest"}
    result = seed_result(tmp_path, reg, manifest, source)
    _validate_seed_result(result, "lstm-64", 42, reg, manifest, source)
    for changes in (
        {"optimizer_updates": 21},
        {"optimizer_updates": 0},
        {"elapsed_seconds": 600.01},
        {"status": "time_cap", "eligible": True},
        {"source_hashes_after": {"module": "changed"}},
        {"prepared_data_id": "wrong"},
        {"normalizer_hash": "wrong"},
        {"seed": 43},
        {"best_epoch": 2},
        {"epochs_completed": 21},
        {"eligible": False},
        {"status": "complete"},
        {"epoch_history": [{"epoch": 1, "optimizer_updates": 1, "tune_nll": 0.2}]},
        {"epoch_history": [{"epoch": epoch, "optimizer_updates": epoch, "tune_nll": 0.1} for epoch in range(1, 6)]},
    ):
        with pytest.raises(ValueError, match="seed"):
            _validate_seed_result({**result, **changes}, "lstm-64", 42, reg, manifest, source)


def test_worker_uses_absolute_registered_subprocess_and_preserves_timeout(tmp_path, monkeypatch):
    from tradecopilot.forecast import supervised_study as study

    calls = []

    class Process:
        returncode = 0

        def poll(self):
            return 0

    def popen(command, **kwargs):
        import json

        calls.append(command)
        output = Path(command[-1])
        output.mkdir()
        from tradecopilot.forecast.contracts import content_hash

        result = {"status": "complete", "eligible": True}
        (output / "result.json").write_text(json.dumps({**result, "result_id": content_hash(result)}))
        return Process()

    monkeypatch.setattr(study.subprocess, "Popen", popen)
    result = study._run_seed_worker(
        tmp_path,
        tmp_path / "registration.json",
        "tcn-32",
        42,
        tmp_path / "seed",
        registration(),
        tmp_path / "progress.json",
    )
    assert result["status"] == "complete"
    assert Path(calls[0][0]).is_absolute() and Path(calls[0][1]).is_absolute()
    assert calls[0][2:] == [
        "--prepared",
        str(tmp_path.resolve()),
        "--registration",
        str((tmp_path / "registration.json").resolve()),
        "--candidate",
        "tcn-32",
        "--seed",
        "42",
        "--output-dir",
        str((tmp_path / "seed").resolve()),
    ]
    assert result["supervisor"]["deadline_seconds"] == 660

    class TimedOut:
        returncode = None

        def poll(self):
            return None

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

    times = iter([0, 661, 662])
    monkeypatch.setattr(study.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(study.subprocess, "Popen", lambda *args, **kwargs: TimedOut())
    timed_out = study._run_seed_worker(
        tmp_path,
        tmp_path / "registration.json",
        "tcn-32",
        43,
        tmp_path / "timed-out",
        registration(),
        tmp_path / "progress.json",
    )
    assert timed_out["status"] == "supervisor_timeout"
    assert timed_out["eligible"] is False
    assert (tmp_path / "timed-out.log").exists()


def mock_study(tmp_path, monkeypatch, *, changed_role=None, failed_seed=None, confidence=None):
    import hashlib
    import json
    from types import SimpleNamespace

    from tradecopilot.forecast import supervised_study as study
    from tradecopilot.forecast.contracts import content_hash
    from tradecopilot.forecast.ohlcv_features import OHLCV_FEATURE_NAMES, OhlcvFeatureRecord
    from tradecopilot.forecast.supervised_data import SupervisedStage

    reg = registration()
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    registration_path = tmp_path / "registration.json"
    registration_path.write_text(json.dumps(reg))
    stages = {}
    roles = {}
    counts = {"TRAIN": 60, "TUNE": 10, "CAL": 10, "GATE": 10, "TEST": 20}
    start = 0
    for role, days in counts.items():
        rows = examples(days, 12)
        # Role inputs have different, label-independent identities; only outcomes mutate.
        rows = [
            row.model_copy(
                update={
                    "observation_ids": (role, *row.observation_ids),
                    "as_of": row.as_of + timedelta(days=start),
                    "target_time": row.target_time + timedelta(days=start),
                    "label_observed_at": row.label_observed_at + timedelta(days=start),
                    "session_date": row.session_date + timedelta(days=start),
                }
            )
            for row in rows
        ]
        start += days
        records = []
        for index, row in enumerate(rows):
            values = dict.fromkeys(OHLCV_FEATURE_NAMES, 0.0)
            values["return_5m_bps"] = float(index % 3)
            records.append(
                OhlcvFeatureRecord(
                    base_example_id=row.example_id,
                    config_id=row.config_id,
                    symbol=row.symbol,
                    as_of=row.as_of,
                    values=values,
                    input_bars_hash=row.example_id,
                )
            )
        static = np.zeros((len(rows), 115), dtype=np.float32)
        static[:, 0] = np.arange(len(rows)) % 3
        if role == changed_role:
            rows = [row.model_copy(update={"label": LABELS[(LABELS.index(row.label) + 1) % 3]}) for row in rows]
        case_ids = tuple(row.example_id for row in rows)
        stages[role] = SupervisedStage(
            role=role,
            sequence=np.zeros((len(rows), 60, 6), dtype=np.float32),
            static=static,
            examples=tuple(rows),
            features=tuple(records),
            targets=np.asarray([LABELS.index(row.label) for row in rows], dtype=np.int64),
            case_ids=case_ids,
        )
        artifact = prepared / f"{role}.json"
        artifact.write_text(json.dumps({"case_ids": case_ids}))
        ref = {
            "path": artifact.name,
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "bytes": artifact.stat().st_size,
        }
        roles[role] = {
            "count": len(rows),
            "case_ids_hash": content_hash(case_ids),
            "sequence": ref,
            "static": ref,
            "targets": ref,
            "examples": ref,
            "features": ref,
            "cases": ref,
        }
    catalog = prepared / "catalog.jsonl"
    catalog.write_text(
        "".join(
            json.dumps({"role": role, "eligible": True, "case_id": identity}) + "\n"
            for role, stage in stages.items()
            for identity in stage.case_ids
        )
    )
    normalizers = {
        "fit_role": "TRAIN",
        "sequence": {"mean": [0.0] * 6, "scale": [1.0] * 6},
        "static": {"mean": [0.0] * 55, "scale": [1.0] * 55, "binary_indices": []},
        "train_case_ids_hash": roles["TRAIN"]["case_ids_hash"],
        "train_raw_inputs_hash": "inputs",
    }
    manifest = {
        "schema_version": "supervised-prepared-v1",
        "registration": reg,
        "registration_id": reg["registration_id"],
        "forecast_config": reg["forecast_config"],
        "config_id": ForecastConfig.model_validate(reg["forecast_config"]).config_id,
        "source_data_id": "synthetic-control-flow-source",
        "source_manifest": {"fixture": True},
        "normalizers": normalizers,
        "normalizer_id": content_hash(normalizers),
        "class_order": list(LABELS),
        "roles": roles,
        "role_counts": {role: stage.targets.size for role, stage in stages.items()},
        "planned_session_counts": counts,
        "catalog": {
            "path": catalog.name,
            "sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
            "bytes": catalog.stat().st_size,
        },
    }
    manifest["prepared_data_id"] = content_hash(manifest)
    (prepared / "manifest.json").write_text(json.dumps(manifest))
    source = {"module": "unchanged"}
    events = []
    confidences = {
        "prior": 0.34,
        "lr-001": 0.45,
        "lr-1": 0.65,
        "hgb-31": 0.55,
        "tcn-32": {42: 0.99, 43: 0.35, 44: 0.35},
        "lstm-64": {42: 0.8, 43: 0.8, 44: 0.8},
    }
    if confidence:
        confidences.update(confidence)

    def load_stage(directory, role, *, selection_path=None):
        events.append(role)
        if role == "TEST":
            assert selection_path is not None
            selection = json.loads(selection_path.read_text())
            assert selection["stage"] == "selection_frozen"
            assert selection["selection_id"] == content_hash(
                {k: v for k, v in selection.items() if k != "selection_id"}
            )
            assert {
                "selected_artifact_ids",
                "selected_checkpoint_ids",
                "normalizer_id",
                "temperature",
                "gate",
                "cpu_reference_artifact_id",
                "source_data_id",
                "config_id",
                "class_order",
            } <= selection.keys()
        return stages[role], manifest

    def p(values, c):
        result = np.full((len(values), 3), (1 - c) / 2)
        result[np.arange(len(values)), np.asarray(values, dtype=int)] = c
        return result

    def fit_cpu(key, train, incoming, registered):
        assert events == ["TRAIN", "TUNE"]
        artifact = {
            "candidate_key": key,
            "prepared_data_id": manifest["prepared_data_id"],
            "registration_id": reg["registration_id"],
            "registration": reg,
            "source_hashes": source,
            "config_id": manifest["config_id"],
            "class_order": list(LABELS),
            "training_ids": list(train.case_ids),
            "training_ids_hash": roles["TRAIN"]["case_ids_hash"],
            "normalizer_hash": manifest["normalizer_id"],
            "normalizers": normalizers,
        }
        artifact["model_id"] = content_hash(artifact)
        return artifact, object()

    def predict_cpu(artifact, features):
        assert all(not hasattr(feature, "label") for feature in features)
        return p([f.values["return_5m_bps"] for f in features], confidences[artifact["candidate_key"]])

    def save_cpu(artifact, path):
        path.write_text(json.dumps(artifact))
        return path

    def neural(path):
        artifact = json.loads(path.read_text())
        c = confidences[artifact["candidate_key"]][artifact["seed"]]
        return SimpleNamespace(
            artifact=artifact, model_id=artifact["model_id"], probabilities=lambda seq, static: p(static[:, 0], c)
        )

    def worker(directory, path, candidate, seed, output, registered, progress):
        assert events == ["TRAIN", "TUNE"]
        result = seed_result(output, reg, manifest, source)
        artifact = json.loads((output / "model.json").read_text())
        artifact.update(
            candidate_key=candidate,
            seed=seed,
            architecture=next(c for c in reg["neural_candidates"] if c["key"] == candidate),
            training_ids=list(stages["TRAIN"].case_ids),
            tune_ids=list(stages["TUNE"].case_ids),
        )
        artifact["budget"] = study._seed_budget(reg, manifest)
        artifact["checkpoint"]["optimizer_updates"] = int(np.ceil(len(stages["TRAIN"].case_ids) / 256))
        artifact["checkpoint"]["tune_nll"] = float(-np.log(confidences[candidate][seed]))
        artifact["model_id"] = content_hash({k: v for k, v in artifact.items() if k != "model_id"})
        (output / "model.json").write_text(json.dumps(artifact))
        (output / "checkpoint-01.json").write_text(json.dumps(artifact))
        result.update(
            candidate_key=candidate,
            seed=seed,
            checkpoint_hash=artifact["model_id"],
            checkpoint_file_sha256=hashlib.sha256((output / "model.json").read_bytes()).hexdigest(),
            optimizer_updates=5 * artifact["checkpoint"]["optimizer_updates"],
            budget=artifact["budget"],
            best_tune_nll=artifact["checkpoint"]["tune_nll"],
        )
        result["epoch_history"] = [
            {
                "epoch": epoch,
                "optimizer_updates": epoch * artifact["checkpoint"]["optimizer_updates"],
                "tune_nll": artifact["checkpoint"]["tune_nll"],
            }
            for epoch in range(1, 6)
        ]
        result["checkpoints"] = [
            {
                "path": str(output / "checkpoint-01.json"),
                "model_id": artifact["model_id"],
                "epoch": 1,
                "sha256": result["checkpoint_file_sha256"],
                "tune_nll": artifact["checkpoint"]["tune_nll"],
            }
        ]
        if (candidate, seed) == failed_seed:
            result.update(status="time_cap", eligible=False)
        inputs = {
            str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob("*")
            if p.is_file()
        } | {"registration_file": hashlib.sha256(path.read_bytes()).hexdigest()}
        result.update(prepared_input_hashes_before=inputs, prepared_input_hashes_after=inputs)
        result["result_id"] = content_hash(result)
        (output / "result.json").write_text(json.dumps(result))
        return result

    models = SimpleNamespace(
        source_hashes=lambda: source,
        fit_cpu=fit_cpu,
        predict_cpu=predict_cpu,
        native_cpu_probabilities=lambda a, n, f: predict_cpu(a, f),
        save_cpu_artifact=save_cpu,
        load_cpu_artifact=lambda path: json.loads(path.read_text()),
        load_neural_artifact=neural,
        prepared_input_hashes=lambda directory, path: (
            {
                str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in directory.rglob("*")
                if p.is_file()
            }
            | {"registration_file": hashlib.sha256(path.read_bytes()).hexdigest()}
        ),
    )
    monkeypatch.setattr(study, "load_stage", load_stage, raising=False)
    monkeypatch.setattr(study, "_model_api", lambda: models, raising=False)
    monkeypatch.setattr(study, "_run_seed_worker", worker)
    return prepared, registration_path, events, stages, manifest


def test_study_freezes_full_selection_before_test_and_never_refits(tmp_path, monkeypatch):
    from tradecopilot.forecast.supervised_study import load_report, run_supervised_study

    prepared, reg, events, _, manifest = mock_study(tmp_path, monkeypatch)
    report_path = run_supervised_study(prepared, reg, tmp_path / "study")
    report = load_report(report_path)
    assert events == ["TRAIN", "TUNE", "CAL", "GATE", "TEST"]
    assert report["selection"]["selected_candidate_id"] == "lstm-64"
    assert report["selection"]["cpu_reference_candidate_id"] == "lr-1"
    assert len(report["selection"]["selected_checkpoint_ids"]) == 3
    assert report["prepared_data_id"] == manifest["prepared_data_id"]
    assert len(report["training"]["neural_seeds"]) == 6
    assert len(report["test"]["candidate_primaries"]) == 6
    assert len(report["test"]["individual_seed_diagnostics"]) == 6
    assert report["test"]["selected_raw"]["accuracy"] == report["test"]["selected_calibrated"]["accuracy"]
    assert report["flags"]["automatic_production_promotion"] is False
    assert report["constraints"]["new_jev_calls"] == report["constraints"]["broker_orders"] == 0
    assert report["corpus_status"] == "retired_after_outcome_scoring"
    assert (report_path.parent.stat().st_mode & 0o077) == 0
    with pytest.raises(FileExistsError):
        run_supervised_study(prepared, reg, report_path.parent)


@pytest.mark.parametrize("role", ["CAL", "GATE", "TEST"])
def test_later_outcomes_cannot_change_earlier_selection(tmp_path, monkeypatch, role):
    from tradecopilot.forecast.supervised_study import load_report, run_supervised_study

    first = tmp_path / "one"
    first.mkdir()
    prepared, reg, _, _, _ = mock_study(first, monkeypatch)
    baseline = load_report(run_supervised_study(prepared, reg, first / "study"))
    second = tmp_path / "two"
    second.mkdir()
    prepared, reg, _, _, _ = mock_study(second, monkeypatch, changed_role=role)
    changed = load_report(run_supervised_study(prepared, reg, second / "study"))
    keys = ("selected_candidate_id", "cpu_reference_candidate_id", "selected_artifact_ids", "selected_checkpoint_ids")
    assert all(baseline["selection"][key] == changed["selection"][key] for key in keys)
    if role in {"GATE", "TEST"}:
        assert baseline["selection"]["temperature"] == changed["selection"]["temperature"]
    if role == "TEST":
        assert baseline["selection"] == changed["selection"]
        assert baseline["test"]["selected_calibrated"]["accuracy"] != changed["test"]["selected_calibrated"]["accuracy"]


def test_best_seed_cannot_win_and_failed_seed_invalidates_whole_family(tmp_path, monkeypatch):
    from tradecopilot.forecast.supervised_study import load_report, run_supervised_study

    prepared, reg, _, _, _ = mock_study(tmp_path, monkeypatch, failed_seed=("lstm-64", 43))
    report = load_report(run_supervised_study(prepared, reg, tmp_path / "study"))
    assert report["selection"]["selected_candidate_id"] == "lr-1"
    assert report["selection"]["selected_checkpoint_ids"] == []
    assert report["candidates"]["lstm-64"]["eligible"] is False
    assert "lstm-64" not in report["test"]["candidate_primaries"]
    assert "lstm-64:42" in report["test"]["individual_seed_diagnostics"]
    assert "lstm-64:43" not in report["test"]["individual_seed_diagnostics"]
    assert report["candidates"]["tcn-32"]["tune_log_loss"] > report["candidates"]["lr-1"]["tune_log_loss"]


def test_report_inventory_rejects_source_selection_prediction_and_extra_file_tampering(tmp_path, monkeypatch):
    from tradecopilot.forecast.supervised_study import load_report, run_supervised_study

    prepared, reg, _, _, _ = mock_study(tmp_path, monkeypatch)
    report_path = run_supervised_study(prepared, reg, tmp_path / "study")
    for name in ("source.json", "frozen-selection.json", "predictions.jsonl"):
        artifact = report_path.parent / name
        original = artifact.read_bytes()
        artifact.write_bytes(original + b" ")
        with pytest.raises(ValueError, match="integrity"):
            load_report(report_path)
        artifact.write_bytes(original)
    extra = report_path.parent / "undeclared.json"
    extra.write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        load_report(report_path)
    extra.unlink()
    original = report_path.read_bytes()
    report_path.write_bytes(original + b"x")
    with pytest.raises(ValueError, match="integrity"):
        load_report(report_path)


def test_source_change_during_training_preserves_failure_and_blocks_later_roles(tmp_path, monkeypatch):
    from tradecopilot.forecast import supervised_study as study

    prepared, reg, events, _, _ = mock_study(tmp_path, monkeypatch)
    models = study._model_api()
    snapshots = iter([{"module": "unchanged"}, {"module": "changed"}])
    models.source_hashes = lambda: next(snapshots)
    with pytest.raises(ValueError, match="source/data/config changed"):
        study.run_supervised_study(prepared, reg, tmp_path / "study")
    assert events == ["TRAIN", "TUNE"]
    assert (tmp_path / "study" / "failure.json").is_file()
    assert (tmp_path / "study" / "training.json").is_file()
    assert not (tmp_path / "study" / "frozen-selection.json").exists()


def test_bad_cpu_and_seed_predictions_are_ineligible_before_selection(tmp_path, monkeypatch):
    from tradecopilot.forecast import supervised_study as study

    prepared, reg, _, _, _ = mock_study(tmp_path, monkeypatch)
    models = study._model_api()
    original = models.predict_cpu
    models.predict_cpu = lambda artifact, features: (
        original(artifact, features)[:-1] if artifact["candidate_key"] == "lr-1" else original(artifact, features)
    )
    original_neural = models.load_neural_artifact

    def neural(path):
        predictor = original_neural(path)
        if predictor.artifact["candidate_key"] == "tcn-32" and predictor.artifact["seed"] == 42:
            method = predictor.probabilities
            predictor.probabilities = lambda seq, static: method(seq, static)[:-1]
        return predictor

    models.load_neural_artifact = neural
    report = study.load_report(study.run_supervised_study(prepared, reg, tmp_path / "study"))
    assert report["candidates"]["lr-1"]["eligible"] is False
    assert report["candidates"]["tcn-32"]["eligible"] is False
    assert report["selection"]["cpu_reference_candidate_id"] == "hgb-31"
    assert "lr-1" not in report["test"]["candidate_primaries"]
    assert "tcn-32:42" not in report["test"]["individual_seed_diagnostics"]


def test_prefit_binds_reviewable_method_registry_and_clarification(tmp_path, monkeypatch):
    import json

    from tradecopilot.forecast import supervised_study as study
    from tradecopilot.forecast.contracts import content_hash

    prepared, reg_path, _, _, manifest = mock_study(tmp_path, monkeypatch)
    reg = json.loads(reg_path.read_text())
    clarification = {
        "schema_version": "supervised-pretraining-clarification-v1",
        "registration_id": reg["registration_id"],
        "before_learning_and_quality_scoring": True,
        "comparison_scope": study.COMPARISON_SCOPE,
    }
    clarification["clarification_id"] = content_hash(clarification)
    (tmp_path / "pretraining-clarification.json").write_text(json.dumps(clarification))
    registry = {
        "schema_version": "supervised-method-registry-v1",
        "registration_id": reg["registration_id"],
        "pretraining_clarification_id": clarification["clarification_id"],
        "before_learning_and_quality_scoring": True,
        "comparison_scope": study.COMPARISON_SCOPE,
        "bootstrap": {
            "ordering": "ascending_session_dates",
            "blocks": "consecutive_nonoverlapping_starting_at_first_role_date",
            "block_size_dates": 5,
            "resample": "same_number_of_blocks_with_replacement",
            "replicates": 1000,
            "seed": 42,
            "weighting": "ratio_of_total_case_sums_to_total_case_count",
            "interval": "percentiles_2.5_97.5",
            "test_block_count": 4,
        },
    }
    registry["method_registry_id"] = content_hash(registry)
    (tmp_path / "method-registry.json").write_text(json.dumps(registry))
    manifest.update(method_registry=registry, method_registry_id=registry["method_registry_id"])
    manifest["prepared_data_id"] = content_hash({k: v for k, v in manifest.items() if k != "prepared_data_id"})
    (prepared / "manifest.json").write_text(json.dumps(manifest))
    report_path = study.run_supervised_study(prepared, reg_path, tmp_path / "study")
    report = study.load_report(report_path)
    assert report["method_registry_id"] == report["selection"]["method_registry_id"] == registry["method_registry_id"]
    assert report["clarification_id"] == clarification["clarification_id"]
    assert "method-registry.json" in report["inventory"]
    assert "pretraining-clarification.json" in report["inventory"]
    assert report["test"]["paired_reference_improvement"]["comparison_scope"] == study.COMPARISON_SCOPE


def test_script_exposes_required_paths_without_training(tmp_path, monkeypatch, capsys):
    import runpy
    import sys

    from tradecopilot.forecast import supervised_study as study

    calls = []

    def run(prepared, registration_path, output):
        calls.append((prepared, registration_path, output))
        return output / "report.json"

    monkeypatch.setattr(study, "run_supervised_study", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_supervised_forecast_study.py",
            "--prepared",
            str(tmp_path / "data"),
            "--registration",
            str(tmp_path / "registration.json"),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_supervised_forecast_study.py"
    runpy.run_path(str(script), run_name="__main__")
    assert calls == [(tmp_path / "data", tmp_path / "registration.json", tmp_path / "out")]
    assert capsys.readouterr().out.strip().endswith("out/report.json")


@pytest.mark.parametrize("tamper", ["result_id", "prepared_input_hashes_after", "checkpoints"])
def test_seed_result_and_input_seals_are_required_before_family_selection(tmp_path, monkeypatch, tamper):
    import json

    from tradecopilot.forecast import supervised_study as study
    from tradecopilot.forecast.contracts import content_hash

    prepared, reg, _, _, _ = mock_study(tmp_path, monkeypatch)
    original = study._run_seed_worker

    def worker(*args):
        result = original(*args)
        result[tamper] = (
            "invalid" if tamper == "result_id" else [] if tamper == "checkpoints" else {"manifest.json": "changed"}
        )
        if tamper != "result_id":
            result["result_id"] = content_hash({k: v for k, v in result.items() if k != "result_id"})
        (args[4] / "result.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(study, "_run_seed_worker", worker)
    report = study.load_report(study.run_supervised_study(prepared, reg, tmp_path / "study"))
    assert report["selection"]["selected_candidate_id"] == "lr-1"
    assert not report["candidates"]["lstm-64"]["eligible"]
    assert not report["candidates"]["tcn-32"]["eligible"]
    assert report["test"]["individual_seed_diagnostics"] == {}
