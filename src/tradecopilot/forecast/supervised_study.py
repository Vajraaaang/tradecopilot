"""Registered direct forecasts with TRAIN/TUNE selection and sealed CAL/GATE/TEST stages."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from contextlib import suppress
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from tradecopilot.forecast.contracts import LABELS, ForecastConfig, ForecastExample, content_hash
from tradecopilot.forecast.selective import (
    GATE_REQUIREMENTS,
    GATE_THRESHOLDS,
    apply_temperature,
    fit_temperature,
    gate_qualifies,
    select_gate,
    selection_metrics,
)

if TYPE_CHECKING:
    from tradecopilot.forecast.supervised_data import SupervisedStage

REPORT_VERSION = "supervised-forecast-report-v1"
COMPARISON_SCOPE = "selected_raw_primary_vs_raw_tune_selected_cpu"


def _validate_probability_cohort(
    expected: tuple[str, ...], actual: tuple[str, ...], probabilities: NDArray[np.float64]
) -> NDArray[np.float64]:
    if not expected or len(set(expected)) != len(expected) or actual != expected or len(set(actual)) != len(actual):
        raise ValueError("prediction cohort must contain every ordered eligible case exactly once")
    p = np.asarray(probabilities, dtype=np.float64)
    if (
        p.shape != (len(expected), 3)
        or not np.isfinite(p).all()
        or (p < 0).any()
        or (p > 1).any()
        or not np.allclose(p.sum(axis=1), 1, atol=1e-8, rtol=0)
    ):
        raise ValueError("invalid complete DOWN/FLAT/UP probabilities")
    return p


def _family_probabilities(
    case_ids: tuple[str, ...],
    outputs: dict[int, tuple[tuple[str, ...], NDArray[np.float64]]],
    seeds: tuple[int, ...],
) -> NDArray[np.float64]:
    if len(seeds) != 3 or len(set(seeds)) != 3 or set(outputs) != set(seeds):
        raise ValueError("a primary neural family requires all registered seeds; partial ensembles are ineligible")
    p = np.mean([_validate_probability_cohort(case_ids, *outputs[seed]) for seed in seeds], axis=0)
    return _validate_probability_cohort(case_ids, case_ids, p)


def _targets(examples: Sequence[ForecastExample]) -> NDArray[np.int64]:
    if not examples or any(row.label not in LABELS for row in examples):
        raise ValueError("complete labeled evaluation examples required")
    return np.asarray([LABELS.index(str(row.label)) for row in examples], dtype=np.int64)


def _losses(p: NDArray[np.float64], targets: NDArray[np.int64]) -> NDArray[np.float64]:
    return -np.log(np.clip(p[np.arange(len(p)), targets], 1e-15, 1))


def _summary(examples: Sequence[ForecastExample], p: NDArray[np.float64]) -> dict[str, Any]:
    y = _targets(examples)
    predicted = p.argmax(axis=1)
    cm = np.bincount(3 * y + predicted, minlength=9).reshape(3, 3)
    supports, predicted_counts = cm.sum(axis=1), cm.sum(axis=0)
    diagonal = np.diag(cm)
    per_class = {}
    f1 = np.zeros(3)
    recalls = []
    for i, label in enumerate(LABELS):
        precision = float(diagonal[i] / predicted_counts[i]) if predicted_counts[i] else None
        recall = float(diagonal[i] / supports[i]) if supports[i] else None
        denominator = supports[i] + predicted_counts[i]
        f1[i] = float(2 * diagonal[i] / denominator) if denominator else 0
        if recall is not None:
            recalls.append(recall)
        per_class[label] = {
            "support": int(supports[i]),
            "predicted": int(predicted_counts[i]),
            "precision": precision,
            "recall": recall,
            "f1": float(f1[i]),
        }
    return {
        "eligible": len(y),
        "accuracy": float((predicted == y).mean()),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "log_loss": float(_losses(p, y).mean()),
        "brier_score": float(np.square(p - np.eye(3)[y]).sum(axis=1).mean()),
        "confusion_matrix": cm.tolist(),
        "labels": list(LABELS),
        "class_counts": dict(zip(LABELS, supports.tolist(), strict=True)),
        "per_class": per_class,
    }


def _metrics(examples: Sequence[ForecastExample], p: NDArray[np.float64], threshold: float | None) -> dict[str, Any]:
    ids = tuple(row.example_id for row in examples)
    p = _validate_probability_cohort(ids, ids, p)
    selection = selection_metrics(examples, p, threshold, confidence_interval=False)
    result = {
        **_summary(examples, p),
        **selection,
        "attempted": len(p),
        "scored": len(p),
        "errors": 0,
        "missing_predictions": 0,
        "abstained": len(p) - selection["selected"],
        "probability_available_coverage": 1.0,
        "target_requirements_met": gate_qualifies(selection),
    }
    for day, metrics in result["by_session"].items():
        indices = [i for i, row in enumerate(examples) if str(row.session_date) == day]
        metrics.update(_summary([examples[i] for i in indices], p[indices]))
    result["by_symbol"] = {}
    for symbol in sorted({row.symbol for row in examples}):
        indices = [i for i, row in enumerate(examples) if row.symbol == symbol]
        members = [examples[i] for i in indices]
        result["by_symbol"][symbol] = {
            **_summary(members, p[indices]),
            **{
                key: value
                for key, value in selection_metrics(members, p[indices], threshold, confidence_interval=False).items()
                if key not in {"by_session", "selective_accuracy_interval"}
            },
        }
    y = _targets(examples)
    correct = p.argmax(axis=1) == y
    result["coverage_curve"] = []
    for value in GATE_THRESHOLDS:
        mask = p.max(axis=1) >= value
        result["coverage_curve"].append(
            {
                "threshold": value,
                "coverage": float(mask.mean()),
                "selected": int(mask.sum()),
                "selective_accuracy": float(correct[mask].mean()) if mask.any() else None,
            }
        )
    return result


def _date_groups(examples: Sequence[ForecastExample], block_days: int = 1) -> list[NDArray[np.int64]]:
    dates = sorted({row.session_date for row in examples})
    return [
        np.asarray(
            [i for i, row in enumerate(examples) if row.session_date in dates[start : start + block_days]],
            dtype=np.int64,
        )
        for start in range(0, len(dates), block_days)
    ]


def _interval(values: NDArray[np.float64]) -> dict[str, float] | None:
    valid = values[np.isfinite(values)]
    if not len(valid):
        return None
    return {"low": float(np.percentile(valid, 2.5)), "high": float(np.percentile(valid, 97.5))}


def _bootstrap_metrics(
    examples: Sequence[ForecastExample],
    p: NDArray[np.float64],
    threshold: float | None,
    settings: dict[str, Any],
    block_days: int,
) -> dict[str, Any]:
    y = _targets(examples)
    predicted = p.argmax(axis=1)
    correct = predicted == y
    selected = np.zeros(len(p), dtype=bool) if threshold is None else p.max(axis=1) >= threshold
    losses = _losses(p, y)
    brier = np.square(p - np.eye(3)[y]).sum(axis=1)
    groups = _date_groups(examples, block_days)
    stats = np.asarray(
        [
            [
                len(indices),
                correct[indices].sum(),
                losses[indices].sum(),
                brier[indices].sum(),
                selected[indices].sum(),
                (correct[indices] & selected[indices]).sum(),
                *np.bincount(3 * y[indices] + predicted[indices], minlength=9),
            ]
            for indices in groups
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(settings["seed"])
    sampled = stats[rng.integers(0, len(groups), (settings["bootstrap_resamples"], len(groups)))].sum(axis=1)
    cm = sampled[:, 6:].reshape(-1, 3, 3)
    diagonal = np.diagonal(cm, axis1=1, axis2=2)
    supports = cm.sum(axis=2)
    denominator = supports + cm.sum(axis=1)
    f1 = np.divide(2 * diagonal, denominator, out=np.zeros_like(diagonal), where=denominator > 0)
    recalls = np.divide(diagonal, supports, out=np.zeros_like(diagonal), where=supports > 0)
    selective = np.divide(sampled[:, 5], sampled[:, 4], out=np.full(len(sampled), np.nan), where=sampled[:, 4] > 0)
    return {
        "method": "whole_session_date_bootstrap_95pct"
        if block_days == 1
        else "consecutive_5_date_block_bootstrap_95pct",
        "resamples": settings["bootstrap_resamples"],
        "seed": settings["seed"],
        "sessions": len({row.session_date for row in examples}),
        "clusters": len(groups),
        "block_days": block_days,
        "block_session_counts": [len({examples[i].session_date for i in indices}) for indices in groups],
        "accuracy": _interval(sampled[:, 1] / sampled[:, 0]),
        "macro_f1": _interval(f1.mean(axis=1)),
        "balanced_accuracy": _interval(recalls.sum(axis=1) / (supports > 0).sum(axis=1)),
        "log_loss": _interval(sampled[:, 2] / sampled[:, 0]),
        "brier_score": _interval(sampled[:, 3] / sampled[:, 0]),
        "coverage": _interval(sampled[:, 4] / sampled[:, 0]),
        "selective_accuracy": _interval(selective),
    }


def _uncertainty(
    examples: Sequence[ForecastExample],
    p: NDArray[np.float64],
    threshold: float | None,
    settings: dict[str, Any],
) -> dict[str, Any]:
    return {
        "date_cluster": _bootstrap_metrics(examples, p, threshold, settings, 1),
        "five_date_block_sensitivity": _bootstrap_metrics(
            examples, p, threshold, settings, settings["block_sensitivity_days"]
        ),
    }


def _paired_uncertainty(
    examples: Sequence[ForecastExample],
    control: NDArray[np.float64],
    selected: NDArray[np.float64],
    settings: dict[str, Any],
) -> dict[str, Any]:
    y = _targets(examples)
    differences = _losses(control, y) - _losses(selected, y)
    result: dict[str, Any] = {"mean_control_minus_selected_log_loss": float(differences.mean())}
    for name, days in (("date_cluster", 1), ("five_date_block_sensitivity", settings["block_sensitivity_days"])):
        groups = _date_groups(examples, days)
        stats = np.asarray([[len(indices), differences[indices].sum()] for indices in groups], dtype=np.float64)
        rng = np.random.default_rng(settings["seed"])
        sampled = stats[rng.integers(0, len(groups), (settings["bootstrap_resamples"], len(groups)))].sum(axis=1)
        result[name] = {
            "method": "paired_whole_session_date_bootstrap_95pct"
            if days == 1
            else "paired_consecutive_5_date_block_95pct",
            "resamples": settings["bootstrap_resamples"],
            "seed": settings["seed"],
            "sessions": len({row.session_date for row in examples}),
            "clusters": len(groups),
            "block_days": days,
            **(_interval(sampled[:, 1] / sampled[:, 0]) or {}),
        }
    return result


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError("symlinked study artifact is not allowed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("study artifact must be a JSON object")
    content_hash(value)  # Reject nonstandard JSON NaN and infinity before using even progress artifacts.
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def _seal(path: Path, value: dict[str, Any], key: str) -> dict[str, Any]:
    result = {**value, key: content_hash(value)}
    _write_json(path, result)
    return result


def _integer(value: Any, minimum: int, maximum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and minimum <= value <= maximum


def _seed_budget(registration: dict[str, Any], manifest: dict[str, Any]) -> dict[str, int]:
    training = registration["neural_training"]
    batches = math.ceil(manifest["roles"]["TRAIN"]["count"] / training["batch_size"])
    return {
        "max_epochs": training["max_epochs"],
        "max_seconds": training["max_seconds_per_seed"],
        "max_optimizer_updates": training["max_epochs"] * batches,
        "batch_size": training["batch_size"],
    }


def _validate_seed_result(
    result: dict[str, Any],
    candidate: str,
    seed: int,
    registration: dict[str, Any],
    manifest: dict[str, Any],
    source: dict[str, str],
) -> list[int]:
    expected = {
        "candidate_key": candidate,
        "seed": seed,
        "registration_id": registration["registration_id"],
        "prepared_data_id": manifest["prepared_data_id"],
        "config_id": manifest["config_id"],
        "normalizer_hash": manifest["normalizer_id"],
        "train_case_ids_hash": manifest["roles"]["TRAIN"]["case_ids_hash"],
        "tune_case_ids_hash": manifest["roles"]["TUNE"]["case_ids_hash"],
        "source_hashes_before": source,
        "source_hashes_after": source,
        "budget": _seed_budget(registration, manifest),
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("neural seed source/data/registration/budget identity mismatch")
    elapsed = result.get("elapsed_seconds")
    epochs = result.get("epochs_completed")
    updates = result.get("optimizer_updates")
    best = result.get("best_epoch")
    loss = result.get("best_tune_nll")
    budget = expected["budget"]
    batches = budget["max_optimizer_updates"] // budget["max_epochs"]
    if (
        result.get("status") not in {"complete", "early_stopped"}
        or result.get("eligible") is not True
        or not _integer(epochs, 1, budget["max_epochs"])
        or not _integer(updates, 1, budget["max_optimizer_updates"])
        or updates != epochs * batches
        or not _integer(best, 1, int(epochs or 0))
        or not isinstance(elapsed, (float, int))
        or isinstance(elapsed, bool)
        or not math.isfinite(elapsed)
        or not 0 <= elapsed <= budget["max_seconds"]
        or not isinstance(loss, (float, int))
        or isinstance(loss, bool)
        or not math.isfinite(loss)
        or loss < 0
    ):
        raise ValueError("neural seed did not finish complete registered epochs within its budget")
    supervisor = result.get("supervisor")
    if supervisor is not None and (
        supervisor.get("timed_out") is not False
        or supervisor.get("returncode") != 0
        or supervisor.get("wall_seconds", math.inf) > budget["max_seconds"] + 60
    ):
        raise ValueError("neural seed exceeded its external supervisor deadline")
    history = result.get("epoch_history")
    if not isinstance(history, list) or len(history) != epochs:
        raise ValueError("neural seed requires every completed epoch in its sealed history")
    best_loss, best_epoch, stale = math.inf, 0, 0
    checkpoint_epochs = []
    for epoch, row in enumerate(history, 1):
        loss = row.get("tune_nll") if isinstance(row, dict) else None
        if (
            not isinstance(row, dict)
            or row.get("epoch") != epoch
            or row.get("optimizer_updates") != epoch * batches
            or not isinstance(loss, (int, float))
            or isinstance(loss, bool)
            or not math.isfinite(loss)
            or loss < 0
        ):
            raise ValueError("neural seed epoch history/update accounting mismatch")
        if best_loss - loss > 1e-4:
            best_loss, best_epoch, stale = loss, epoch, 0
            checkpoint_epochs.append(epoch)
        else:
            stale += 1
        if stale >= 4 and epoch != epochs:
            raise ValueError("neural seed continued beyond its registered patience")
    if (
        best_epoch != result["best_epoch"]
        or best_loss != result["best_tune_nll"]
        or (result["status"] == "complete" and (epochs != budget["max_epochs"] or stale >= 4))
        or (result["status"] == "early_stopped" and stale != 4)
    ):
        raise ValueError("neural seed earliest-best checkpoint/termination status mismatch")
    return checkpoint_epochs


def _run_seed_worker(
    prepared: Path,
    registration_path: Path,
    candidate: str,
    seed: int,
    output: Path,
    registration: dict[str, Any],
    progress: Path,
) -> dict[str, Any]:
    worker = Path(__file__).resolve().parents[3] / "scripts" / "train_supervised_seed.py"
    command = [
        str(Path(sys.executable).absolute()),
        str(worker),
        "--prepared",
        str(prepared.resolve()),
        "--registration",
        str(registration_path.resolve()),
        "--candidate",
        candidate,
        "--seed",
        str(seed),
        "--output-dir",
        str(output.resolve()),
    ]
    if output.exists():
        raise FileExistsError("choose a new seed directory; learner artifacts are immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    deadline = registration["neural_training"]["max_seconds_per_seed"] + 60
    started = time.monotonic()
    timed_out = False
    log_path = output.parent / f"{output.name}.log"
    with log_path.open("w", encoding="utf-8") as log:
        log_path.chmod(0o600)
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            status = {"stage": "training", "candidate_key": candidate, "seed": seed}
            child = output / "progress.json"
            if child.is_file():
                with suppress(ValueError, OSError):
                    status["worker"] = _read_json(child)
            _write_json(progress, status)
            if time.monotonic() - started > deadline:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                break
            time.sleep(0.2)
    wall = time.monotonic() - started
    supervisor = {
        "command": command,
        "deadline_seconds": deadline,
        "setup_grace_seconds": 60,
        "wall_seconds": wall,
        "returncode": process.returncode,
        "timed_out": timed_out,
    }
    if timed_out or process.returncode != 0:
        return {
            "candidate_key": candidate,
            "seed": seed,
            "status": "supervisor_timeout" if timed_out else "worker_failed",
            "eligible": False,
            "supervisor": supervisor,
        }
    try:
        result = _read_sealed(output / "result.json", "result_id")
    except (ValueError, OSError) as exc:
        return {
            "candidate_key": candidate,
            "seed": seed,
            "status": "invalid_worker_result",
            "eligible": False,
            "reason": str(exc),
            "supervisor": supervisor,
        }
    return {**result, "supervisor": supervisor}


def load_stage(
    prepared: Path, role: str, *, selection_path: Path | None = None
) -> tuple[SupervisedStage, dict[str, Any]]:
    from tradecopilot.forecast.supervised_data import load_stage as read_stage

    return read_stage(prepared, role, selection_path=selection_path)


def _model_api() -> Any:
    from tradecopilot.forecast import supervised_models

    return supervised_models


def _file_hash(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or symlinked study input/artifact")
    return sha256(path.read_bytes()).hexdigest()


def _read_sealed(path: Path, key: str) -> dict[str, Any]:
    value = _read_json(path)
    if value.get(key) != content_hash({name: item for name, item in value.items() if name != key}):
        raise ValueError("study JSON integrity seal mismatch")
    return value


def _registration(path: Path) -> dict[str, Any]:
    reg = _read_sealed(path, "registration_id")
    if (
        reg.get("schema_version") != "supervised-forecast-registration-v1"
        or reg.get("experiment_type") != "DIRECT_SUPERVISED_FORECAST"
        or [c["key"] for c in reg.get("cpu_candidates", [])] != ["prior", "lr-001", "lr-1", "hgb-31"]
        or [c["key"] for c in reg.get("neural_candidates", [])] != ["tcn-32", "lstm-64"]
        or reg.get("gate_requirements") != GATE_REQUIREMENTS
        or reg.get("uncertainty")
        != {
            "unit": "wholesessiondatewithallsymbolcases",
            "bootstrap_resamples": 1000,
            "seed": 42,
            "block_sensitivity_days": 5,
        }
        or reg.get("selection", {}).get("production_promotion") is not False
        or any(reg.get("constraints", {}).get(key) != 0 for key in ("new_jev_calls", "broker_orders"))
        or reg.get("constraints", {}).get("no_test_retuning") is not True
    ):
        raise ValueError("unregistered supervised study design")
    training = reg["neural_training"]
    required = {
        "seeds": [42, 43, 44],
        "device": "cpu",
        "max_epochs": 20,
        "max_seconds_per_seed": 600,
        "batch_size": 256,
        "class_order": list(LABELS),
    }
    if any(training.get(key) != value for key, value in required.items()):
        raise ValueError("unregistered supervised neural training budget")
    return reg


def _validate_manifest(manifest: dict[str, Any], registration: dict[str, Any]) -> None:
    config = ForecastConfig.model_validate(registration["forecast_config"])
    if (
        manifest.get("schema_version") != "supervised-prepared-v1"
        or manifest.get("prepared_data_id")
        != content_hash({k: v for k, v in manifest.items() if k != "prepared_data_id"})
        or manifest.get("registration_id") != registration["registration_id"]
        or manifest.get("registration") != registration
        or manifest.get("forecast_config") != registration["forecast_config"]
        or manifest.get("config_id") != config.config_id
        or manifest.get("class_order") != list(LABELS)
        or manifest.get("normalizer_id") != content_hash(manifest.get("normalizers"))
        or manifest.get("normalizers", {}).get("fit_role") != "TRAIN"
        or manifest.get("normalizers", {}).get("train_case_ids_hash") != manifest["roles"]["TRAIN"]["case_ids_hash"]
        or manifest.get("planned_session_counts") != {"TRAIN": 60, "TUNE": 10, "CAL": 10, "GATE": 10, "TEST": 20}
    ):
        raise ValueError("supervised prepared data/registration/normalizer identity mismatch")


def _validate_stage(stage: SupervisedStage, role: str, manifest: dict[str, Any]) -> None:
    count = manifest["roles"][role]["count"]
    ids = stage.case_ids
    if (
        stage.role != role
        or len(ids) != count
        or not count
        or len(set(ids)) != count
        or tuple(row.example_id for row in stage.examples) != ids
        or tuple(record.base_example_id for record in stage.features) != ids
        or content_hash(ids) != manifest["roles"][role]["case_ids_hash"]
        or stage.sequence.shape != (count, 60, 6)
        or stage.static.shape != (count, 115)
        or stage.sequence.dtype != np.float32
        or stage.static.dtype != np.float32
        or not np.isfinite(stage.sequence).all()
        or not np.isfinite(stage.static).all()
        or stage.targets.dtype != np.int64
        or stage.targets.shape != (count,)
        or not np.array_equal(stage.targets, _targets(stage.examples))
        or any(
            row.config_id != manifest["config_id"]
            or record.config_id != row.config_id
            or record.symbol != row.symbol
            or record.as_of != row.as_of
            for row, record in zip(stage.examples, stage.features, strict=True)
        )
    ):
        raise ValueError("supervised stage must contain the complete ordered common input/target cohort")


def _data_snapshot(prepared: Path, manifest: dict[str, Any]) -> dict[str, str]:
    artifacts = [manifest["catalog"]]
    artifacts.extend(
        role[name]
        for role in manifest["roles"].values()
        for name in ("sequence", "static", "targets", "examples", "features", "cases")
    )
    result = {"manifest.json": _file_hash(prepared / "manifest.json")}
    for item in artifacts:
        name = item["path"]
        path = prepared / name
        if (
            not isinstance(name, str)
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or not path.resolve().is_relative_to(prepared)
            or _file_hash(path) != item["sha256"]
            or path.stat().st_size != item["bytes"]
        ):
            raise ValueError("supervised prepared input artifact integrity mismatch")
        result[name] = item["sha256"]
    if manifest.get("source_inventory"):
        root = Path(manifest["experiment_root"]).resolve()
        for item in manifest["source_inventory"]:
            name = item["path"]
            path = root / name
            if (
                Path(name).is_absolute()
                or ".." in Path(name).parts
                or not path.resolve().is_relative_to(root)
                or _file_hash(path) != item["sha256"]
                or path.stat().st_size != item["bytes"]
            ):
                raise ValueError("supervised raw source inventory integrity mismatch")
            result[f"source:{name}"] = item["sha256"]
    return result


def _source_snapshot(models: Any) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for name in ("tradecopilot", "numpy", "scikit-learn", "torch", "exchange-calendars", "pydantic"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return {
        "source_hashes": dict(models.source_hashes()),
        "dependencies": versions,
        "python_executable": str(Path(sys.executable).absolute()),
        "python_version": sys.version,
    }


def _clarification(registration_path: Path, registration: dict[str, Any]) -> dict[str, Any] | None:
    path = registration_path.parent / "pretraining-clarification.json"
    if not path.exists():
        return None
    value = _read_sealed(path, "clarification_id")
    if (
        value.get("schema_version") != "supervised-pretraining-clarification-v1"
        or value.get("registration_id") != registration["registration_id"]
        or value.get("comparison_scope") != COMPARISON_SCOPE
        or value.get("before_learning_and_quality_scoring") is not True
    ):
        raise ValueError("pretraining comparison clarification mismatch")
    return value


def _method_registry(
    registration_path: Path, registration: dict[str, Any], clarification: dict[str, Any] | None
) -> dict[str, Any] | None:
    path = registration_path.parent / "method-registry.json"
    if not path.exists():
        return None
    value = _read_sealed(path, "method_registry_id")
    expected_bootstrap = {
        "ordering": "ascending_session_dates",
        "blocks": "consecutive_nonoverlapping_starting_at_first_role_date",
        "block_size_dates": registration["uncertainty"]["block_sensitivity_days"],
        "resample": "same_number_of_blocks_with_replacement",
        "replicates": registration["uncertainty"]["bootstrap_resamples"],
        "seed": registration["uncertainty"]["seed"],
        "weighting": "ratio_of_total_case_sums_to_total_case_count",
        "interval": "percentiles_2.5_97.5",
        "test_block_count": 4,
    }
    if (
        value.get("schema_version") != "supervised-method-registry-v1"
        or value.get("registration_id") != registration["registration_id"]
        or clarification is None
        or value.get("pretraining_clarification_id") != clarification["clarification_id"]
        or value.get("before_learning_and_quality_scoring") is not True
        or value.get("comparison_scope") != COMPARISON_SCOPE
        or value.get("bootstrap") != expected_bootstrap
    ):
        raise ValueError("pretraining method registry identity/scope/bootstrap mismatch")
    return value


def _artifact_identity(
    artifact: dict[str, Any],
    candidate: str,
    registration: dict[str, Any],
    manifest: dict[str, Any],
    source: dict[str, str],
) -> None:
    expected = {
        "candidate_key": candidate,
        "registration_id": registration["registration_id"],
        "prepared_data_id": manifest["prepared_data_id"],
        "config_id": manifest["config_id"],
        "class_order": list(LABELS),
        "normalizer_hash": manifest["normalizer_id"],
        "normalizers": manifest["normalizers"],
        "source_hashes": source,
        "training_ids_hash": manifest["roles"]["TRAIN"]["case_ids_hash"],
    }
    if any(artifact.get(key) != value for key, value in expected.items()):
        raise ValueError("model artifact source/data/class/preprocessing identity mismatch")
    if artifact.get("model_id") != content_hash({k: v for k, v in artifact.items() if k != "model_id"}):
        raise ValueError("model artifact integrity mismatch")


def _seed_predictor(
    models: Any,
    directory: Path,
    result: dict[str, Any],
    candidate: str,
    seed: int,
    registration: dict[str, Any],
    manifest: dict[str, Any],
    source: dict[str, str],
    train: SupervisedStage,
    tune: SupervisedStage,
    prepared_inputs: dict[str, str],
) -> Any:
    child_result = _read_sealed(directory / "result.json", "result_id")
    if {key: value for key, value in result.items() if key != "supervisor"} != child_result:
        raise ValueError("seed result is not the sealed learner result")
    if (
        result.get("prepared_input_hashes_before") != prepared_inputs
        or result.get("prepared_input_hashes_after") != prepared_inputs
    ):
        raise ValueError("seed prepared input hashes differ from the prefit seal")
    checkpoint_epochs = _validate_seed_result(result, candidate, seed, registration, manifest, source)
    checkpoints = result.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != len(checkpoint_epochs):
        raise ValueError("seed requires all registered best-checkpoint artifacts")
    for epoch, row in zip(checkpoint_epochs, checkpoints, strict=True):
        path = directory / f"checkpoint-{epoch:02d}.json"
        if (
            not isinstance(row, dict)
            or row.get("path") != str(path.resolve())
            or row.get("epoch") != epoch
            or _file_hash(path) != row.get("sha256")
            or row.get("tune_nll") != result["epoch_history"][epoch - 1]["tune_nll"]
        ):
            raise ValueError("seed retained checkpoint inventory mismatch")
        checkpoint_artifact = _read_json(path)
        _artifact_identity(checkpoint_artifact, candidate, registration, manifest, source)
        if (
            row.get("model_id") != checkpoint_artifact["model_id"]
            or checkpoint_artifact.get("checkpoint", {}).get("epoch") != epoch
        ):
            raise ValueError("seed retained checkpoint model identity mismatch")
    if (
        checkpoints[-1]["model_id"] != result["checkpoint_hash"]
        or checkpoints[-1]["sha256"] != result["checkpoint_file_sha256"]
    ):
        raise ValueError("seed selected checkpoint is not its last registered best checkpoint")
    checkpoint = directory / "model.json"
    if result.get("model_path") != str(checkpoint.resolve()) or _file_hash(checkpoint) != result.get(
        "checkpoint_file_sha256"
    ):
        raise ValueError("seed checkpoint file identity mismatch")
    predictor = models.load_neural_artifact(checkpoint)
    artifact = predictor.artifact
    _artifact_identity(artifact, candidate, registration, manifest, source)
    architecture = next(item for item in registration["neural_candidates"] if item["key"] == candidate)
    if (
        predictor.model_id != result["checkpoint_hash"]
        or artifact.get("architecture") != architecture
        or artifact.get("seed") != seed
        or artifact.get("training_ids") != list(train.case_ids)
        or artifact.get("tune_ids") != list(tune.case_ids)
        or artifact.get("tune_ids_hash") != manifest["roles"]["TUNE"]["case_ids_hash"]
        or artifact.get("budget") != result["budget"]
        or artifact.get("checkpoint", {}).get("epoch") != result["best_epoch"]
        or artifact.get("checkpoint", {}).get("optimizer_updates")
        != result["best_epoch"] * math.ceil(len(train.case_ids) / 256)
        or artifact.get("checkpoint", {}).get("tune_nll") != result["best_tune_nll"]
    ):
        raise ValueError("seed checkpoint architecture/cohort/budget selection mismatch")
    p = _validate_probability_cohort(tune.case_ids, tune.case_ids, predictor.probabilities(tune.sequence, tune.static))
    if not math.isclose(float(_losses(p, tune.targets).mean()), result["best_tune_nll"], abs_tol=1e-5, rel_tol=1e-6):
        raise ValueError("seed checkpoint does not reproduce its complete TUNE log loss")
    return predictor


def _primary(
    models: Any,
    stage: SupervisedStage,
    candidate: str,
    cpu: dict[str, dict[str, Any]],
    neural: dict[str, dict[int, Any]],
    seeds: tuple[int, ...],
) -> NDArray[np.float64]:
    if candidate in cpu:
        return _validate_probability_cohort(
            stage.case_ids, stage.case_ids, models.predict_cpu(cpu[candidate], stage.features)
        )
    return _family_probabilities(
        stage.case_ids,
        {
            seed: (stage.case_ids, predictor.probabilities(stage.sequence, stage.static))
            for seed, predictor in neural[candidate].items()
        },
        seeds,
    )


def _inventory(directory: Path) -> dict[str, str]:
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("symlinked study bundle artifact")
        if path.is_file() and path != directory / "report.json":
            result[str(path.relative_to(directory))] = _file_hash(path)
    return result


def run_supervised_study(prepared: Path, registration_path: Path, output_dir: Path) -> Path:
    """Fit registered arms once, freeze staged selection, then score the cold common TEST cohort."""
    prepared, registration_path, output_dir = prepared.resolve(), registration_path.resolve(), output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("choose a new immutable supervised study directory")
    reg = _registration(registration_path)
    clarification = _clarification(registration_path, reg)
    registry = _method_registry(registration_path, reg, clarification)
    train, manifest = load_stage(prepared, "TRAIN")
    tune, tune_manifest = load_stage(prepared, "TUNE")
    _validate_manifest(manifest, reg)
    if manifest.get("method_registry") != registry or manifest.get("method_registry_id") != (
        registry["method_registry_id"] if registry else None
    ):
        raise ValueError("prepared data method registry identity mismatch")
    if tune_manifest != manifest:
        raise ValueError("TRAIN/TUNE data identity mismatch")
    _validate_stage(train, "TRAIN", manifest)
    _validate_stage(tune, "TUNE", manifest)
    models = _model_api()
    source = _source_snapshot(models)
    data = _data_snapshot(prepared, manifest)
    prepared_inputs = models.prepared_input_hashes(prepared, registration_path)
    registration_sha = _file_hash(registration_path)
    clarification_sha = (
        _file_hash(registration_path.parent / "pretraining-clarification.json") if clarification else None
    )
    registry_sha = _file_hash(registration_path.parent / "method-registry.json") if registry else None
    output_dir.mkdir(parents=True, mode=0o700)
    output_dir.chmod(0o700)
    for name, value in (("registration.json", reg), ("prepared-data.json", manifest), ("source.json", source)):
        _write_json(output_dir / name, value)
    if clarification:
        _write_json(output_dir / "pretraining-clarification.json", clarification)
    if registry:
        _write_json(output_dir / "method-registry.json", registry)
    prefit = _seal(
        output_dir / "prefit.json",
        {
            "stage": "registered_before_any_fit",
            "registration_id": reg["registration_id"],
            "prepared_data_id": manifest["prepared_data_id"],
            "source_provenance": source,
            "prepared_input_hashes": data,
            "registration_file_sha256": registration_sha,
            "model_prepared_input_hashes": prepared_inputs,
            "clarification_id": clarification["clarification_id"] if clarification else None,
            "clarification_file_sha256": clarification_sha,
            "comparison_scope": COMPARISON_SCOPE,
            "method_registry_id": registry["method_registry_id"] if registry else None,
            "method_registry_file_sha256": registry_sha,
        },
        "prefit_id",
    )
    progress = output_dir / "progress.json"

    def verify_inputs() -> None:
        if (
            _source_snapshot(models) != source
            or _data_snapshot(prepared, manifest) != data
            or models.prepared_input_hashes(prepared, registration_path) != prepared_inputs
            or _file_hash(registration_path) != registration_sha
            or (
                clarification
                and _file_hash(registration_path.parent / "pretraining-clarification.json") != clarification_sha
            )
            or (registry and _file_hash(registration_path.parent / "method-registry.json") != registry_sha)
        ):
            raise ValueError("registered source/data/config changed during supervised study")

    def next_stage(role: str, selection_path: Path | None = None) -> SupervisedStage:
        stage, current = load_stage(prepared, role, selection_path=selection_path)
        if current != manifest:
            raise ValueError("supervised stage changed prepared data identity")
        _validate_stage(stage, role, manifest)
        return stage

    try:
        cpu: dict[str, dict[str, Any]] = {}
        neural: dict[str, dict[int, Any]] = {}
        candidates: dict[str, dict[str, Any]] = {}
        cpu_runs: list[dict[str, Any]] = []
        runs: list[dict[str, Any]] = []
        seeds = tuple(reg["neural_training"]["seeds"])
        cpu_dir, neural_dir = output_dir / "cpu", output_dir / "neural"
        cpu_dir.mkdir(mode=0o700)
        neural_dir.mkdir(mode=0o700)
        for candidate in reg["cpu_candidates"]:
            key = candidate["key"]
            status: dict[str, Any] = {"candidate_key": key, "eligible": False}
            try:
                artifact, native = models.fit_cpu(key, train, manifest, reg)
                path = cpu_dir / f"{key}.json"
                models.save_cpu_artifact(artifact, path)
                artifact = models.load_cpu_artifact(path)
                _artifact_identity(artifact, key, reg, manifest, source["source_hashes"])
                if artifact.get("training_ids") != list(train.case_ids):
                    raise ValueError("CPU arm did not use every ordered TRAIN case")
                p = _validate_probability_cohort(
                    tune.case_ids, tune.case_ids, models.predict_cpu(artifact, tune.features)
                )
                if native is not None:
                    parity = models.native_cpu_probabilities(artifact, native, tune.features)
                    _validate_probability_cohort(tune.case_ids, tune.case_ids, parity)
                    if not np.allclose(p, parity, atol=1e-10, rtol=0):
                        raise ValueError("CPU native/portable TUNE prediction parity failed")
                cpu[key] = artifact
                status.update(
                    eligible=True,
                    status="complete",
                    model_id=artifact["model_id"],
                    native_portable_parity="passed" if native is not None else "prior_has_no_native_estimator",
                )
            except (ValueError, OSError, TypeError, KeyError, RuntimeError) as exc:
                status.update(status="failed", reason=str(exc))
            cpu_runs.append(status)
            candidates[key] = {
                "eligible": status["eligible"],
                "representation": "TRAIN class prior"
                if key == "prior"
                else "55 raw causal OHLCV features plus 5 symbol indicators",
                "input_dimensions": {} if key == "prior" else {"static": 60},
                "artifact_ids": [cpu[key]["model_id"]] if key in cpu else [],
                "failure_reason": status.get("reason"),
            }
            _write_json(output_dir / "training.json", {"cpu_arms": cpu_runs, "neural_seeds": runs})
        for candidate in reg["neural_candidates"]:
            key = candidate["key"]
            neural[key] = {}
            failures = []
            for seed in seeds:
                directory = neural_dir / f"{key}-{seed}"
                try:
                    result = _run_seed_worker(prepared, registration_path, key, seed, directory, reg, progress)
                except (ValueError, OSError, subprocess.SubprocessError) as exc:
                    result = {
                        "candidate_key": key,
                        "seed": seed,
                        "status": "worker_launch_failed",
                        "eligible": False,
                        "reason": str(exc),
                    }
                validation: dict[str, Any] = {"eligible": False}
                try:
                    predictor = _seed_predictor(
                        models,
                        directory,
                        result,
                        key,
                        seed,
                        reg,
                        manifest,
                        source["source_hashes"],
                        train,
                        tune,
                        prepared_inputs,
                    )
                    neural[key][seed] = predictor
                    validation["eligible"] = True
                except (ValueError, OSError, TypeError, KeyError) as exc:
                    validation["reason"] = str(exc)
                    failures.append({"seed": seed, "reason": str(exc), "status": result.get("status")})
                runs.append({**result, "validation": validation})
                _write_json(output_dir / "training.json", {"cpu_arms": cpu_runs, "neural_seeds": runs})
            candidates[key] = {
                "eligible": set(neural[key]) == set(seeds),
                "representation": "60x6 normalized causal minute sequence plus 115 static inputs",
                "input_dimensions": {"sequence": [60, 6], "static": 115},
                "parameter_count_per_seed": next(
                    (p.artifact.get("parameter_count") for p in neural[key].values()), None
                ),
                "artifact_ids": [neural[key][seed].model_id for seed in seeds if seed in neural[key]],
                "seed_ids": list(seeds),
                "failed_seeds": failures,
                "primary_distribution": "arithmetic_mean_of_all_three_registered_seed_probabilities",
            }
        verify_inputs()
        order = [candidate["key"] for candidate in reg["cpu_candidates"] + reg["neural_candidates"]]
        eligible = [key for key in order if candidates[key]["eligible"]]
        if "prior" not in cpu or not eligible:
            raise ValueError("complete prior and at least one registered CPU reference required before TEST")
        tune_p = {key: _primary(models, tune, key, cpu, neural, seeds) for key in eligible}
        for key, p in tune_p.items():
            candidates[key]["tune_log_loss"] = float(_losses(p, tune.targets).mean())
        chosen = min(eligible, key=lambda key: candidates[key]["tune_log_loss"])
        reference = min([key for key in order if key in cpu], key=lambda key: candidates[key]["tune_log_loss"])
        selected_paths = (
            [cpu_dir / f"{chosen}.json"]
            if chosen in cpu
            else [neural_dir / f"{chosen}-{seed}" / "model.json" for seed in seeds]
        )
        selected_ids = candidates[chosen]["artifact_ids"]
        weights = _seal(
            output_dir / "frozen-weights.json",
            {
                "stage": "weights_frozen_before_CAL",
                "registration_id": reg["registration_id"],
                "prepared_data_id": manifest["prepared_data_id"],
                "prefit_id": prefit["prefit_id"],
                "selected_candidate_id": chosen,
                "selected_artifact_ids": selected_ids,
                "selected_checkpoint_ids": selected_ids if chosen not in cpu else [],
                "selected_models": [
                    {"path": str(path.relative_to(output_dir)), "sha256": _file_hash(path), "model_id": identity}
                    for path, identity in zip(selected_paths, selected_ids, strict=True)
                ],
                "normalizer_id": manifest["normalizer_id"],
                "class_order": list(LABELS),
                "cpu_reference_candidate_id": reference,
                "cpu_reference_artifact_id": cpu[reference]["model_id"],
                "source_data_id": manifest["source_data_id"],
                "config_id": manifest["config_id"],
                "source_provenance_id": content_hash(source),
                "comparison_scope": COMPARISON_SCOPE,
                "clarification_id": clarification["clarification_id"] if clarification else None,
                "method_registry_id": registry["method_registry_id"] if registry else None,
                "weights_refit_on_later_roles": False,
            },
            "weights_id",
        )
        _write_json(progress, {"stage": "weights_frozen", "weights_id": weights["weights_id"]})
        cal = next_stage("CAL")
        cal_p = _primary(models, cal, chosen, cpu, neural, seeds)
        temperature = fit_temperature(cal_p, cal.targets)
        calibration = _seal(
            output_dir / "calibration.json",
            {
                "stage": "temperature_frozen_before_GATE",
                "weights_id": weights["weights_id"],
                "role": "CAL",
                "case_ids_hash": manifest["roles"]["CAL"]["case_ids_hash"],
                "temperature": temperature,
                "grid": {"count": 81, "minimum": 0.25, "maximum": 4},
                "raw_log_loss": float(_losses(cal_p, cal.targets).mean()),
                "calibrated_log_loss": float(_losses(apply_temperature(cal_p, temperature), cal.targets).mean()),
            },
            "calibration_id",
        )
        gate_stage = next_stage("GATE")
        gate_p = apply_temperature(_primary(models, gate_stage, chosen, cpu, neural, seeds), temperature)
        gate = select_gate(gate_stage.examples, gate_p)
        gate["metrics"] = _metrics(gate_stage.examples, gate_p, gate["threshold"])
        gate["calibration_id"] = calibration["calibration_id"]
        gate = _seal(output_dir / "gate.json", gate, "gate_id")
        verify_inputs()
        selection = _seal(
            output_dir / "frozen-selection.json",
            {
                **{key: value for key, value in weights.items() if key != "stage"},
                "stage": "selection_frozen",
                "temperature": temperature,
                "calibration_id": calibration["calibration_id"],
                "gate": gate,
                "primary_unit": "single_CPU_model" if chosen in cpu else "all_three_seed_arithmetic_probability_mean",
                "gate_case_ids_hash": manifest["roles"]["GATE"]["case_ids_hash"],
            },
            "selection_id",
        )
        _write_json(progress, {"stage": "selection_frozen_before_TEST", "selection_id": selection["selection_id"]})
        test = next_stage("TEST", output_dir / "frozen-selection.json")
        test_p = {key: _primary(models, test, key, cpu, neural, seeds) for key in eligible}
        diagnostics = {
            f"{key}:{seed}": _validate_probability_cohort(
                test.case_ids, test.case_ids, predictor.probabilities(test.sequence, test.static)
            )
            for key, predictors in neural.items()
            for seed, predictor in predictors.items()
        }
        raw, calibrated = test_p[chosen], apply_temperature(test_p[chosen], temperature)
        if not np.array_equal(raw.argmax(axis=1), calibrated.argmax(axis=1)):
            raise ValueError("scalar temperature unexpectedly changed TEST argmax predictions")

        def score(p: NDArray[np.float64], threshold: float | None) -> dict[str, Any]:
            return {
                **_metrics(test.examples, p, threshold),
                "uncertainty": _uncertainty(test.examples, p, threshold, reg["uncertainty"]),
            }

        primary_metrics = {key: score(p, 0.0) for key, p in test_p.items()}
        seed_metrics = {key: score(p, 0.0) for key, p in diagnostics.items()}
        raw_metrics, calibrated_metrics = score(raw, gate["threshold"]), score(calibrated, gate["threshold"])
        paired = _paired_uncertainty(test.examples, test_p[reference], raw, reg["uncertainty"])
        paired.update(comparison_scope=COMPARISON_SCOPE, selected_candidate_id=chosen, reference_candidate_id=reference)
        groups = {**test_p, **diagnostics, "selected_calibrated": calibrated}
        with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for key, p in groups.items():
                for identity, probabilities in zip(test.case_ids, p, strict=True):
                    handle.write(
                        json.dumps(
                            {
                                "candidate_id": key,
                                "case_id": identity,
                                "class_order": list(LABELS),
                                "probabilities": probabilities.tolist(),
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
        (output_dir / "examples.jsonl").write_text(
            "".join(row.model_dump_json() + "\n" for row in test.examples), encoding="utf-8"
        )
        verify_inputs()
        _write_json(progress, {"stage": "complete", "selection_id": selection["selection_id"]})
        for path in output_dir.rglob("*"):
            path.chmod(0o700 if path.is_dir() else 0o600)
        inventory = _inventory(output_dir)
        report = {
            "schema_version": REPORT_VERSION,
            "experiment_type": "DIRECT_SUPERVISED_FORECAST",
            "evidence_mode": "historical_project_holdout",
            "evaluation_kind": "historical_project_holdout"
            if all(row.provenance == "historical" for row in train.examples)
            and not (
                manifest["source_manifest"].get("source_metadata", {}).get("fixture")
                or manifest["source_manifest"].get("fixture")
            )
            else "synthetic_control_flow",
            "registration_id": reg["registration_id"],
            "prepared_data_id": manifest["prepared_data_id"],
            "source_data_id": manifest["source_data_id"],
            "config_id": manifest["config_id"],
            "normalizer_id": manifest["normalizer_id"],
            "prefit_id": prefit["prefit_id"],
            "clarification_id": clarification["clarification_id"] if clarification else None,
            "method_registry_id": registry["method_registry_id"] if registry else None,
            "comparison_scope": COMPARISON_SCOPE,
            "source_provenance": source,
            "candidates": candidates,
            "training": {"cpu_arms": cpu_runs, "neural_seeds": runs},
            "selection": selection,
            "tune": {
                "candidate_primaries": {key: _summary(tune.examples, p) for key, p in tune_p.items()},
                "cpu_reference_candidate_id": reference,
            },
            "calibration": calibration,
            "gate": gate,
            "test": {
                "candidate_primaries": primary_metrics,
                "individual_seed_diagnostics": seed_metrics,
                "selected_raw": raw_metrics,
                "selected_calibrated": calibrated_metrics,
                "prior": primary_metrics["prior"],
                "cpu_reference": primary_metrics[reference],
                "paired_reference_improvement": paired,
            },
            "flags": {
                "observed_selective_target_met": gate["enabled"]
                and gate["metrics"]["target_requirements_met"]
                and calibrated_metrics["target_requirements_met"],
                "improvement_supported": raw_metrics["log_loss"] < primary_metrics[reference]["log_loss"]
                and paired["date_cluster"]["low"] > 0,
                "automatic_production_promotion": False,
            },
            "split": {
                role: {
                    "planned_sessions": manifest["planned_session_counts"][role],
                    "eligible_cases": value["count"],
                    "case_ids_hash": value["case_ids_hash"],
                }
                for role, value in manifest["roles"].items()
            },
            "planned_anchor_counts": manifest.get("planned_anchor_counts", {}),
            "exclusions": manifest.get("exclusions", {}),
            "catalog_counts": manifest.get("catalog_counts", {}),
            "constraints": {
                "new_jev_calls": 0,
                "broker_orders": 0,
                "no_test_retuning": True,
                "default_model_changed": False,
            },
            "corpus_status": "retired_after_outcome_scoring",
            "jev_improvement_claim": False,
            "limitations": [
                "Historical project holdout after earlier related studies; not prospective deployment confirmation.",
                "All-case accuracy and proper losses remain primary context for selective accuracy and coverage.",
                "Outcome windows overlap; uncertainty resamples whole dates and reports five-date block sensitivity.",
                "Directional precision does not measure profitable broker execution.",
                "The selection and CPU reference remain fixed even if another TEST diagnostic ranks higher.",
                "No automatic model promotion and no Jev-model improvement claim.",
            ],
            "inventory": inventory,
            "inventory_id": content_hash(inventory),
        }
        _seal(output_dir / "report.json", report, "report_id")
        return output_dir / "report.json"
    except Exception as exc:
        _seal(
            output_dir / "failure.json",
            {"stage": "failed", "error": str(exc), "error_type": type(exc).__name__, "prefit_id": prefit["prefit_id"]},
            "failure_id",
        )
        _write_json(progress, {"stage": "failed", "error": str(exc)})
        raise


def load_report(path: Path) -> dict[str, Any]:
    """Verify the sealed report, full private inventory, and frozen selection before display."""
    path = path.resolve()
    if path.is_dir():
        path /= "report.json"
    try:
        report = _read_sealed(path, "report_id")
        inventory = report["inventory"]
        required = {
            "registration.json",
            "prepared-data.json",
            "source.json",
            "prefit.json",
            "frozen-weights.json",
            "calibration.json",
            "gate.json",
            "frozen-selection.json",
            "predictions.jsonl",
            "examples.jsonl",
        }
        if (
            report.get("schema_version") != REPORT_VERSION
            or not isinstance(inventory, dict)
            or not required <= inventory.keys()
            or report.get("inventory_id") != content_hash(inventory)
        ):
            raise ValueError("invalid supervised report")
        for name, digest in inventory.items():
            artifact = path.parent / name
            if (
                Path(name).is_absolute()
                or ".." in Path(name).parts
                or not artifact.resolve().is_relative_to(path.parent)
                or _file_hash(artifact) != digest
            ):
                raise ValueError("invalid supervised report artifact")
        if _inventory(path.parent) != inventory:
            raise ValueError("supervised report has undeclared or changed artifacts")
        selection = _read_sealed(path.parent / "frozen-selection.json", "selection_id")
        prefit = _read_sealed(path.parent / "prefit.json", "prefit_id")
        if (
            selection != report["selection"]
            or prefit["prefit_id"] != report["prefit_id"]
            or selection["registration_id"] != report["registration_id"]
            or selection["prepared_data_id"] != report["prepared_data_id"]
        ):
            raise ValueError("supervised report frozen selection mismatch")
    except (ValueError, OSError, TypeError, KeyError):
        raise ValueError("supervised report integrity verification failed") from None
    return report
