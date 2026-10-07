"""Registered, offline capacity comparison with sealed tuning selection and cold test."""

from __future__ import annotations

import importlib
import json
import math
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import replace
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import Any

from tradecopilot.forecast.contracts import content_hash

from . import training
from .data import load_prepared
from .evaluation import aggregate_episodes, paired_interval, replay_policy, select_architecture
from .policies import FixedPolicy, NeuralPolicy


def _write(path: Path, value: dict[str, Any]) -> None:
    training._atomic_json(path, value)


def _seal(path: Path, value: dict[str, Any], key: str) -> dict[str, Any]:
    result = {**value, key: content_hash(value)}
    _write(path, result)
    return result


def _validate_budget(budget: dict[str, Any], reg: dict[str, Any]) -> None:
    if budget.get("registration_id") != reg["registration_id"]:
        raise ValueError("budget registration identity mismatch")
    steps, seconds = budget.get("shared_steps"), budget.get("max_seconds")
    if (
        not isinstance(steps, int)
        or isinstance(steps, bool)
        or not 10240 <= steps <= 100000
        or steps % 128
        or not isinstance(seconds, (float, int))
        or isinstance(seconds, bool)
        or not math.isfinite(seconds)
        or not 0 < seconds <= 600
        or budget.get("device") not in ("cpu", "mps")
    ):
        raise ValueError("invalid training budget caps")


def _lock_hash() -> str | None:
    lock = Path(__file__).resolve().parents[3] / "uv.lock"
    return sha256(lock.read_bytes()).hexdigest() if lock.exists() else None


def _execution_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    return {
        "python": str(Path(sys.executable).absolute()),
        "scripts": {
            name: sha256((root / "scripts" / name).read_bytes()).hexdigest()
            for name in ("rl_train_worker.py", "run_rl_capacity_study.py")
        },
    }


def _verify_checkpoint(directory: Path, result: dict[str, Any]) -> Path:
    path = directory / "model.zip"
    if path.is_symlink() or not path.is_file() or sha256(path.read_bytes()).hexdigest() != result["checkpoint_sha256"]:
        raise ValueError("checkpoint integrity mismatch")
    return path


def _run_worker(
    prepared: Path,
    registration: Path,
    budget_path: Path,
    candidate: str,
    seed: int,
    output: Path,
    budget: dict[str, Any],
    progress: Path,
) -> dict[str, Any]:
    worker = Path(__file__).resolve().parents[3] / "scripts" / "rl_train_worker.py"
    command = [
        str(Path(sys.executable).absolute()),
        str(worker),
        "--prepared",
        str(prepared),
        "--registration",
        str(registration),
        "--budget",
        str(budget_path),
        "--candidate",
        candidate,
        "--seed",
        str(seed),
        "--output-dir",
        str(output),
    ]
    started = time.monotonic()
    with (output.parent / f"{output.name}.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            child = output / "progress.json"
            status = {"candidate": candidate, "seed": seed, "stage": "training"}
            if child.exists():
                with suppress(ValueError, OSError):
                    status.update(json.loads(child.read_text()))
            _write(progress, status)
            if time.monotonic() - started > budget["max_seconds"] + 60:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                return {"candidate_key": candidate, "seed": seed, "status": "timeout"}
            time.sleep(0.2)
    if process.returncode != 0:
        return {"candidate_key": candidate, "seed": seed, "status": "failed", "returncode": process.returncode}
    return training._read_sealed(output / "result.json", "training_id")


def _verify_training(
    runs: list[dict[str, Any]], reg: dict[str, Any], budget: dict[str, Any], source: dict[str, str], lock: str | None
) -> None:
    expected = {(c["key"], s) for c in reg.get("candidates", []) for s in reg.get("seeds", [])}
    actual = [(r.get("candidate_key"), r.get("seed")) for r in runs]
    if len(runs) != 9 or len(set(actual)) != 9 or set(actual) != expected:
        raise ValueError("all nine complete registered runs required")
    config_hash = training.registered_config(reg).content_hash
    versions = runs[0].get("versions")
    for run in runs:
        identity = {
            "registration_id": reg["registration_id"],
            "prepared_data_id": budget["prepared_data_id"],
            "budget_id": budget["budget_id"],
            "source_provenance": source,
            "lock_sha256": lock,
            "config_hash": config_hash,
            "versions": versions,
            "candidate": training._CANDIDATES[run["candidate_key"]],
        }
        if any(run.get(key) != value for key, value in identity.items()):
            raise ValueError("training source/data/budget identity mismatch")
        if (
            run.get("status") != "complete"
            or run.get("actual_timesteps") != budget["shared_steps"]
            or run.get("n_updates") != budget["shared_steps"] // 128 * 10
        ):
            raise ValueError("all nine complete equal-budget training runs required")


def _neural(directory: Path, run: dict[str, Any], device: str) -> NeuralPolicy:
    path = _verify_checkpoint(directory, run)
    recurrent = run["candidate"]["algorithm"] == "RecurrentPPO"
    module = importlib.import_module("sb3_contrib" if recurrent else "stable_baselines3")
    cls = module.RecurrentPPO if recurrent else module.PPO
    return NeuralPolicy(cls.load(str(path), device=device), recurrent=recurrent)


def _controls(manifest: dict[str, Any], reg: dict[str, Any]) -> dict[str, FixedPolicy]:
    normalizer = manifest["normalizer"]
    name = "return_5m_bps"
    feature_index = manifest["feature_names"].index(name)
    normalizer_index = normalizer["feature_names"].index(name)
    return {
        "cash": FixedPolicy("cash"),
        "hold": FixedPolicy("hold"),
        "rule": FixedPolicy(
            "rule",
            feature_index=feature_index,
            missing_index=manifest["feature_names"].index(f"missing_{name}"),
            mean=normalizer["mean"][normalizer_index],
            scale=normalizer["scale"][normalizer_index],
            threshold=reg["rule"]["enter_when_return_5m_bps_above"],
        ),
    }


def _score(episodes: list[Any], config: Any, policy: Any, output: Path) -> dict[str, Any]:
    rows = [replay_policy(episode, config, policy, keep_ledger=True) for episode in episodes]
    _write(output, {"episodes": rows})
    return aggregate_episodes(rows)


def _improvement(
    primary: dict[str, Any],
    controls: dict[str, Any],
    stress: dict[str, Any],
    all_metrics: list[dict[str, Any]],
    limit: float,
) -> dict[str, Any]:
    valid = bool(primary.get("valid")) and all(m.get("valid") for m in all_metrics)
    if not valid:
        return {"passed": False, "reason": "invalid or unresolved scored episodes", "intervals": {}}
    intervals = {
        key: paired_interval(primary["daily_returns"], value["daily_returns"]) for key, value in controls.items()
    }
    checks = {
        "positive_primary": primary["mean_daily_return"] > 0,
        "paired_lower_bounds": all(value["low"] > 0 for value in intervals.values()),
        "drawdown_limit": all(m["max_episode_drawdown"] <= limit for m in all_metrics),
        "positive_doubled_cost_increment": stress["neural"]["mean_daily_return"]
        > stress["control"]["mean_daily_return"],
    }
    return {"passed": all(checks.values()), "checks": checks, "intervals": intervals}


def run_study(prepared: Path, registration: Path, budget: Path, output_dir: Path) -> Path:
    """Run only local registered workers; freeze selection before any locked test load."""
    if output_dir.exists():
        raise ValueError("study output is immutable; choose a fresh directory")
    prepared, registration, budget, output_dir = [p.resolve() for p in (prepared, registration, budget, output_dir)]
    reg = training._read_sealed(registration, "registration_id")
    allocation = training._read_sealed(budget, "budget_id")
    _validate_budget(allocation, reg)
    if (
        reg["seeds"] != [42, 43, 44]
        or reg["candidates"] != list(training._CANDIDATES.values())
        or reg["optimizer"] != training.OPTIMIZER
    ):
        raise ValueError("unregistered architecture/seed/optimizer comparison")
    # Loading TRAIN verifies the entire prepared identity without releasing TEST outcomes.
    _, manifest = load_prepared(prepared, role="train")
    if manifest["registration_id"] != reg["registration_id"] or manifest["data_id"] != allocation["prepared_data_id"]:
        raise ValueError("prepared data identity mismatch")
    source, lock = training._source_provenance(), _lock_hash()
    execution = _execution_provenance()
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    _seal(
        output_dir / "study-seal.json",
        {
            "registration": reg,
            "budget": allocation,
            "manifest": manifest,
            "source_provenance": source,
            "lock_sha256": lock,
            "execution_provenance": execution,
        },
        "study_id",
    )
    runs, directories = [], {}
    for candidate in reg["candidates"]:
        for seed in reg["seeds"]:
            directory = output_dir / f"{candidate['key']}-{seed}"
            result = _run_worker(
                prepared,
                registration,
                budget,
                candidate["key"],
                seed,
                directory,
                allocation,
                output_dir / "progress.json",
            )
            runs.append(result)
            directories[(candidate["key"], seed)] = directory
    _write(output_dir / "training.json", {"runs": runs})
    _verify_training(runs, reg, allocation, source, lock)
    for run in runs:
        _verify_checkpoint(directories[(run["candidate_key"], run["seed"])], run)
    if source != training._source_provenance() or lock != _lock_hash() or execution != _execution_provenance():
        raise ValueError("source changed before rating")
    episodes, current = load_prepared(prepared, role="tune")
    if current != manifest:
        raise ValueError("prepared data changed before tuning")
    config = training.registered_config(reg)
    tune = []
    for run in runs:
        key, seed = run["candidate_key"], run["seed"]
        metrics = _score(
            episodes,
            config,
            _neural(directories[(key, seed)], run, allocation["device"]),
            output_dir / f"tune-{key}-{seed}.json",
        )
        tune.append(
            {
                "candidate": key,
                "seed": seed,
                "training_status": run["status"],
                "metrics": metrics,
                "training_id": run["training_id"],
                "checkpoint_sha256": run["checkpoint_sha256"],
            }
        )
    controls = _controls(manifest, reg)
    tune_controls = {
        key: _score(episodes, config, policy, output_dir / f"tune-{key}.json") for key, policy in controls.items()
    }
    chosen = select_architecture(tune, tuple(reg["seeds"]), tuple(c["key"] for c in reg["candidates"]))
    if not all(m["valid"] for m in tune_controls.values()):
        raise ValueError("invalid tuning controls")
    strongest = max(controls, key=lambda key: (tune_controls[key]["mean_daily_return"], -list(controls).index(key)))
    selected_run = next(r for r in runs if (r["candidate_key"], r["seed"]) == (chosen["candidate"], chosen["seed"]))
    selection_path = output_dir / "selection.json"
    selection = _seal(
        selection_path,
        {
            **chosen,
            "stage": "selection_frozen",
            "registration_id": reg["registration_id"],
            "prepared_data_id": manifest["data_id"],
            "budget_id": allocation["budget_id"],
            "checkpoint_sha256": selected_run["checkpoint_sha256"],
            "strongest_control": strongest,
            "training_ids": [r["training_id"] for r in runs],
            "checkpoint_hashes": {f"{r['candidate_key']}-{r['seed']}": r["checkpoint_sha256"] for r in runs},
        },
        "selection_id",
    )
    episodes, current = load_prepared(prepared, role="test", selection_path=selection_path)
    if (
        current != manifest
        or source != training._source_provenance()
        or lock != _lock_hash()
        or execution != _execution_provenance()
    ):
        raise ValueError("source/data changed before locked test")
    test = {}
    for run in runs:
        if run["candidate_key"] == chosen["candidate"]:
            seed = run["seed"]
            test[str(seed)] = _score(
                episodes,
                config,
                _neural(directories[(chosen["candidate"], seed)], run, allocation["device"]),
                output_dir / f"test-neural-{seed}.json",
            )
    test_controls = {
        key: _score(episodes, config, policy, output_dir / f"test-{key}.json") for key, policy in controls.items()
    }
    stress = {}
    for cost in (4, 8):
        stressed = replace(config, cost_bps=Decimal(cost))
        stress[str(cost)] = {
            "neural": _score(
                episodes,
                stressed,
                _neural(directories[(chosen["candidate"], chosen["seed"])], selected_run, allocation["device"]),
                output_dir / f"test-stress-{cost}-neural.json",
            ),
            "control": _score(
                episodes, stressed, controls[strongest], output_dir / f"test-stress-{cost}-{strongest}.json"
            ),
        }
    primary = test[str(chosen["seed"])]
    all_metrics = (
        [r["metrics"] for r in tune] + list(tune_controls.values()) + list(test.values()) + list(test_controls.values())
    )
    all_metrics += [m for item in stress.values() for m in item.values()]
    gate = _improvement(
        primary,
        {"cash": test_controls["cash"], strongest: test_controls[strongest]},
        stress["4"],
        all_metrics,
        float(config.daily_loss) / float(config.initial_cash),
    )
    _, final_manifest = load_prepared(prepared, role="train")
    if (
        final_manifest != manifest
        or source != training._source_provenance()
        or lock != _lock_hash()
        or execution != _execution_provenance()
    ):
        raise ValueError("source/data changed during rating")
    _write(output_dir / "progress.json", {"stage": "complete", "selection_id": selection["selection_id"]})
    inventory = {
        str(p.relative_to(output_dir)): sha256(p.read_bytes()).hexdigest()
        for p in sorted(output_dir.rglob("*"))
        if p.is_file()
    }
    report = {
        "schema_version": "rl-capacity-report-v1",
        "scope": "MARKET_ONLY",
        "evidence_mode": "retrospective_chronological",
        "status": "improved_among_tested" if gate["passed"] else "inconclusive_not_promoted",
        "selection": selection,
        "training": runs,
        "tune": tune,
        "tune_controls": tune_controls,
        "test_selected_architecture_seeds": test,
        "test_controls": test_controls,
        "cost_stress": stress,
        "improvement_gate": gate,
        "source_provenance": source,
        "lock_sha256": lock,
        "execution_provenance": execution,
        "registration_id": reg["registration_id"],
        "prepared_data_id": manifest["data_id"],
        "budget_id": allocation["budget_id"],
        "seed_spread": max(m["mean_daily_return"] for m in test.values())
        - min(m["mean_daily_return"] for m in test.values())
        if all(m["valid"] for m in test.values())
        else None,
        "constraints": {"new_jev_calls": 0, "broker_orders": 0},
        "inventory": inventory,
        "interpretation": (
            "Independent daily capital resets across five symbol books; no annualized operational profit claim"
        ),
    }
    _seal(output_dir / "report.json", report, "report_id")
    return output_dir / "report.json"


def load_report(path: Path) -> dict[str, Any]:
    """Validate the sealed RL report and every private artifact without importing policies."""
    report = training._read_sealed(path, "report_id")
    if report.get("schema_version") != "rl-capacity-report-v1":
        raise ValueError("unsupported RL report schema")
    root = path.parent.resolve()
    for name, expected in report["inventory"].items():
        artifact = root / name
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or artifact.is_symlink()
            or not artifact.is_file()
            or not artifact.resolve().is_relative_to(root)
            or sha256(artifact.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("RL report artifact integrity mismatch")
    return report
