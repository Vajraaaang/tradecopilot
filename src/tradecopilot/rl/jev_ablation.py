"""Matched recurrent policy ablation over an immutable retrospective Jev cache."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import replace
from datetime import datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from tradecopilot.forecast.contracts import content_hash
from tradecopilot.forecast.ohlcv_features import OHLCV_FEATURE_NAMES

from . import study, training
from .contracts import EpisodeData
from .data import load_prepared
from .evaluation import paired_interval

PROFILES = ("CONTEXT_ONLY", "JEV_ASSISTED")
HORIZONS = ("15m", "60m", "close")
CLASSES = ("DOWN", "FLAT", "UP")
FORECAST_NAMES = tuple(
    f"jev_{horizon}_{name}"
    for horizon in HORIZONS
    for name in ("p_down", "p_flat", "p_up", "confidence", "age", "time_to_target", "anchor_return", "available")
)
_write = training._atomic_json
_seal = study._seal
_lock_hash = study._lock_hash
_score = study._score
_neural = study._neural
_verify_checkpoint = study._verify_checkpoint


def _verify_hash(value: dict[str, Any], key: str) -> None:
    if value.get(key) != content_hash({k: v for k, v in value.items() if k != key}):
        raise ValueError(f"invalid {key}")


def _load_cache(path: Path) -> dict[str, Any]:
    from tradecopilot.forecast.jev_rl import load_cache

    return load_cache(path)


def _registration(reg: dict[str, Any]) -> None:
    _verify_hash(reg, "registration_id")
    if (
        reg.get("profiles") != list(PROFILES)
        or reg.get("seeds") != [42, 43, 44]
        or reg.get("candidates") != [training._CANDIDATES["recurrent-256"]]
        or reg.get("optimizer") != training.OPTIMIZER
        or reg["jev"].get("horizons") != list(HORIZONS)
        or reg["jev"].get("as_of_minutes_after_open") != 61
        or reg["jev"].get("simulated_latency_seconds") != 60
    ):
        raise ValueError("unregistered profile/architecture/seed/optimizer/timing comparison")
    training.registered_config(reg)


def _validate_cache(
    cache: dict[str, Any], manifest: dict[str, Any], reg: dict[str, Any]
) -> dict[tuple[str, str, str], dict[str, Any]]:
    _registration(reg)
    _verify_hash(manifest, "data_id")
    _verify_hash(cache, "cache_id")
    expected = {
        "schema_version": "jev-rl-cache-v1",
        "registration_id": reg["registration_id"],
        "source_data_id": manifest["source_data_id"],
        "base_prepared_data_id": manifest["data_id"],
        "model": reg["jev"]["model"],
        "prompt_version": reg["jev"]["prompt_version"],
        "availability_mode": "hypothetical_as_of_plus_60_seconds",
    }
    if manifest.get("registration_id") != reg["registration_id"] or any(cache.get(k) != v for k, v in expected.items()):
        raise ValueError("cache/source/prepared/registration identity mismatch")
    pairs = {(r["symbol"], r["session_date"]) for r in manifest["episodes"]}
    required = {(symbol, day, h) for symbol, day in pairs for h in HORIZONS}
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    date_asofs: dict[str, int] = {}
    for record in cache["records"]:
        key = (record["symbol"], record["session_date"], record["horizon_key"])
        if key not in required or key in result:
            raise ValueError("cache duplicate or unregistered forecast scope")
        as_of, available, target = (record[k] for k in ("as_of", "replay_available_at", "target_time"))
        if any(not isinstance(t, int) or isinstance(t, bool) or t % 60 for t in (as_of, available, target)):
            raise ValueError("cache requires exact minute integer timestamps")
        if available != as_of + 60 or target <= available:
            raise ValueError("cache availability or target mismatch")
        h = record["horizon_key"]
        if h != "close" and target != as_of + int(h[:-1]) * 60:
            raise ValueError("cache horizon target mismatch")
        if record["session_date"] in date_asofs and date_asofs[record["session_date"]] != as_of:
            raise ValueError("cache date must use one same-as-of batch")
        date_asofs[record["session_date"]] = as_of
        probability = record["probabilities"]
        if set(probability) != set(CLASSES):
            raise ValueError("cache class order mismatch")
        numbers = [probability[c] for c in CLASSES] + [record["model_confidence"]]
        if any(
            not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) or not 0 <= x <= 1
            for x in numbers
        ) or not math.isclose(sum(numbers[:3]), 1, abs_tol=1e-6):
            raise ValueError("cache invalid probabilities/confidence")
        price = Decimal(str(record["anchor_price"]))
        if not price.is_finite() or price <= 0 or record["status"] not in ("ok", "abstained"):
            raise ValueError("cache invalid anchor/status")
        generated = datetime.fromisoformat(record["generated_at"])
        if generated.tzinfo is None or generated.utcoffset() is None:
            raise ValueError("cache real generation time must be aware")
        if any(not isinstance(record.get(k), str) or not record[k] for k in ("request_id", "payload_id", "input_id")):
            raise ValueError("cache request/input identity missing")
        result[key] = record
    if set(result) != required:
        raise ValueError("cache incomplete forecast coverage")
    return result


def augment_episodes(
    episodes: Sequence[EpisodeData],
    cache: dict[str, Any],
    profile: str,
    manifest: dict[str, Any],
    reg: dict[str, Any],
) -> list[EpisodeData]:
    """Add raw forecast blocks, available at simulated completion through target equality."""
    if profile not in PROFILES:
        raise ValueError("unregistered forecast profile")
    records = _validate_cache(cache, manifest, reg)
    names = tuple(manifest["feature_names"])
    expected_names = (
        *OHLCV_FEATURE_NAMES,
        *(f"missing_{n}" for n in OHLCV_FEATURE_NAMES),
        *(f"symbol_{symbol}" for symbol in reg["symbols"]),
    )
    if len(names) != 115 or names != expected_names:
        raise ValueError("base market feature order must have 115 fields")
    augmented_id = content_hash(
        {
            "base_prepared_data_id": manifest["data_id"],
            "cache_id": cache["cache_id"],
            "profile": profile,
            "feature_names": (*names, *FORECAST_NAMES),
            "version": "jev-rl-features-v1",
        }
    )
    output = []
    for episode in episodes:
        if episode.data_id != manifest["data_id"] or episode.feature_names != names:
            raise ValueError("episode base identity/feature order mismatch")
        extra = np.zeros((len(episode.ends), 24), dtype=np.float32)
        for ordinal, horizon in enumerate(HORIZONS):
            record = records[(episode.symbol, episode.session_date.isoformat(), horizon)]
            as_of = record["as_of"]
            target = episode.session_close if horizon == "close" else as_of + int(horizon[:-1]) * 60
            anchor_rows = np.flatnonzero((episode.ends == as_of) & (episode.available_at <= as_of))
            if (
                as_of != episode.session_open + 61 * 60
                or record["target_time"] != target
                or len(anchor_rows) != 1
                or not math.isclose(float(record["anchor_price"]), float(episode.closes[anchor_rows[0]]), rel_tol=1e-12)
            ):
                raise ValueError("cache as-of/target/past anchor mismatch")
            allowed = (episode.ends >= record["replay_available_at"]) & (episode.ends <= target)
            allowed &= episode.available_at <= episode.ends
            block = extra[:, ordinal * 8 : (ordinal + 1) * 8]
            block[allowed, :3] = (
                [record["probabilities"][c] for c in CLASSES] if profile == "JEV_ASSISTED" else [1 / 3] * 3
            )
            block[allowed, 3] = record["model_confidence"] if profile == "JEV_ASSISTED" else 0
            block[allowed, 4] = (episode.ends[allowed] - as_of) / (390 * 60)
            block[allowed, 5] = (target - episode.ends[allowed]) / (390 * 60)
            block[allowed, 6] = np.clip(episode.closes[allowed] / float(record["anchor_price"]) - 1, -1, 1)
            block[allowed, 7] = 1
        output.append(
            replace(
                episode,
                features=np.concatenate((episode.features, extra), axis=1),
                feature_names=(*names, *FORECAST_NAMES),
                data_id=augmented_id,
                episode_id="",
            )
        )
    return output


def validate_budget(
    budget: dict[str, Any], reg: dict[str, Any], manifest: dict[str, Any], cache: dict[str, Any]
) -> None:
    steps, seconds = budget.get("shared_steps"), budget.get("max_seconds")
    if (
        not isinstance(steps, int)
        or isinstance(steps, bool)
        or not 10240 <= steps <= 92160
        or steps % 128
        or not isinstance(seconds, (int, float))
        or isinstance(seconds, bool)
        or not math.isfinite(seconds)
        or not 0 < seconds <= 600
        or budget.get("device") != "cpu"
    ):
        raise ValueError("invalid registered CPU training budget caps")
    expected = {
        "registration_id": reg["registration_id"],
        "base_prepared_data_id": manifest["data_id"],
        "cache_id": cache["cache_id"],
    }
    if any(budget.get(k) != v for k, v in expected.items()):
        raise ValueError("budget cache/prepared/registration identity mismatch")


def _execution_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    return {
        "python": str(Path(sys.executable).absolute()),
        "scripts": {
            "run_jev_rl_ablation.py": sha256((root / "scripts" / "run_jev_rl_ablation.py").read_bytes()).hexdigest()
        },
    }


def train_profile(
    prepared: Path,
    cache_path: Path,
    registration: Path,
    budget_path: Path,
    profile: str,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError("training output is immutable")
    reg = training._read_sealed(registration, "registration_id")
    budget = training._read_sealed(budget_path, "budget_id")
    base, manifest = load_prepared(prepared, role="train")
    cache = _load_cache(cache_path)
    validate_budget(budget, reg, manifest, cache)
    if seed not in reg["seeds"]:
        raise ValueError("unregistered seed")
    episodes = augment_episodes(base, cache, profile, manifest, reg)
    source, lock, execution = training._source_provenance(), _lock_hash(), _execution_provenance()
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    config = training.registered_config(reg)
    model = training.make_model(reg["candidates"][0], episodes, config, seed, "cpu")
    started = time.monotonic()
    try:
        model.learn(
            total_timesteps=budget["shared_steps"],
            callback=training.make_budget_callback(
                budget["shared_steps"], budget["max_seconds"], output_dir / "progress.json"
            ),
        )
        elapsed = time.monotonic() - started
        result = training._metrics(model, elapsed, budget["shared_steps"])
        if elapsed > budget["max_seconds"]:
            result["status"] = "deadline_overrun"
        _, after_manifest = load_prepared(prepared, role="train")
        after_cache = _load_cache(cache_path)
        if (
            after_manifest != manifest
            or after_cache != cache
            or source != training._source_provenance()
            or lock != _lock_hash()
            or execution != _execution_provenance()
        ):
            result["status"] = "source_changed"
        temporary = output_dir / ".checkpoint.zip"
        model.save(temporary)
        temporary.replace(output_dir / "model.zip")
        torch, sb3, contrib = training._extras()
        result.update(
            {
                "profile": profile,
                "seed": seed,
                "candidate": reg["candidates"][0],
                "registration_id": reg["registration_id"],
                "base_prepared_data_id": manifest["data_id"],
                "augmented_data_id": episodes[0].data_id,
                "cache_id": cache["cache_id"],
                "budget_id": budget["budget_id"],
                "config_hash": config.content_hash,
                "source_provenance": source,
                "lock_sha256": lock,
                "execution_provenance": execution,
                "observation_dim": len(episodes[0].feature_names) + 12,
                "versions": {
                    "torch": torch.__version__,
                    "stable_baselines3": sb3.__version__,
                    "sb3_contrib": contrib.__version__,
                },
                "checkpoint_sha256": sha256((output_dir / "model.zip").read_bytes()).hexdigest(),
            }
        )
        return _seal(output_dir / "result.json", result, "training_id")
    finally:
        model.get_env().close()


def _run_worker(
    prepared: Path,
    cache_path: Path,
    registration: Path,
    budget_path: Path,
    profile: str,
    seed: int,
    output: Path,
    budget: dict[str, Any],
    progress: Path,
) -> dict[str, Any]:
    script = Path(__file__).resolve().parents[3] / "scripts" / "run_jev_rl_ablation.py"
    command = [str(Path(sys.executable).absolute()), str(script), "--worker", "--profile", profile, "--seed", str(seed)]
    for name, path in (
        ("prepared", prepared),
        ("cache", cache_path),
        ("registration", registration),
        ("budget", budget_path),
        ("output-dir", output),
    ):
        command += [f"--{name}", str(path)]
    started = time.monotonic()
    with (output.parent / f"{output.name}.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            state: dict[str, Any] = {"profile": profile, "seed": seed, "stage": "training"}
            child = output / "progress.json"
            if child.exists():
                with suppress(ValueError, OSError):
                    state.update(json.loads(child.read_text()))
            _write(progress, state)
            if time.monotonic() - started > budget["max_seconds"] + 60:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                return {"profile": profile, "seed": seed, "status": "timeout"}
            time.sleep(0.2)
    if process.returncode != 0:
        return {"profile": profile, "seed": seed, "status": "failed", "returncode": process.returncode}
    return training._read_sealed(output / "result.json", "training_id")


def _verify_training(
    runs: list[dict[str, Any]],
    reg: dict[str, Any],
    budget: dict[str, Any],
    source: dict[str, str],
    lock: str | None,
    execution: dict[str, Any],
) -> None:
    pairs = [(r.get("profile"), r.get("seed")) for r in runs]
    if len(pairs) != 6 or len(set(pairs)) != 6 or set(pairs) != {(p, s) for p in PROFILES for s in reg["seeds"]}:
        raise ValueError("all six complete registered runs required")
    capacity = runs[0].get("n_parameters")
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
        raise ValueError("training capacity evidence missing")
    for run in runs:
        _verify_hash(run, "training_id")
        expected = {
            "registration_id": reg["registration_id"],
            "base_prepared_data_id": budget["base_prepared_data_id"],
            "cache_id": budget["cache_id"],
            "budget_id": budget["budget_id"],
            "source_provenance": source,
            "lock_sha256": lock,
            "execution_provenance": execution,
            "config_hash": training.registered_config(reg).content_hash,
            "candidate": reg["candidates"][0],
            "versions": runs[0].get("versions"),
            "n_parameters": runs[0].get("n_parameters"),
            "observation_dim": 151,
        }
        if any(run.get(k) != v for k, v in expected.items()):
            raise ValueError("training source/data/budget/capacity identity mismatch")
        elapsed = run.get("elapsed_seconds")
        if (
            not isinstance(elapsed, (int, float))
            or isinstance(elapsed, bool)
            or not math.isfinite(elapsed)
            or not 0 < elapsed <= budget["max_seconds"]
        ):
            raise ValueError("training deadline exceeded or elapsed evidence invalid")
        if (
            run.get("status") != "complete"
            or run.get("actual_timesteps") != budget["shared_steps"]
            or run.get("n_updates") != budget["shared_steps"] // 128 * 10
        ):
            raise ValueError("all six complete equal-budget training runs required")


def mean_seed_metrics(seeds: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(seeds) != {"42", "43", "44"}:
        raise ValueError("three registered seed metrics required")
    values = list(seeds.values())
    valid = all(m["valid"] for m in values)
    days = set(values[0]["daily_returns"])
    if valid and any(set(m["daily_returns"]) != days for m in values):
        raise ValueError("seed evaluation dates differ")
    daily = {day: float(np.mean([m["daily_returns"][day] for m in values])) for day in sorted(days)} if valid else {}
    return {
        "valid": valid,
        "daily_returns": daily,
        "mean_daily_return": float(np.mean(list(daily.values()))) if daily else None,
        "max_episode_drawdown": max(m["max_episode_drawdown"] for m in values),
        "seeds": [42, 43, 44],
    }


def _improvement(
    means: dict[str, dict[str, Any]],
    controls: dict[str, dict[str, Any]],
    stress: dict[str, Any],
    metrics: list[dict[str, Any]],
    limit: float,
) -> dict[str, Any]:
    if not all(m["valid"] for m in metrics):
        return {"passed": False, "reason": "invalid or unresolved scored episodes", "intervals": {}}
    jev, context = (means[p] for p in ("JEV_ASSISTED", "CONTEXT_ONLY"))
    intervals = {
        "jev_minus_context": paired_interval(jev["daily_returns"], context["daily_returns"]),
        "jev_minus_cash": paired_interval(jev["daily_returns"], controls["cash"]["daily_returns"]),
    }
    checks = {
        "positive_mean_seed_increment_lower_bound": intervals["jev_minus_context"]["low"] > 0,
        "positive_jev_mean_seed_net_return": jev["mean_daily_return"] > 0,
        "positive_cash_lower_bound": intervals["jev_minus_cash"]["low"] > 0,
        "drawdown_limit": all(m["max_episode_drawdown"] <= limit for m in metrics),
        "positive_doubled_cost_mean_seed_increment": stress["4"]["mean_seed"]["JEV_ASSISTED"]["mean_daily_return"]
        > stress["4"]["mean_seed"]["CONTEXT_ONLY"]["mean_daily_return"],
    }
    return {"passed": all(checks.values()), "checks": checks, "intervals": intervals}


def _forecast_metrics(probabilities: list[list[float]], labels: list[int], confidence: list[float]) -> dict[str, Any]:
    if not labels:
        raise ValueError("nonempty forecast grading targets required")
    probs, actual = np.asarray(probabilities, dtype=float), np.asarray(labels, dtype=int)
    chosen = probs.argmax(axis=1)
    selected = np.minimum(probs.max(axis=1), np.asarray(confidence)) >= 0.6
    return {
        "records": len(labels),
        "accuracy": float(np.mean(chosen == actual)),
        "log_loss": float(-np.log(np.clip(probs[np.arange(len(actual)), actual], 1e-15, 1)).mean()),
        "Brier": float(np.mean(np.sum((probs - np.eye(3)[actual]) ** 2, axis=1))),
        "coverage_0_6": float(selected.mean()),
        "selected_accuracy_0_6": float(np.mean(chosen[selected] == actual[selected])) if selected.any() else None,
    }


def grade_forecasts(
    train: Sequence[EpisodeData], tune: Sequence[EpisodeData], test: Sequence[EpisodeData], cache: dict[str, Any]
) -> dict[str, Any]:
    """Call only after both policy selections freeze; priors never enter policy observations."""
    records = {(r["symbol"], r["session_date"], r["horizon_key"]): r for r in cache["records"]}
    report: dict[str, Any] = {
        "evidence_mode": "hypothetical_retrospective_teacher_forecasts",
        "vendor_pretraining_cutoff": "unknown",
        "horizons": {},
    }
    for horizon in HORIZONS:
        groups: dict[str, tuple[list[int], list[list[float]], list[float]]] = {}
        for role, episodes in (("train", train), ("tune", tune), ("test", test)):
            labels, probabilities, confidence = [], [], []
            for episode in episodes:
                record = records[(episode.symbol, episode.session_date.isoformat(), horizon)]
                target = record["target_time"]
                indices = np.flatnonzero((episode.ends == target) & (episode.available_at <= target))
                if len(indices) != 1:
                    raise ValueError("forecast exact minute-close target unavailable")
                change = Decimal(str(episode.closes[indices[0]])) / Decimal(str(record["anchor_price"])) - 1
                labels.append(0 if change < Decimal("-.001") else 2 if change > Decimal(".001") else 1)
                probabilities.append([record["probabilities"][c] for c in CLASSES])
                confidence.append(record["model_confidence"])
            groups[role] = labels, probabilities, confidence
        training_labels = groups["train"][0]
        if not training_labels:
            raise ValueError("TRAIN-only forecast prior labels absent")
        prior = (np.bincount(training_labels, minlength=3) / len(training_labels)).tolist()
        grades: dict[str, Any] = {"train_prior": prior, "train_labels": len(training_labels)}
        for role in ("tune", "test"):
            labels, probabilities, confidence = groups[role]
            grades[role] = {
                "jev": _forecast_metrics(probabilities, labels, confidence),
                "prior": _forecast_metrics([prior] * len(labels), labels, [max(prior)] * len(labels)),
            }
        report["horizons"][horizon] = grades
    return report


def _inventory(output: Path) -> dict[str, str]:
    return {
        str(path.relative_to(output)): sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "report.json"
    }


def run_ablation(prepared: Path, cache_path: Path, registration: Path, budget: Path, output_dir: Path) -> Path:
    """Six bounded workers, all-seed TUNE, both sealed selections, then common locked TEST."""
    if output_dir.exists():
        raise ValueError("ablation output is immutable")
    prepared, cache_path, registration, budget, output_dir = [
        path.resolve() for path in (prepared, cache_path, registration, budget, output_dir)
    ]
    reg = training._read_sealed(registration, "registration_id")
    allocation = training._read_sealed(budget, "budget_id")
    _, manifest = load_prepared(prepared, role="train")
    cache = _load_cache(cache_path)
    _validate_cache(cache, manifest, reg)
    validate_budget(allocation, reg, manifest, cache)
    source, lock, execution = training._source_provenance(), _lock_hash(), _execution_provenance()
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    _seal(
        output_dir / "ablation-seal.json",
        {
            "registration": reg,
            "budget": allocation,
            "manifest": manifest,
            "cache_id": cache["cache_id"],
            "source_provenance": source,
            "lock_sha256": lock,
            "execution_provenance": execution,
        },
        "ablation_id",
    )
    runs: list[dict[str, Any]] = []
    try:
        return _execute(
            prepared,
            cache_path,
            registration,
            budget,
            output_dir,
            reg,
            allocation,
            manifest,
            cache,
            source,
            lock,
            execution,
            runs,
        )
    except Exception as error:
        _write(
            output_dir / "progress.json", {"stage": "failed", "error_type": type(error).__name__, "error": str(error)}
        )
        _seal(
            output_dir / "report.json",
            {
                "schema_version": "jev-rl-ablation-report-v1",
                "status": "failed",
                "training": runs,
                "registration_id": reg["registration_id"],
                "base_prepared_data_id": manifest["data_id"],
                "cache_id": cache["cache_id"],
                "budget_id": allocation["budget_id"],
                "error_type": type(error).__name__,
                "error": str(error),
                "inventory": _inventory(output_dir),
            },
            "report_id",
        )
        raise


def _execute(
    prepared: Path,
    cache_path: Path,
    registration: Path,
    budget: Path,
    output: Path,
    reg: dict[str, Any],
    allocation: dict[str, Any],
    manifest: dict[str, Any],
    cache: dict[str, Any],
    source: dict[str, str],
    lock: str | None,
    execution: dict[str, Any],
    runs: list[dict[str, Any]],
) -> Path:
    directories = {}
    for profile in PROFILES:
        for seed in reg["seeds"]:
            directory = output / f"{profile}-{seed}"
            runs.append(
                _run_worker(
                    prepared,
                    cache_path,
                    registration,
                    budget,
                    profile,
                    seed,
                    directory,
                    allocation,
                    output / "progress.json",
                )
            )
            directories[(profile, seed)] = directory
    _write(output / "training.json", {"runs": runs})
    _verify_training(runs, reg, allocation, source, lock, execution)
    for run in runs:
        _verify_checkpoint(directories[(run["profile"], run["seed"])], run)

    def unchanged() -> None:
        if (
            source != training._source_provenance()
            or lock != _lock_hash()
            or execution != _execution_provenance()
            or cache != _load_cache(cache_path)
        ):
            raise ValueError("source/cache changed during ablation")
        if training._read_sealed(registration, "registration_id") != reg:
            raise ValueError("registration changed during ablation")
        if training._read_sealed(budget, "budget_id") != allocation:
            raise ValueError("budget changed during ablation")

    unchanged()
    base_tune, current = load_prepared(prepared, role="tune")
    if current != manifest:
        raise ValueError("prepared data changed before tuning")
    config = training.registered_config(reg)
    tune: dict[str, dict[str, Any]] = {p: {} for p in PROFILES}
    for profile in PROFILES:
        episodes = augment_episodes(base_tune, cache, profile, manifest, reg)
        for run in (r for r in runs if r["profile"] == profile):
            tune[profile][str(run["seed"])] = _score(
                episodes,
                config,
                _neural(directories[(profile, run["seed"])], run, "cpu"),
                output / f"tune-{profile}-{run['seed']}.json",
            )
    if not all(m["valid"] for profile in tune.values() for m in profile.values()):
        raise ValueError("invalid or unresolved TUNE policy episodes")
    chosen = {}
    for profile in PROFILES:
        seed = max(reg["seeds"], key=lambda s: (tune[profile][str(s)]["mean_daily_return"], -s))
        run = next(r for r in runs if (r["profile"], r["seed"]) == (profile, seed))
        chosen[profile] = {
            "seed": seed,
            "training_id": run["training_id"],
            "checkpoint_sha256": run["checkpoint_sha256"],
        }
    selection_path = output / "selection.json"
    selection = _seal(
        selection_path,
        {
            "stage": "selection_frozen",
            "profiles": chosen,
            "registration_id": reg["registration_id"],
            "prepared_data_id": manifest["data_id"],
            "base_prepared_data_id": manifest["data_id"],
            "cache_id": cache["cache_id"],
            "budget_id": allocation["budget_id"],
            "training_ids": [r["training_id"] for r in runs],
            "checkpoint_hashes": {f"{r['profile']}-{r['seed']}": r["checkpoint_sha256"] for r in runs},
            "criterion": (
                "best TUNE mean date net return within profile; tie lowest seed; TEST averages all three seeds"
            ),
        },
        "selection_id",
    )
    # Both profile choices are in the single seal required by unchanged base loader.
    unchanged()
    base_test, current = load_prepared(prepared, role="test", selection_path=selection_path)
    if current != manifest:
        raise ValueError("prepared data changed before locked TEST")
    controls = study._controls(manifest, reg)
    tune_controls = {
        key: _score(base_tune, config, policy, output / f"tune-{key}.json") for key, policy in controls.items()
    }
    test_controls = {
        key: _score(base_test, config, policy, output / f"test-{key}.json") for key, policy in controls.items()
    }
    test: dict[str, dict[str, Any]] = {p: {} for p in PROFILES}
    stress: dict[str, Any] = {}
    metrics = (
        [m for group in tune.values() for m in group.values()]
        + list(tune_controls.values())
        + list(test_controls.values())
    )
    for cost in (2, 4, 8):
        results: dict[str, dict[str, Any]] = {p: {} for p in PROFILES}
        for profile in PROFILES:
            episodes = augment_episodes(base_test, cache, profile, manifest, reg)
            for run in (r for r in runs if r["profile"] == profile):
                results[profile][str(run["seed"])] = _score(
                    episodes,
                    replace(config, cost_bps=Decimal(cost)),
                    _neural(directories[(profile, run["seed"])], run, "cpu"),
                    output / f"test-{cost}bps-{profile}-{run['seed']}.json",
                )
        metrics.extend(m for group in results.values() for m in group.values())
        if cost == 2:
            test = results
        else:
            stressed_controls = {
                key: _score(
                    base_test, replace(config, cost_bps=Decimal(cost)), policy, output / f"test-{cost}bps-{key}.json"
                )
                for key, policy in controls.items()
            }
            metrics.extend(stressed_controls.values())
            stress[str(cost)] = {
                "profiles": results,
                "controls": stressed_controls,
                "mean_seed": {p: mean_seed_metrics(results[p]) for p in PROFILES},
            }
    means = {p: mean_seed_metrics(test[p]) for p in PROFILES}
    gate = _improvement(means, test_controls, stress, metrics, float(config.daily_loss / config.initial_cash))
    # Forecast quality does not influence policy training or checkpoint selection.
    base_train, current = load_prepared(prepared, role="train")
    if current != manifest:
        raise ValueError("prepared data changed during grading")
    forecast = grade_forecasts(base_train, base_tune, base_test, cache)
    unchanged()
    if training._read_sealed(selection_path, "selection_id") != selection:
        raise ValueError("selection changed during locked TEST")
    for run in runs:
        _verify_checkpoint(directories[(run["profile"], run["seed"])], run)
    _write(output / "progress.json", {"stage": "complete", "selection_id": selection["selection_id"]})
    _seal(
        output / "report.json",
        {
            "schema_version": "jev-rl-ablation-report-v1",
            "experiment_type": "JEV_RL_ABLATION",
            "evidence_mode": "hypothetical_retrospective_jev_cache",
            "status": "improved_among_tested" if gate["passed"] else "inconclusive_not_promoted",
            "selection": selection,
            "training": runs,
            "tune_seeds": tune,
            "tune_controls": tune_controls,
            "test_seeds": test,
            "test_mean_seed": means,
            "test_controls": test_controls,
            "cost_stress": stress,
            "improvement_gate": gate,
            "forecast_grading": forecast,
            "source_provenance": source,
            "lock_sha256": lock,
            "execution_provenance": execution,
            "registration_id": reg["registration_id"],
            "base_prepared_data_id": manifest["data_id"],
            "cache_id": cache["cache_id"],
            "budget_id": allocation["budget_id"],
            "constraints": {
                "new_jev_calls": 0,
                "broker_orders": 0,
                "vendor_pretraining_cutoff": "unknown",
                "jev_weights_trained": False,
                "calibration": "raw vendor probabilities",
            },
            "interpretation": (
                "Retrospective teacher outputs; mean of three fixed seeds per date; "
                "independent symbol-day capital resets"
            ),
            "inventory": _inventory(output),
        },
        "report_id",
    )
    return output / "report.json"


def load_report(path: Path) -> dict[str, Any]:
    report = training._read_sealed(path, "report_id")
    if report.get("schema_version") != "jev-rl-ablation-report-v1":
        raise ValueError("unsupported Jev RL report schema")
    root = path.parent.resolve()
    for name, digest in report["inventory"].items():
        artifact = root / name
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or artifact.is_symlink()
            or not artifact.is_file()
            or not artifact.resolve().is_relative_to(root)
            or sha256(artifact.read_bytes()).hexdigest() != digest
        ):
            raise ValueError("Jev RL report artifact integrity mismatch")
    return report
