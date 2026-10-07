"""Optional, bounded offline learners; never release tuning or locked test observations."""

from __future__ import annotations

import importlib
import json
import math
import os
import time
from collections.abc import Sequence
from contextlib import suppress
from hashlib import sha256
from pathlib import Path
from typing import Any

from tradecopilot.forecast.contracts import content_hash

from .contracts import EpisodeData, SimConfig

OPTIMIZER: dict[str, Any] = {
    "n_steps": 128,
    "batch_size": 128,
    "n_epochs": 10,
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
}
_CANDIDATES: dict[str, dict[str, Any]] = {
    "ppo-64": {"key": "ppo-64", "algorithm": "PPO", "pi": [64, 64], "vf": [64, 64]},
    "ppo-256": {"key": "ppo-256", "algorithm": "PPO", "pi": [256, 256], "vf": [256, 256]},
    "recurrent-256": {
        "key": "recurrent-256",
        "algorithm": "RecurrentPPO",
        "pi": [256, 128],
        "vf": [256, 128],
        "lstm_hidden_size": 256,
        "n_lstm_layers": 1,
        "shared_lstm": False,
        "enable_critic_lstm": True,
    },
}
_THREADS_CONFIGURED = False


def _extras() -> tuple[Any, Any, Any]:
    global _THREADS_CONFIGURED
    try:
        torch = importlib.import_module("torch")
        sb3 = importlib.import_module("stable_baselines3")
        contrib = importlib.import_module("sb3_contrib")
    except ImportError as error:
        raise RuntimeError("Neural training requires the optional tradecopilot[rl] dependencies") from error
    if not _THREADS_CONFIGURED:
        torch.set_num_threads(1)
        # PyTorch permits configuring the interop pool only before it starts.
        with suppress(RuntimeError):
            torch.set_num_interop_threads(1)
        _THREADS_CONFIGURED = True
    return torch, sb3, contrib


def device_support() -> dict[str, Any]:
    torch, _, _ = _extras()
    return {
        "cpu": True,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_parity": "unverified",
    }


def _candidate(candidate: dict[str, Any]) -> None:
    expected = _CANDIDATES.get(candidate.get("key", ""))
    if expected is None or candidate != expected:
        raise ValueError("candidate must exactly match a declared architecture")


def make_model(
    candidate: dict[str, Any], episodes: Sequence[EpisodeData], config: SimConfig, seed: int, device: str = "cpu"
) -> Any:
    _candidate(candidate)
    if device not in ("cpu", "mps"):
        raise ValueError("explicit cpu or mps device required")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("invalid seed")
    if not isinstance(config, SimConfig):
        raise ValueError("SimConfig required")
    torch, sb3, contrib = _extras()
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    from .env import TradeCopilotEnv

    kwargs = {"net_arch": {"pi": candidate["pi"], "vf": candidate["vf"]}}
    recurrent = candidate["algorithm"] == "RecurrentPPO"
    if recurrent:
        kwargs.update(
            {key: candidate[key] for key in ("lstm_hidden_size", "n_lstm_layers", "shared_lstm", "enable_critic_lstm")}
        )
    algorithm = contrib.RecurrentPPO if recurrent else sb3.PPO
    return algorithm(
        "MlpLstmPolicy" if recurrent else "MlpPolicy",
        TradeCopilotEnv(episodes, config),
        policy_kwargs=kwargs,
        seed=seed,
        device=device,
        verbose=0,
        **OPTIMIZER,
    )


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def make_budget_callback(steps: int, max_seconds: float, progress_path: Path | None) -> Any:
    """A completed rollout reaches its optimizer update, including at the exact step cap."""
    _extras()
    base = importlib.import_module("stable_baselines3.common.callbacks").BaseCallback
    started = time.monotonic()

    def progress(self: Any) -> None:
        if progress_path is not None:
            _atomic_json(
                progress_path,
                {"actual_timesteps": int(self.num_timesteps), "elapsed_seconds": time.monotonic() - started},
            )

    def on_step(self: Any) -> bool:
        now = time.monotonic() - started
        if self.num_timesteps > steps:
            return False
        # SB3 invokes this before committing the transition to its rollout buffer.
        # Never discard a final complete rollout because its last step hit a deadline.
        return bool(self.num_timesteps % 128 == 0 or now < max_seconds)

    callback = type("BudgetCallback", (base,), {"_on_step": on_step, "_on_rollout_end": progress})()
    return callback


def _metrics(model: Any, elapsed: float, steps: int) -> dict[str, Any]:
    return {
        "actual_timesteps": int(model.num_timesteps),
        "n_updates": int(model._n_updates),
        "elapsed_seconds": elapsed,
        "training_fps": model.num_timesteps / max(elapsed, 1e-9),
        "n_parameters": sum(p.numel() for p in model.policy.parameters()),
        "device": str(model.device),
        "status": "complete" if model.num_timesteps == steps and model._n_updates == steps // 128 * 10 else "stopped",
    }


def benchmark_model(
    candidate: dict[str, Any],
    episodes: Sequence[EpisodeData],
    config: SimConfig,
    seed: int,
    device: str,
    steps: int = 1024,
    max_seconds: float = 30,
) -> dict[str, Any]:
    if isinstance(steps, bool) or steps < 128 or steps > 10240 or steps % 128:
        raise ValueError("benchmark requires 128..10240 steps in complete rollouts")
    if not math.isfinite(max_seconds) or not 0 < max_seconds <= 60:
        raise ValueError("benchmark requires a finite deadline <=60 seconds")
    start = time.monotonic()
    model = make_model(candidate, episodes, config, seed, device)
    warmup = time.monotonic() - start
    start = time.monotonic()
    try:
        model.learn(total_timesteps=steps, callback=make_budget_callback(steps, max_seconds, None))
        result = _metrics(model, time.monotonic() - start, steps)
        result["initialization_seconds"] = warmup
        return result
    finally:
        model.get_env().close()


def _read_sealed(path: Path, key: str) -> dict[str, Any]:
    value: Any = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get(key) != content_hash({k: v for k, v in value.items() if k != key}):
        raise ValueError(f"invalid {key}")
    return value


def registered_config(registration: dict[str, Any]) -> SimConfig:
    env = registration["environment"]
    return SimConfig(
        **{
            name: env[alias]
            for name, alias in {
                "initial_cash": "initial_cash",
                "notional_cap": "notional_cap",
                "daily_loss": "loss_lock",
                "cost_bps": "base_cost_bps_per_side",
                "fee_per_fill": "fee_per_fill",
                "capacity_fraction": "capacity_fraction",
                "quantity_quantum": "quantity_quantum",
                "warmup_minutes": "warmup_minutes",
                "latency_minutes": "latency_minutes",
                "forced_close_buffer_minutes": "forced_close_buffer_minutes",
            }.items()
        }
    )


def _source_provenance() -> dict[str, str]:
    package = Path(__file__).resolve().parent.parent
    paths = sorted(package.rglob("*.py"))
    return {str(path.relative_to(package)): sha256(path.read_bytes()).hexdigest() for path in paths}


def train_seed(
    prepared_directory: Path,
    registration_path: Path,
    budget_path: Path,
    candidate_key: str,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError("training output is immutable")
    reg = _read_sealed(registration_path, "registration_id")
    budget = _read_sealed(budget_path, "budget_id")
    steps, seconds = budget.get("shared_steps"), budget.get("max_seconds")
    if (
        not isinstance(steps, int)
        or isinstance(steps, bool)
        or not 10240 <= steps <= 100000
        or steps % 128
        or not isinstance(seconds, (int, float))
        or isinstance(seconds, bool)
        or not math.isfinite(seconds)
        or not 0 < seconds <= 600
    ):
        raise ValueError("invalid training budget caps")
    if budget.get("registration_id") != reg["registration_id"] or reg.get("optimizer") != OPTIMIZER:
        raise ValueError("registration/budget/optimizer mismatch")
    candidates = [candidate for candidate in reg["candidates"] if candidate["key"] == candidate_key]
    if len(candidates) != 1 or seed not in reg["seeds"]:
        raise ValueError("unregistered candidate or seed")
    _candidate(candidates[0])
    config = registered_config(reg)
    from .data import load_prepared

    episodes, manifest = load_prepared(prepared_directory, role="train")
    if manifest["registration_id"] != reg["registration_id"] or manifest["data_id"] != budget["prepared_data_id"]:
        raise ValueError("prepared data identity mismatch")
    source = _source_provenance()
    lock = Path(__file__).resolve().parents[3] / "uv.lock"
    lock_hash = sha256(lock.read_bytes()).hexdigest() if lock.exists() else None
    output_dir.mkdir(parents=True, exist_ok=False)
    model = make_model(candidates[0], episodes, config, seed, budget["device"])
    start = time.monotonic()
    try:
        model.learn(total_timesteps=steps, callback=make_budget_callback(steps, seconds, output_dir / "progress.json"))
        result = _metrics(model, time.monotonic() - start, steps)
        current_lock = sha256(lock.read_bytes()).hexdigest() if lock.exists() else None
        if source != _source_provenance() or lock_hash != current_lock:
            result["status"] = "source_changed"
        temporary = output_dir / ".checkpoint.zip"
        model.save(temporary)
        temporary.replace(output_dir / "model.zip")
        torch, sb3, contrib = _extras()
        result.update(
            {
                "candidate_key": candidate_key,
                "candidate": candidates[0],
                "seed": seed,
                "registration_id": reg["registration_id"],
                "prepared_data_id": manifest["data_id"],
                "budget_id": budget["budget_id"],
                "config_hash": config.content_hash,
                "source_provenance": source,
                "lock_sha256": lock_hash,
                "versions": {
                    "torch": torch.__version__,
                    "stable_baselines3": sb3.__version__,
                    "sb3_contrib": contrib.__version__,
                },
                "checkpoint_sha256": sha256((output_dir / "model.zip").read_bytes()).hexdigest(),
            }
        )
        result["training_id"] = content_hash(result)
        _atomic_json(output_dir / "result.json", result)
        _atomic_json(
            output_dir / "progress.json",
            {key: result[key] for key in ("actual_timesteps", "elapsed_seconds", "status")},
        )
        return result
    finally:
        model.get_env().close()
