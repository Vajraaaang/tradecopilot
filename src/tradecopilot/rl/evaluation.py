"""Policy outcomes are scored from ledgers, independently of forecast accuracy."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from tradecopilot.rl.contracts import EpisodeData, SimConfig


class ReplayPolicy(Protocol):
    def reset_memory(self) -> None: ...
    def act(self, observation: NDArray[np.float32]) -> int: ...


def replay_policy(
    episode: EpisodeData, config: SimConfig, policy: ReplayPolicy, *, keep_ledger: bool = False
) -> dict[str, Any]:
    from tradecopilot.rl.env import TradeCopilotEnv

    env = TradeCopilotEnv([episode], config)
    observation, _ = env.reset(seed=42, options={"episode_id": episode.episode_id})
    policy.reset_memory()
    for _ in range(len(episode.starts) + 1):
        action = policy.act(observation)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    else:
        raise RuntimeError("replay exceeded registered episode bounds")
    summary = env.episode_summary()
    if summary["valid"]:
        expected = float(summary["net_return"]) * 10000
        if not np.isclose(summary["reward_sum_bps"], expected, atol=1e-7, rtol=0):
            raise ValueError("reward and net-equity ledger do not reconcile")
    if keep_ledger:
        summary["ledger"] = env.ledger
    policy.reset_memory()
    env.close()
    return summary


def aggregate_episodes(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows or len({r["episode_id"] for r in rows}) != len(rows):
        raise ValueError("nonempty unique episode outcomes are required")
    valid = all(r["valid"] and r["resolved"] and r["net_return"] is not None for r in rows)
    daily: dict[str, list[float]] = defaultdict(list)
    if valid:
        for row in rows:
            value = float(row["net_return"])
            if not np.isfinite(value):
                raise ValueError("nonfinite return")
            daily[row["session_date"]].append(value)
    means = {day: float(np.mean(values)) for day, values in sorted(daily.items())}
    returns = np.asarray(list(means.values()), dtype=float)
    deviation = float(returns.std(ddof=1)) if len(returns) > 1 else 0
    return {
        "valid": valid,
        "episodes": len(rows),
        "dates": len({r["session_date"] for r in rows}),
        "invalid_episodes": sum(not (r["valid"] and r["resolved"]) for r in rows),
        "mean_daily_return": float(returns.mean()) if valid else None,
        "cumulative_fixed_notional_return": float(returns.sum()) if valid else None,
        "daily_sharpe": float(returns.mean()) / deviation if valid and deviation > 1e-12 else None,
        "sharpe_annualized": False,
        "daily_returns": means,
        "max_episode_drawdown": max(float(r["max_drawdown"]) for r in rows),
        "trade_count": sum(len(r["trades"]) for r in rows),
        "fees": sum(float(r["fees"]) for r in rows),
        "execution_drag": sum(float(r["execution_drag"]) for r in rows),
        "turnover": sum(float(r["turnover"]) for r in rows),
        "interpretation": "Equal-weight independent symbol-day books; fixed capital resets, no shared-book claim",
    }


def select_architecture(
    rows: Sequence[dict[str, Any]], seeds: tuple[int, ...], candidates: tuple[str, ...]
) -> dict[str, Any]:
    pairs = [(r["candidate"], r["seed"]) for r in rows]
    expected = {(c, s) for c in candidates for s in seeds}
    if len(pairs) != len(set(pairs)) or set(pairs) != expected:
        raise ValueError("complete registered candidate/seed comparison required")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["candidate"]].append(row)
    eligible = []
    table = []
    for key in candidates:
        values = groups[key]
        good = all(r["training_status"] == "complete" and r["metrics"]["valid"] for r in values)
        if good:
            score = float(np.mean([r["metrics"]["mean_daily_return"] for r in values]))
            if not np.isfinite(score):
                raise ValueError("invalid selection score")
            eligible.append((score, -candidates.index(key), key))
        else:
            score = None
        table.append({"candidate": key, "eligible": good, "mean_seed_daily_return": score})
    if not eligible:
        raise ValueError("no complete valid architecture comparison")
    key = max(eligible)[2]
    best = max(groups[key], key=lambda r: (r["metrics"]["mean_daily_return"], -r["seed"]))
    return {
        "candidate": key,
        "seed": best["seed"],
        "ranking": table,
        "criterion": "mean tuning date return across every registered seed; no test scores",
    }


def paired_interval(model: dict[str, float], control: dict[str, float]) -> dict[str, Any]:
    days = sorted(model)
    if set(days) != set(control) or len(days) < 5:
        raise ValueError("paired date interval requires same dates and at least five sessions")
    differences = np.asarray([model[day] - control[day] for day in days])
    rng = np.random.default_rng(42)
    samples = [float(differences[rng.integers(0, len(days), len(days))].mean()) for _ in range(1000)]
    return {
        "method": "paired_date_bootstrap_95pct",
        "dates": len(days),
        "mean": float(differences.mean()),
        "low": float(np.percentile(samples, 2.5)),
        "high": float(np.percentile(samples, 97.5)),
    }
