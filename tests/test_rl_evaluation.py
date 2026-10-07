from datetime import UTC, date, datetime
from decimal import Decimal

import numpy as np
import pytest

from tradecopilot.rl.contracts import EpisodeData, SimConfig


def episode():
    start = int(datetime(2026, 1, 5, 14, 30, tzinfo=UTC).timestamp())
    starts = np.arange(80, dtype=np.int64) * 60 + start
    return EpisodeData(
        "AAPL",
        date(2026, 1, 5),
        start,
        start + 80 * 60,
        starts,
        starts + 60,
        starts + 60,
        np.full(80, 100.0),
        np.full(80, 100.0),
        np.full(80, 10000.0),
        np.zeros((80, 1), dtype=np.float32),
        ("return_5m_bps",),
        "fixture",
    )


def test_policy_replay_cash_and_roundtrip_costs_reconcile():
    from tradecopilot.rl.evaluation import aggregate_episodes, replay_policy
    from tradecopilot.rl.policies import FixedPolicy

    ep = episode()
    config = SimConfig(cost_bps=Decimal(0), fee_per_fill=Decimal(1))
    cash = replay_policy(ep, config, FixedPolicy("cash"), keep_ledger=True)
    hold = replay_policy(ep, config, FixedPolicy("hold"), keep_ledger=True)
    assert cash["net_return"] == 0 and cash["trades"] == []
    assert hold["net_return"] == pytest.approx(-0.0002)
    assert hold["reward_sum_bps"] == pytest.approx(-2)
    assert len(hold["trades"]) == 2
    metrics = aggregate_episodes([hold])
    assert metrics["valid"] and metrics["mean_daily_return"] == pytest.approx(-0.0002)
    assert metrics["daily_sharpe"] is None
    bad = dict(hold, valid=False, resolved=False, net_return=None)
    invalid = aggregate_episodes([bad])
    assert not invalid["valid"] and invalid["mean_daily_return"] is None
    assert invalid["episodes"] == 1


def test_selection_averages_registered_seeds_and_rejects_missing_runs():
    from tradecopilot.rl.evaluation import select_architecture

    rows = []
    values = {"ppo-64": [0.5, -0.5, -0.5], "ppo-256": [0.1, 0.1, 0.1], "recurrent-256": [0.2, -0.2, 0]}
    for name, scores in values.items():
        for seed, score in zip((42, 43, 44), scores, strict=True):
            rows.append(
                {
                    "candidate": name,
                    "seed": seed,
                    "training_status": "complete",
                    "metrics": {"valid": True, "mean_daily_return": score, "max_episode_drawdown": 0.001},
                }
            )
    result = select_architecture(rows, (42, 43, 44), tuple(values))
    assert result["candidate"] == "ppo-256" and result["seed"] == 42
    with pytest.raises(ValueError, match="complete"):
        select_architecture(rows[:-1], (42, 43, 44), tuple(values))


def test_recurrent_wrapper_carries_state_and_resets_on_episode_boundary():
    from tradecopilot.rl.policies import NeuralPolicy

    class Model:
        def __init__(self):
            self.calls = []

        def predict(self, obs, **kwargs):
            self.calls.append(kwargs)
            return np.asarray(2), len(self.calls)

    model = Model()
    policy = NeuralPolicy(model, recurrent=True)
    policy.reset_memory()
    assert policy.act(np.zeros(3, dtype=np.float32)) == 2
    policy.act(np.zeros(3, dtype=np.float32))
    policy.reset_memory()
    policy.act(np.zeros(3, dtype=np.float32))
    assert model.calls[0]["state"] is None and model.calls[0]["episode_start"].all()
    assert model.calls[1]["state"] == 1 and not model.calls[1]["episode_start"].any()
    assert model.calls[2]["state"] is None and model.calls[2]["episode_start"].all()


def test_rule_holds_cash_when_feature_is_missing():
    from tradecopilot.rl.policies import FixedPolicy

    policy = FixedPolicy("rule", feature_index=0, missing_index=1, mean=10)
    assert policy.act(np.asarray([0, 0], dtype=np.float32)) == 2
    assert policy.act(np.asarray([0, 1], dtype=np.float32)) == 0
