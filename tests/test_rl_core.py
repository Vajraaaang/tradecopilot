from datetime import date
from decimal import Decimal as D

import numpy as np
import pytest

pytest.importorskip("gymnasium")
from tradecopilot.rl.contracts import EpisodeData, SimConfig
from tradecopilot.rl.env import TradeCopilotEnv


def episode(prices=None, volumes=None, late=None, name="a"):
    n = 80
    starts = np.arange(n, dtype=np.int64) * 60
    ends = starts + 60
    available = ends.copy()
    if late is not None:
        available[late] += 1
    p = np.full(n, 100.0, dtype=np.float64) if prices is None else prices
    v = np.full(n, 10000.0, dtype=np.float64) if volumes is None else volumes
    return EpisodeData(
        "SPY",
        date(2026, 1, 2),
        0,
        n * 60,
        starts,
        ends,
        available,
        p.copy(),
        p.copy(),
        v,
        np.zeros((n, 2), dtype=np.float32),
        ("x", "y"),
        name,
    )


def env(ep=None, **kw):
    return TradeCopilotEnv([episode() if ep is None else ep], SimConfig(cost_bps=D(0), **kw))


def test_round_trip_exact_fees_and_reward():
    e = env(fee_per_fill=D(1))
    e.reset()
    _, r1, _, _, _ = e.step(2)
    assert e.state.shares == D(10)
    _, r2, _, _, _ = e.step(0)
    assert e.state.cash == D(9998)
    assert r1 + r2 == -2


@pytest.mark.parametrize("gap", [50.0, 200.0])
def test_flatten_is_shares_across_gap(gap):
    p = np.full(80, 100.0, dtype=np.float64)
    p[64:] = gap
    e = env(episode(p), daily_loss=D(10000))
    e.reset()
    e.step(2)
    _, _, _, _, info = e.step(0)
    assert e.state.shares == 0
    assert D(info["fills"][0]["quantity"]) == D(10)


def test_partial_latch_no_retry_and_observable():
    e = env(capacity_fraction=D("0.0001"))
    e.reset()
    e.step(2)
    assert e.state.shares == 1
    assert e.state.last_committed_target_level == 2
    obs, _, _, _, info = e.step(2)
    assert not info["fills"]
    assert e.state.shares == 1
    assert np.isfinite(obs).all()


def test_late_execution_no_fill():
    e = env(episode(late=62))
    e.reset()
    _, _, terminated, truncated, info = e.step(2)
    assert truncated and not terminated
    assert not info["fills"]
    assert e.state.cash == D(10000)
    assert not info["valid"]


def test_config_and_array_guards():
    with pytest.raises(ValueError):
        SimConfig(capacity_fraction=D("NaN"))
    ep = episode()
    assert not ep.features.flags.writeable
    assert SimConfig().content_hash == SimConfig().content_hash


def test_seed_isolation_and_future_mutation():
    p = np.full(80, 100.0, dtype=np.float64)
    p[65:] = 200
    a, b = env(), env(episode(p))
    assert np.array_equal(a.reset(seed=42)[0], b.reset(seed=42)[0])
    assert np.array_equal(a.step(2)[0], b.step(2)[0])
    a.reset(seed=42)
    assert a.state.shares == 0 and a.state.last_committed_target_level == 0


def test_unresolved_capacity_and_checker():
    from gymnasium.utils.env_checker import check_env

    check_env(env(), skip_render_check=True)
    v = np.full(80, 10000.0, dtype=np.float64)
    v[63:] = 0
    e = env(episode(volumes=v))
    e.reset()
    e.step(2)
    done = False
    while not done:
        _, _, term, trunc, _ = e.step(2)
        done = term or trunc
    summary = e.episode_summary()
    assert e.state.shares > 0
    assert not summary["resolved"] and not summary["valid"]
    assert summary["net_return"] is None


def test_rejection_latch_and_no_fee_on_hold():
    v = np.zeros(80, dtype=np.float64)
    e = env(episode(volumes=v), fee_per_fill=D(1))
    e.reset()
    _, _, _, _, info = e.step(2)
    assert e.state.last_committed_target_level == 2
    assert info["rejections"] == ["no_liquidity"]
    assert e.state.fees == 0
    _, _, _, _, info = e.step(2)
    assert not info["fills"] and not info["rejections"]
    e = env(fee_per_fill=D(1))
    e.reset()
    e.step(2)
    e.step(2)
    assert e.state.fees == 1


def test_risk_lock_forces_flat_and_charges_once():
    p = np.full(80, 100.0, dtype=np.float64)
    p[62:] = 80
    ep = episode(p)
    opens = ep.opens.copy()
    opens[62] = 100
    ep = EpisodeData(
        ep.symbol,
        ep.session_date,
        ep.session_open,
        ep.session_close,
        ep.starts,
        ep.ends,
        ep.available_at,
        opens,
        ep.closes,
        ep.volumes,
        ep.features,
        ep.feature_names,
        ep.data_id,
    )
    e = env(ep, fee_per_fill=D(1))
    e.reset()
    e.step(2)
    assert e.state.risk_locked
    _, _, terminated, truncated, info = e.step(2)
    assert terminated and not truncated and info["projected_target"] == 0
    assert e.state.shares == 0 and e.state.fees == 2
    assert e.episode_summary()["resolved"]


def test_forced_close_and_reward_reconciliation():
    e = env(fee_per_fill=D("1.001"))
    e.reset()
    done = False
    reward = 0.0
    while not done:
        _, r, term, trunc, _ = e.step(2)
        reward += r
        done = term or trunc
        assert e.state.cash >= 0 and e.state.shares >= 0
    assert e.state.shares == 0 and e.state.fees == D("2.02")
    assert reward == pytest.approx(float((e.state.cash - D(10000)) / D(10000) * 10000))


def test_cash_and_cost_clipping_exact():
    from tradecopilot.rl.accounting import apply_fill
    from tradecopilot.rl.contracts import LedgerState, Order
    from tradecopilot.rl.execution import execute

    state = LedgerState(D(100))
    cfg = SimConfig(cost_bps=D(5), fee_per_fill=D("1.001"))
    fill, _ = execute(
        Order("buy", 60, 120, buy_notional=D(1000)),
        state,
        cfg,
        opening_price=D(100),
        known_volume=D(10000),
        execution_volume=D(10000),
        available_at=180,
    )
    assert fill is not None
    result = apply_fill(state, fill)
    assert result.cash >= 0 and result.shares > 0
    assert result.fees == D("1.01")
    assert result.cash + result.shares * D(100) == D(100) - result.fees - result.execution_drag


def test_no_network_or_keychain_callbacks(monkeypatch):
    import socket

    import keyring

    def forbidden(*args, **kwargs):
        raise AssertionError("offline simulator attempted external access")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(keyring, "get_password", forbidden)
    e = env()
    e.reset()
    e.step(2)
    e.step(0)


def test_late_initial_anchor_masks_features_and_never_backdates():
    ep = episode(late=60)
    features = ep.features.copy()
    features[60] = 999
    closes = ep.closes.copy()
    closes[60] = 999
    ep = EpisodeData(
        ep.symbol,
        ep.session_date,
        ep.session_open,
        ep.session_close,
        ep.starts,
        ep.ends,
        ep.available_at,
        ep.opens,
        closes,
        ep.volumes,
        features,
        ep.feature_names,
        ep.data_id,
    )
    e = env(ep)
    obs, _ = e.reset()
    assert (obs[:2] == 0).all()
    _, reward, term, trunc, info = e.step(2)
    assert reward == 0 and trunc and not term
    assert info["end_reason"] == "late_decision_anchor"
    assert not info["fills"]


def test_invalid_action_does_not_commit():
    e = env()
    e.reset()
    with pytest.raises(ValueError):
        e.step(3)
    assert e.state.last_committed_target_level == 0


def test_sparse_execution_has_no_free_terminal_liquidation():
    v = np.full(80, 10000.0, dtype=np.float64)
    v[74:] = 0
    p = np.full(80, 100.0, dtype=np.float64)
    p[74:] = 200
    e = env(episode(p, v))
    e.reset()
    done = False
    while not done:
        _, _, term, trunc, _ = e.step(2)
        done = term or trunc
    summary = e.episode_summary()
    assert summary["marked_return"] > 0
    assert summary["net_return"] is None
    assert not summary["valid"] and not summary["resolved"]


def test_delayed_cadence_and_marked_cap_reduction():
    p = np.full(80, 100.0, dtype=np.float64)
    p[62:] = 200
    ep = episode(p)
    opens = ep.opens.copy()
    opens[62] = 100
    ep = EpisodeData(
        ep.symbol,
        ep.session_date,
        ep.session_open,
        ep.session_close,
        ep.starts,
        ep.ends,
        ep.available_at,
        opens,
        ep.closes,
        ep.volumes,
        ep.features,
        ep.feature_names,
        ep.data_id,
    )
    e = env(ep)
    e.reset()
    _, _, _, _, info = e.step(2)
    assert info["decision_at"] == 61 * 60
    assert info["execution_at"] == 62 * 60
    assert info["next_observation_at"] == 63 * 60
    assert info["cap_exceeded_by_market"]
    e.step(1)
    assert e.state.shares * D(200) == D(500)


def test_contract_invalid_values_and_identity_boundaries():
    from tradecopilot.rl.contracts import LedgerState, Order

    for kwargs in ({"notional_cap": D(10001)}, {"daily_loss": D(10001)}):
        with pytest.raises(ValueError):
            SimConfig(**kwargs)
    assert SimConfig(capacity_fraction=D(0)).capacity_fraction == 0
    for kwargs in (
        {"cash": D(-1)},
        {"cash": D("NaN")},
        {"cash": D(100), "shares": D(-1)},
        {"cash": D(100), "cost_basis": D("Infinity")},
    ):
        with pytest.raises(ValueError):
            LedgerState(**kwargs)
    for args in (
        ("invalid", 60, 120, D(100), D(0)),
        ("buy", 60, 120, D("NaN"), D(0)),
        ("sell", 60, 120, D(100), D(1)),
        ("buy", 60, 120, D(0), D(0)),
    ):
        with pytest.raises(ValueError):
            Order(*args)
    ep = episode()
    later_close = EpisodeData(
        ep.symbol,
        ep.session_date,
        ep.session_open,
        ep.session_close + 60,
        ep.starts,
        ep.ends,
        ep.available_at,
        ep.opens,
        ep.closes,
        ep.volumes,
        ep.features,
        ep.feature_names,
        ep.data_id,
    )
    renamed_schema = EpisodeData(
        ep.symbol,
        ep.session_date,
        ep.session_open,
        ep.session_close,
        ep.starts,
        ep.ends,
        ep.available_at,
        ep.opens,
        ep.closes,
        ep.volumes,
        ep.features,
        ("other", "name"),
        ep.data_id,
    )
    assert ep.episode_id != later_close.episode_id
    assert ep.episode_id != renamed_schema.episode_id


@pytest.mark.parametrize(("latency", "buffer"), [(1, 2), (2, 3), (5, 6)])
def test_last_executable_grid_forces_flat_with_short_buffer(latency, buffer):
    e = env(latency_minutes=latency, forced_close_buffer_minutes=buffer, fee_per_fill=D(1))
    e.reset()
    done = False
    while not done:
        _, _, term, trunc, _ = e.step(2)
        done = term or trunc
    summary = e.episode_summary()
    assert e.state.shares == 0
    assert summary["resolved"] and summary["valid"]
    assert e.state.fees == D(2)


@pytest.mark.parametrize(("latency", "buffer"), [(1, 2), (2, 3), (5, 6)])
def test_last_grid_missing_liquidity_still_unresolved(latency, buffer):
    volumes = np.full(80, 10000.0, dtype=np.float64)
    volumes[70:] = 0
    e = env(episode(volumes=volumes), latency_minutes=latency, forced_close_buffer_minutes=buffer, fee_per_fill=D(1))
    e.reset()
    done = False
    while not done:
        _, _, term, trunc, _ = e.step(2)
        done = term or trunc
    summary = e.episode_summary()
    assert e.state.shares > 0
    assert not summary["resolved"] and not summary["valid"]
    assert summary["net_return"] is None
    assert e.state.fees == D(1)
