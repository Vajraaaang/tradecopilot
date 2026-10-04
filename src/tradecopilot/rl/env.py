"""Causal fixed-grid Gymnasium adapter over immutable local arrays."""

from collections.abc import Sequence
from dataclasses import asdict, replace
from decimal import Decimal
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from .accounting import apply_fill, mark_to_market
from .contracts import EpisodeData, LedgerState, Order, SimConfig
from .execution import execute

OBS_PORTFOLIO_NAMES = (
    "cash_fraction",
    "equity_fraction",
    "position_cap_fraction",
    "average_cost_relative",
    "current_drawdown",
    "target_flat",
    "target_half",
    "target_full",
    "risk_locked",
    "time_to_close_fraction",
    "last_fill_fraction",
    "last_rejection",
)


def _d(value: float | np.float64) -> Decimal:
    return Decimal(str(float(value)))


def _json(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {k: _json(v) for k, v in value.items()}
    return value


class TradeCopilotEnv(gym.Env[NDArray[np.float32], int]):
    """Actions commit flat/half/full; repeats hold except mandatory reductions."""

    def __init__(self, episodes: Sequence[EpisodeData], config: SimConfig):
        super().__init__()
        self.metadata = {"render_modes": []}
        if not episodes or len({e.episode_id for e in episodes}) != len(episodes):
            raise ValueError("episodes must be nonempty and uniquely identified")
        names = episodes[0].feature_names
        if any(e.feature_names != names for e in episodes):
            raise ValueError("inconsistent feature schema")
        self.episodes = tuple(episodes)
        self.config = config
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(len(names) + len(OBS_PORTFOLIO_NAMES),), dtype=np.float32
        )
        self.episode = self.episodes[0]
        self.state = LedgerState(config.initial_cash, peak_equity=config.initial_cash)
        self.ledger: list[dict[str, Any]] = []
        self.index = 0
        self.last_fill_fraction = 0.0
        self.last_rejection = False
        self.valid = True
        self.resolved = False
        self.done = False
        self.end_reason = ""
        self.step_count = 0
        self.turnover = Decimal(0)
        self.reward_sum = 0.0

    @property
    def current_market_features(self) -> NDArray[np.float32]:
        if self.episode.available_at[self.index] > self.decision_at:
            return np.zeros(len(self.episode.feature_names), dtype=np.float32)
        return np.array(self.episode.features[self.index], dtype=np.float32, copy=True)

    def _safe_mark_price(self) -> Decimal:
        available = np.flatnonzero(self.episode.available_at[: self.index + 1] <= self.decision_at)
        # No mark is consumed before availability; at reset no shares are held.
        return _d(self.episode.closes[int(available[-1])]) if len(available) else Decimal(1)

    @property
    def decision_at(self) -> int:
        return int(self.episode.ends[self.index])

    def _observation(self) -> NDArray[np.float32]:
        price = self._safe_mark_price()
        equity = self.state.equity(price)
        peak = self.state.peak_equity
        average = self.state.cost_basis / self.state.shares if self.state.shares else price
        portfolio = np.asarray(
            [
                float(self.state.cash / self.config.initial_cash),
                float(equity / self.config.initial_cash),
                float(self.state.shares * price / self.config.notional_cap),
                float(average / price - 1),
                float((peak - equity) / peak) if peak else 0,
                *[float(self.state.last_committed_target_level == level) for level in range(3)],
                float(self.state.risk_locked),
                (self.episode.session_close - self.decision_at) / 60 / 390,
                self.last_fill_fraction,
                float(self.last_rejection),
            ],
            dtype=np.float32,
        )
        obs = np.concatenate((self.current_market_features, portfolio))
        if not np.isfinite(obs).all():
            raise ValueError("nonfinite observation")
        return obs

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        episode_id = (options or {}).get("episode_id")
        if episode_id is None:
            self.episode = self.episodes[int(self.np_random.integers(len(self.episodes)))]
        else:
            selected = [e for e in self.episodes if e.episode_id == episode_id]
            if not selected:
                raise ValueError("unknown episode_id")
            self.episode = selected[0]
        eligible = np.flatnonzero(
            self.episode.ends >= self.episode.session_open + (self.config.warmup_minutes + 1) * 60
        )
        if len(eligible) == 0:
            raise ValueError("episode lacks warmup")
        self.index = int(eligible[0])
        self.state = LedgerState(self.config.initial_cash, peak_equity=self.config.initial_cash)
        self.ledger = []
        self.last_fill_fraction = 0.0
        self.last_rejection = False
        self.valid = True
        self.resolved = False
        self.done = False
        self.end_reason = ""
        self.step_count = 0
        self.turnover = Decimal(0)
        self.reward_sum = 0.0
        return self._observation(), {"episode_id": self.episode.episode_id, "decision_at": self.decision_at}

    def _info(
        self,
        *,
        requested: int,
        projected: int,
        fills: list[dict[str, Any]],
        rejections: list[str],
        previous: Decimal,
        reward: float,
        stale: bool = False,
    ) -> dict[str, Any]:
        price = self._safe_mark_price()
        return {
            "episode_id": self.episode.episode_id,
            "decision_at": self.decision_at,
            "requested_target": requested,
            "projected_target": projected,
            "last_committed_target_level": self.state.last_committed_target_level,
            "fills": fills,
            "rejections": rejections,
            "current_equity": str(self.state.equity(price)),
            "shares": str(self.state.shares),
            "cash": str(self.state.cash),
            "fees": str(self.state.fees),
            "execution_drag": str(self.state.execution_drag),
            "marked_exposure": str(self.state.shares * price),
            "cap_exceeded_by_market": self.state.shares * price > self.config.notional_cap,
            "risk_locked": self.state.risk_locked,
            "valid": self.valid,
            "resolved": self.resolved,
            "end_reason": self.end_reason,
            "stale_mark": stale,
            "reward_components": {
                "previous_equity": str(previous),
                "next_equity": str(self.state.equity(price)),
                "net_equity_bps": reward,
            },
        }

    def _truncate(
        self, action: int, previous: Decimal, reason: str
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        self.valid = False
        self.resolved = False
        self.done = True
        self.end_reason = reason
        self.last_rejection = True
        self.step_count += 1
        info = self._info(
            requested=action,
            projected=self.state.last_committed_target_level,
            fills=[],
            rejections=[reason],
            previous=previous,
            reward=0,
            stale=True,
        )
        self.ledger.append(info)
        return self._observation(), 0.0, False, True, info

    def step(self, action: int) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("reset required after episode end")
        if not self.action_space.contains(action):
            raise ValueError("action must be 0, 1 or 2")
        action = int(action)
        ep = self.episode
        t = self.decision_at
        price = self._safe_mark_price()
        previous = self.state.equity(price)
        if ep.available_at[self.index] > t:
            return self._truncate(action, previous, "late_decision_anchor")
        execution_at = t + self.config.latency_minutes * 60
        transition_at = execution_at + 60
        if transition_at > ep.session_close:
            self.done = True
            self.resolved = self.state.shares == 0
            self.valid = self.valid and self.resolved
            self.end_reason = "session_complete" if self.resolved else "unresolved_session_close"
            info = self._info(requested=action, projected=0, fills=[], rejections=[], previous=previous, reward=0)
            return self._observation(), 0.0, True, False, info
        matches = np.flatnonzero(ep.starts == execution_at)
        if len(matches) != 1:
            return self._truncate(action, previous, "missing_execution_bar")
        next_index = int(matches[0])
        if ep.available_at[next_index] > transition_at:
            return self._truncate(action, previous, "late_execution_report")
        locked = self.state.risk_locked or self.config.initial_cash - previous >= self.config.daily_loss
        cadence_seconds = (self.config.latency_minutes + 1) * 60
        forced = (
            t >= ep.session_close - self.config.forced_close_buffer_minutes * 60
            or t + 2 * cadence_seconds > ep.session_close
        )
        projected = 0 if locked or forced else action
        self.state = replace(self.state, risk_locked=locked)
        target = self.config.notional_cap * Decimal(projected) / 2
        marked = self.state.shares * price
        order: Order | None = None
        # Commitment is updated even when capacity later rejects or partially fills.
        if projected == 0 and self.state.shares > 0:
            order = Order("sell", t, execution_at, sell_shares=self.state.shares)
        elif marked > self.config.notional_cap:
            reduction_target = (
                target if projected != self.state.last_committed_target_level else self.config.notional_cap
            )
            order = Order("sell", t, execution_at, sell_shares=(marked - reduction_target) / price)
        elif projected != self.state.last_committed_target_level:
            if target > marked:
                dollars = min(target - marked, max(Decimal(0), self.state.cash - self.config.fee_per_fill))
                if dollars > 0:
                    order = Order("buy", t, execution_at, buy_notional=dollars)
            elif target < marked:
                order = Order("sell", t, execution_at, sell_shares=(marked - target) / price)
        self.state = replace(self.state, last_committed_target_level=projected)
        fills: list[dict[str, Any]] = []
        rejections: list[str] = []
        self.last_fill_fraction = 0.0
        prior_fees, prior_drag = self.state.fees, self.state.execution_drag
        if order is not None:
            fill, rejection = execute(
                order,
                self.state,
                self.config,
                opening_price=_d(ep.opens[next_index]),
                known_volume=_d(ep.volumes[self.index]),
                execution_volume=_d(ep.volumes[next_index]),
                available_at=int(ep.available_at[next_index]),
            )
            if rejection:
                rejections.append(rejection)
            if fill:
                self.state = apply_fill(self.state, fill)
                fills.append(_json(asdict(fill)))
                self.turnover += fill.quantity * fill.price
                denominator = order.buy_notional / fill.price if order.side == "buy" else order.sell_shares
                self.last_fill_fraction = float(fill.quantity / denominator)
        self.last_rejection = bool(rejections)
        self.index = next_index
        next_price = self._safe_mark_price()
        self.state = mark_to_market(self.state, next_price)
        equity = self.state.equity(next_price)
        if self.config.initial_cash - equity >= self.config.daily_loss:
            self.state = replace(self.state, risk_locked=True)
        reward = float(10000 * (equity - previous) / self.config.initial_cash)
        self.reward_sum += reward
        self.step_count += 1
        last_window = self.decision_at + (self.config.latency_minutes + 1) * 60 > ep.session_close
        terminated = last_window or (self.state.risk_locked and self.state.shares == 0)
        if terminated:
            self.done = True
            self.resolved = self.state.shares == 0
            self.valid = self.valid and self.resolved
            self.end_reason = (
                ("risk_complete" if self.state.risk_locked else "session_complete")
                if (self.resolved)
                else "unresolved_session_close"
            )
        info = self._info(
            requested=action, projected=projected, fills=fills, rejections=rejections, previous=previous, reward=reward
        )
        info.update(
            {
                "anchor_end_at": t,
                "decision_at": t,
                "execution_at": execution_at,
                "next_observation_at": transition_at,
                "fill_report_available_at": int(ep.available_at[next_index]),
            }
        )
        fee_delta, drag_delta = self.state.fees - prior_fees, self.state.execution_drag - prior_drag
        info["reward_components"].update(
            {
                "fees_delta": str(fee_delta),
                "execution_drag_delta": str(drag_delta),
                "gross_equity_change": str(equity - previous + fee_delta + drag_delta),
            }
        )
        if order:
            info["order"] = _json(asdict(order))
        self.ledger.append(info)
        return self._observation(), reward, terminated, False, info

    def episode_summary(self) -> dict[str, Any]:
        equity = self.state.equity(self._safe_mark_price())
        marked_return = float((equity - self.config.initial_cash) / self.config.initial_cash)
        return {
            "episode_id": self.episode.episode_id,
            "symbol": self.episode.symbol,
            "session_date": self.episode.session_date.isoformat(),
            "config_hash": self.config.content_hash,
            "valid": self.valid and self.resolved,
            "resolved": self.resolved,
            "end_reason": self.end_reason,
            "step_count": self.step_count,
            "net_return": marked_return if self.valid and self.resolved else None,
            "marked_return": marked_return,
            "final_equity": str(equity),
            "max_drawdown": float(self.state.max_drawdown),
            "turnover": str(self.turnover),
            "fees": str(self.state.fees),
            "execution_drag": str(self.state.execution_drag),
            "reward_sum_bps": self.reward_sum,
            "trades": [fill for row in self.ledger for fill in row["fills"]],
            "rejections": [reason for row in self.ledger for reason in row["rejections"]],
            "ledger_state": _json(asdict(self.state)),
        }
