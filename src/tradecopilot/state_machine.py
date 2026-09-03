from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from tradecopilot.models import DecisionState, StateTransition, StrategyDecision


class IllegalTransitionError(ValueError):
    pass


_ALLOWED: dict[DecisionState, frozenset[DecisionState]] = {
    DecisionState.DATA_STALE: frozenset(
        {
            DecisionState.DATA_STALE,
            DecisionState.DATA_INSUFFICIENT,
            DecisionState.NO_TRADE,
            DecisionState.WATCH,
            DecisionState.ARMED,
            DecisionState.REENTRY_WATCH,
            DecisionState.HOLD,
            DecisionState.EXIT_WARNING,
            DecisionState.SELL,
            DecisionState.DAY_STOP,
        }
    ),
    DecisionState.DATA_INSUFFICIENT: frozenset(
        {
            DecisionState.DATA_INSUFFICIENT,
            DecisionState.DATA_STALE,
            DecisionState.NO_TRADE,
            DecisionState.WATCH,
            DecisionState.ARMED,
            DecisionState.REENTRY_WATCH,
            DecisionState.HOLD,
            DecisionState.SELL,
            DecisionState.DAY_STOP,
        }
    ),
    DecisionState.NO_TRADE: frozenset(
        {
            DecisionState.NO_TRADE,
            DecisionState.DATA_STALE,
            DecisionState.DATA_INSUFFICIENT,
            DecisionState.WATCH,
            DecisionState.ARMED,
            DecisionState.REENTRY_WATCH,
            DecisionState.DAY_STOP,
        }
    ),
    DecisionState.WATCH: frozenset(
        {
            DecisionState.WATCH,
            DecisionState.ARMED,
            DecisionState.REENTRY_WATCH,
            DecisionState.NO_TRADE,
            DecisionState.DATA_STALE,
            DecisionState.DATA_INSUFFICIENT,
            DecisionState.DAY_STOP,
        }
    ),
    DecisionState.ARMED: frozenset(
        {
            DecisionState.ARMED,
            DecisionState.BUY,
            DecisionState.WATCH,
            DecisionState.NO_TRADE,
            DecisionState.DATA_STALE,
            DecisionState.DATA_INSUFFICIENT,
            DecisionState.DAY_STOP,
        }
    ),
    DecisionState.BUY: frozenset(
        {
            DecisionState.BUY,
            DecisionState.ARMED,
            DecisionState.HOLD,
            DecisionState.EXIT_WARNING,
            DecisionState.SELL,
            DecisionState.DATA_STALE,
            DecisionState.DATA_INSUFFICIENT,
            DecisionState.DAY_STOP,
        }
    ),
    DecisionState.HOLD: frozenset(
        {
            DecisionState.HOLD,
            DecisionState.EXIT_WARNING,
            DecisionState.SELL,
            DecisionState.DATA_STALE,
            DecisionState.DATA_INSUFFICIENT,
            DecisionState.DAY_STOP,
        }
    ),
    DecisionState.EXIT_WARNING: frozenset(
        {
            DecisionState.EXIT_WARNING,
            DecisionState.HOLD,
            DecisionState.SELL,
            DecisionState.DATA_STALE,
            DecisionState.DATA_INSUFFICIENT,
            DecisionState.DAY_STOP,
        }
    ),
    DecisionState.SELL: frozenset(
        {
            DecisionState.SELL,
            DecisionState.REENTRY_WATCH,
            DecisionState.WATCH,
            DecisionState.NO_TRADE,
            DecisionState.DATA_STALE,
            DecisionState.DATA_INSUFFICIENT,
            DecisionState.DAY_STOP,
        }
    ),
    DecisionState.REENTRY_WATCH: frozenset(
        {
            DecisionState.REENTRY_WATCH,
            DecisionState.BUY,
            DecisionState.WATCH,
            DecisionState.ARMED,
            DecisionState.NO_TRADE,
            DecisionState.DATA_STALE,
            DecisionState.DATA_INSUFFICIENT,
            DecisionState.DAY_STOP,
        }
    ),
    DecisionState.DAY_STOP: frozenset({DecisionState.DAY_STOP}),
}


class TradingStateMachine:
    def __init__(self) -> None:
        self.current_state: DecisionState | None = None

    def apply(self, decision: StrategyDecision) -> StateTransition | None:
        prior = self.current_state
        exposed_safety_exit = (
            prior == DecisionState.DAY_STOP
            and decision.state == DecisionState.SELL
            and decision.position is not None
            and decision.position.quantity > 0
        )
        if prior is not None and decision.state not in _ALLOWED[prior] and not exposed_safety_exit:
            raise IllegalTransitionError(f"illegal transition: {prior} -> {decision.state}")
        self.current_state = decision.state
        if prior == decision.state:
            return None
        plan = decision.trade_plan
        return StateTransition(
            provider_timestamp=decision.provider_timestamp,
            receipt_timestamp=decision.receipt_timestamp,
            age_seconds=decision.age_seconds,
            source="deterministic_state_machine",
            quality=decision.quality,
            symbol=decision.symbol,
            prior_state=prior,
            new_state=decision.state,
            strategy_version=decision.strategy_version,
            exact_reasons=decision.reasons,
            rules_passed=decision.rules_passed,
            rules_failed=decision.rules_failed,
            source_data_timestamps=decision.source_data_timestamps,
            invalidation_level=plan.invalidation_level if plan else None,
            trigger_level=plan.trigger_price if plan else None,
            active_position=decision.position,
        )

    def administrative_reset(self, *, replay_or_test: bool, timestamp: datetime) -> None:
        del timestamp
        if not replay_or_test:
            raise PermissionError("live DAY_STOP locks cannot be administratively reset")
        self.current_state = None


def assert_stop_not_widened(original: Decimal, current: Decimal) -> None:
    if current < original:
        raise ValueError("structural stop cannot be widened after entry")
