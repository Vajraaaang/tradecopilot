from __future__ import annotations

from datetime import date, time
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tradecopilot.config import StrategyConfig
from tradecopilot.journal import Journal
from tradecopilot.models import DecisionState, MarketFrame
from tradecopilot.state_machine import IllegalTransitionError, TradingStateMachine
from tradecopilot.strategy import DecisionEngine


def _risk_update(frame: MarketFrame, **updates: object) -> MarketFrame:
    assert frame.account_risk is not None
    return frame.model_copy(update={"account_risk": frame.account_risk.model_copy(update=updates)})


def test_half_profit_giveback_requires_positive_peak(day_stop_frame: MarketFrame) -> None:
    no_positive_peak = _risk_update(
        day_stop_frame,
        peak_realized_session_pnl=Decimal("0"),
        realized_session_pnl=Decimal("0"),
    )
    decision = DecisionEngine(StrategyConfig()).evaluate(no_positive_peak)
    assert decision.state != DecisionState.DAY_STOP


def test_half_profit_giveback_triggers_day_stop(day_stop_frame: MarketFrame) -> None:
    decision = DecisionEngine(StrategyConfig()).evaluate(day_stop_frame)
    assert decision.state == DecisionState.DAY_STOP
    assert "half" in decision.reasons[0].lower()


def test_maximum_daily_loss_triggers_day_stop(day_stop_frame: MarketFrame) -> None:
    frame = _risk_update(
        day_stop_frame,
        peak_realized_session_pnl=Decimal("0"),
        realized_session_pnl=Decimal("-75"),
    )
    assert DecisionEngine(StrategyConfig()).evaluate(frame).state == DecisionState.DAY_STOP


def test_three_consecutive_losses_trigger_day_stop(day_stop_frame: MarketFrame) -> None:
    frame = _risk_update(
        day_stop_frame,
        peak_realized_session_pnl=Decimal("0"),
        realized_session_pnl=Decimal("0"),
        consecutive_losses=3,
    )
    assert DecisionEngine(StrategyConfig()).evaluate(frame).state == DecisionState.DAY_STOP


def test_time_cutoff_triggers_day_stop(day_stop_frame: MarketFrame) -> None:
    risk_safe = _risk_update(
        day_stop_frame,
        peak_realized_session_pnl=Decimal("0"),
        realized_session_pnl=Decimal("0"),
    )
    config = StrategyConfig(trading_cutoff_eastern=time(9, 55))
    assert DecisionEngine(config).evaluate(risk_safe).state == DecisionState.DAY_STOP


def test_missing_account_risk_fails_closed(yxt_frames: list[MarketFrame]) -> None:
    frame = yxt_frames[0].model_copy(update={"account_risk": None})
    decision = DecisionEngine(StrategyConfig()).evaluate(frame)
    assert decision.state == DecisionState.DATA_INSUFFICIENT
    assert "account risk" in " ".join(decision.missing_data)


def test_day_stop_survives_application_restart(tmp_path) -> None:
    path = tmp_path / "journal.sqlite3"
    with Journal(path) as journal:
        journal.lock_session(date(2026, 8, 10), "risk stop", "replay")
    with Journal(path) as reopened:
        assert reopened.is_session_locked(date(2026, 8, 10))


def test_live_day_stop_cannot_be_reset(tmp_path) -> None:
    with Journal(tmp_path / "journal.sqlite3") as journal:
        journal.lock_session(date(2026, 8, 10), "risk stop", "live")
        with pytest.raises(PermissionError):
            journal.administrative_reset(date(2026, 8, 10), replay_or_test=False)


def test_illegal_state_transition_is_rejected(yxt_frames: list[MarketFrame]) -> None:
    engine = DecisionEngine(StrategyConfig())
    watch = engine.evaluate(yxt_frames[0])
    machine = TradingStateMachine()
    machine.apply(watch)
    illegal = watch.model_copy(update={"state": DecisionState.HOLD})
    with pytest.raises(IllegalTransitionError):
        machine.apply(illegal)


@given(state=st.sampled_from([state for state in DecisionState if state != DecisionState.DAY_STOP]))
def test_day_stop_is_absorbing(state: DecisionState, yxt_frames: list[MarketFrame]) -> None:
    machine = TradingStateMachine()
    base = DecisionEngine(StrategyConfig()).evaluate(yxt_frames[0])
    machine.current_state = DecisionState.DAY_STOP
    with pytest.raises(IllegalTransitionError):
        machine.apply(base.model_copy(update={"state": state}))


def test_day_stop_still_allows_structural_sell_for_open_position(yxt_frames: list[MarketFrame]) -> None:
    frame = yxt_frames[5]
    assert frame.position is not None and frame.position.quantity > 0
    decision = (
        DecisionEngine(StrategyConfig())
        .evaluate(frame)
        .model_copy(update={"state": DecisionState.SELL, "position": frame.position})
    )
    machine = TradingStateMachine()
    machine.current_state = DecisionState.DAY_STOP
    transition = machine.apply(decision)
    assert transition is not None
    assert transition.new_state == DecisionState.SELL
