from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from tradecopilot.config import StrategyConfig
from tradecopilot.features import ImpulseDetector, PullbackDetector
from tradecopilot.indicators import calculate_indicators
from tradecopilot.models import DecisionState, MarketFrame
from tradecopilot.state_machine import TradingStateMachine
from tradecopilot.strategy import DecisionEngine


def _evaluate(frames: list[MarketFrame]) -> list:
    engine = DecisionEngine(StrategyConfig())
    return [engine.evaluate(frame) for frame in frames]


def _quote_at(frame: MarketFrame, price: Decimal, *, age: float | None = None) -> MarketFrame:
    assert frame.quote is not None
    quote = frame.quote.model_copy(
        update={
            "last": price,
            "bid": price - Decimal("0.01"),
            "ask": price + Decimal("0.01"),
            **(
                {
                    "provider_timestamp": frame.event_time - timedelta(seconds=age),
                    "receipt_timestamp": frame.event_time,
                    "age_seconds": age,
                }
                if age is not None
                else {}
            ),
        }
    )
    return frame.model_copy(update={"quote": quote})


def _pillar(frame: MarketFrame, name: str):
    assert frame.quote is not None
    engine = DecisionEngine(StrategyConfig())
    indicator = calculate_indicators(
        frame.quote,
        frame.bars_1m,
        frame.bars_5m,
        frame.resistance_levels,
        engine.config,
    )
    pillars, missing = engine._pillars(frame, indicator)
    assert not missing
    return next(pillar for pillar in pillars if pillar.name == name)


def test_four_of_five_pillars_passes(yxt_frames: list[MarketFrame]) -> None:
    frame = yxt_frames[2].model_copy(update={"catalyst_evidence": None, "market_leader": False})
    assert frame.quote is not None
    engine = DecisionEngine(StrategyConfig())
    indicator = calculate_indicators(frame.quote, frame.bars_1m, frame.bars_5m, frame.resistance_levels, engine.config)
    pillars, missing = engine._pillars(frame, indicator)
    assert not missing
    assert sum(pillar.passed for pillar in pillars) == 4


def test_verified_float_is_mandatory(yxt_frames: list[MarketFrame]) -> None:
    frame = yxt_frames[0].model_copy(update={"float_evidence": None})
    decision = DecisionEngine(StrategyConfig()).evaluate(frame)
    assert decision.state == DecisionState.DATA_INSUFFICIENT
    assert "float" in " ".join(decision.missing_data).lower()


def test_catalyst_exception_requires_obvious_market_leader(yxt_frames: list[MarketFrame]) -> None:
    original = yxt_frames[0]
    assert original.quote is not None
    quote = original.quote.model_copy(update={"total_volume": 12_000_000, "last": Decimal("9.30")})
    not_leader = original.model_copy(update={"quote": quote, "catalyst_evidence": None, "market_leader": False})
    leader = not_leader.model_copy(update={"market_leader": True})
    assert not _pillar(not_leader, "catalyst").passed
    assert _pillar(leader, "catalyst").passed


def test_price_origin_rule_allows_current_price_above_twenty(yxt_frames: list[MarketFrame]) -> None:
    frame = _quote_at(yxt_frames[0], Decimal("25"))
    assert _pillar(frame, "price").passed


def test_exactly_fifty_percent_retracement_passes(yxt_frames: list[MarketFrame]) -> None:
    config = StrategyConfig()
    bars = list(yxt_frames[2].bars_1m)
    impulse = ImpulseDetector(config).detect(bars)
    assert impulse is not None
    target = impulse.high - (impulse.high - impulse.low) * Decimal("0.50")
    bars[-2] = bars[-2].model_copy(update={"low": target + Decimal("0.01")})
    bars[-1] = bars[-1].model_copy(update={"low": target})
    pullback = PullbackDetector(config).detect(bars, impulse)
    assert pullback is not None
    assert pullback.retracement_fraction == Decimal("0.50")
    assert pullback.valid


def test_greater_than_fifty_percent_retracement_fails(yxt_frames: list[MarketFrame]) -> None:
    config = StrategyConfig()
    bars = list(yxt_frames[2].bars_1m)
    impulse = ImpulseDetector(config).detect(bars)
    assert impulse is not None
    target = impulse.high - (impulse.high - impulse.low) * Decimal("0.501")
    bars[-2] = bars[-2].model_copy(update={"low": target + Decimal("0.01")})
    bars[-1] = bars[-1].model_copy(update={"low": target})
    pullback = PullbackDetector(config).detect(bars, impulse)
    assert pullback is not None
    assert pullback.retracement_fraction > Decimal("0.50")
    assert not pullback.valid


def test_green_impulse_volume_must_exceed_red_pullback_volume(
    yxt_frames: list[MarketFrame],
) -> None:
    config = StrategyConfig()
    bars = list(yxt_frames[2].bars_1m)
    bars[-2] = bars[-2].model_copy(update={"volume": 2_000_000})
    bars[-1] = bars[-1].model_copy(update={"volume": 2_000_000})
    impulse = ImpulseDetector(config).detect(bars)
    pullback = PullbackDetector(config).detect(bars, impulse)
    assert impulse is not None and pullback is not None
    assert pullback.average_red_volume >= impulse.average_green_volume
    assert not pullback.valid


def test_vwap_loss_invalidates_setup(yxt_frames: list[MarketFrame]) -> None:
    baseline = _evaluate(yxt_frames[:3])[-1]
    assert baseline.indicator is not None
    frame = _quote_at(yxt_frames[2], baseline.indicator.vwap - Decimal("0.01"))
    engine = DecisionEngine(StrategyConfig())
    engine.evaluate(yxt_frames[0])
    engine.evaluate(yxt_frames[1])
    decision = engine.evaluate(frame)
    assert decision.state == DecisionState.NO_TRADE
    assert "vwap" in " ".join(decision.reasons).lower()


def test_ema9_loss_invalidates_setup(yxt_frames: list[MarketFrame]) -> None:
    baseline = _evaluate(yxt_frames[:3])[-1]
    assert baseline.indicator is not None
    assert baseline.indicator.ema9_1m > baseline.indicator.vwap
    price = (baseline.indicator.ema9_1m + baseline.indicator.vwap) / Decimal(2)
    frame = _quote_at(yxt_frames[2], price)
    engine = DecisionEngine(StrategyConfig())
    engine.evaluate(yxt_frames[0])
    engine.evaluate(yxt_frames[1])
    decision = engine.evaluate(frame)
    assert decision.state == DecisionState.NO_TRADE
    assert "ema9" in " ".join(decision.reasons).lower()


def test_trigger_does_not_fire_before_previous_candle_high(
    yxt_frames: list[MarketFrame],
) -> None:
    decision = _evaluate(yxt_frames[:3])[-1]
    assert decision.state == DecisionState.ARMED
    assert decision.trade_plan is not None
    assert decision.current_price < decision.trade_plan.trigger_price


def test_trigger_fires_on_first_actual_cross(yxt_frames: list[MarketFrame]) -> None:
    decisions = _evaluate(yxt_frames[:5])
    assert decisions[-2].state == DecisionState.ARMED
    assert decisions[-1].state == DecisionState.BUY
    assert decisions[-1].crossing_fresh


def test_stale_crossing_cannot_produce_buy(yxt_frames: list[MarketFrame]) -> None:
    engine = DecisionEngine(StrategyConfig())
    for frame in yxt_frames[:4]:
        engine.evaluate(frame)
    stale_cross = _quote_at(yxt_frames[4], Decimal("9.15"), age=2.01)
    decision = engine.evaluate(stale_cross)
    assert decision.state == DecisionState.DATA_STALE


def test_duplicate_quote_timestamp_cannot_produce_buy(yxt_frames: list[MarketFrame]) -> None:
    engine = DecisionEngine(StrategyConfig())
    for frame in yxt_frames[:4]:
        engine.evaluate(frame)
    assert yxt_frames[3].quote is not None and yxt_frames[4].quote is not None
    duplicate_quote = yxt_frames[4].quote.model_copy(
        update={"provider_timestamp": yxt_frames[3].quote.provider_timestamp}
    )
    decision = engine.evaluate(yxt_frames[4].model_copy(update={"quote": duplicate_quote}))
    assert decision.state != DecisionState.BUY
    assert not decision.crossing_fresh


def test_less_than_two_r_cannot_produce_buy(yxt_frames: list[MarketFrame]) -> None:
    low_reward = yxt_frames[2].model_copy(update={"resistance_levels": (Decimal("9.20"),)})
    engine = DecisionEngine(StrategyConfig())
    engine.evaluate(yxt_frames[0])
    engine.evaluate(yxt_frames[1])
    decision = engine.evaluate(low_reward)
    assert decision.state == DecisionState.NO_TRADE
    assert "reward-to-risk" in " ".join(decision.reasons)


def test_structural_stop_produces_sell(yxt_frames: list[MarketFrame]) -> None:
    engine = DecisionEngine(StrategyConfig())
    for frame in yxt_frames[:6]:
        engine.evaluate(frame)
    at_stop = _quote_at(yxt_frames[5], Decimal("8.89"))
    decision = engine.evaluate(at_stop)
    assert decision.state == DecisionState.SELL
    assert "Structural stop" in decision.reasons[0]


def test_failed_breakout_produces_sell(yxt_frames: list[MarketFrame]) -> None:
    engine = DecisionEngine(StrategyConfig())
    for frame in yxt_frames[:6]:
        engine.evaluate(frame)
    failed = _quote_at(yxt_frames[5], Decimal("9.05"))
    decision = engine.evaluate(failed)
    assert decision.state == DecisionState.SELL
    assert "failed to reclaim" in decision.reasons[0]


def test_buying_slowdown_warns_before_confirmation(yxt_frames: list[MarketFrame]) -> None:
    decisions = _evaluate(yxt_frames[:8])
    assert decisions[-1].state == DecisionState.EXIT_WARNING
    assert "slowing" in " ".join(decisions[-1].reasons).lower()


def test_hold_requires_explicit_intact_thesis_evidence(yxt_frames: list[MarketFrame]) -> None:
    decision = _evaluate(yxt_frames[:6])[-1]
    assert decision.state == DecisionState.HOLD
    reasons = " ".join(decision.reasons).lower()
    assert "structural stop" in reasons
    assert "higher low" in reasons
    assert "no confirmed exit" in reasons


def test_reentry_requires_complete_new_setup(yxt_frames: list[MarketFrame]) -> None:
    decisions = _evaluate(yxt_frames)
    assert decisions[-2].state == DecisionState.SELL
    assert decisions[-1].state == DecisionState.REENTRY_WATCH
    assert decisions[-1].trade_plan is not None
    assert decisions[-1].pullback is not None and decisions[-1].pullback.valid


def test_reentry_trigger_can_produce_fresh_buy(yxt_frames: list[MarketFrame]) -> None:
    engine = DecisionEngine(StrategyConfig())
    machine = TradingStateMachine()
    decisions = []
    for frame in yxt_frames:
        decision = engine.evaluate(frame)
        machine.apply(decision)
        decisions.append(decision)
    reentry = decisions[-1]
    assert reentry.state == DecisionState.REENTRY_WATCH
    assert reentry.trade_plan is not None
    crossed = _quote_at(yxt_frames[-1], reentry.trade_plan.trigger_price)
    assert crossed.quote is not None
    next_time = crossed.event_time + timedelta(seconds=1)
    crossed = crossed.model_copy(
        update={
            "event_time": next_time,
            "quote": crossed.quote.model_copy(update={"provider_timestamp": next_time, "receipt_timestamp": next_time}),
        }
    )
    decision = engine.evaluate(crossed)
    machine.apply(decision)
    assert decision.state == DecisionState.BUY
    assert decision.crossing_fresh


def test_structural_exit_has_priority_over_day_stop(yxt_frames: list[MarketFrame]) -> None:
    engine = DecisionEngine(StrategyConfig())
    for frame in yxt_frames[:6]:
        engine.evaluate(frame)
    frame = _quote_at(yxt_frames[5], Decimal("8.89"))
    assert frame.account_risk is not None
    frame = frame.model_copy(update={"account_risk": frame.account_risk.model_copy(update={"session_locked": True})})
    decision = engine.evaluate(frame)
    assert decision.state == DecisionState.SELL
    assert "Structural stop" in decision.reasons[0]


def test_stale_data_with_open_position_never_returns_hold(
    yxt_frames: list[MarketFrame],
) -> None:
    engine = DecisionEngine(StrategyConfig())
    for frame in yxt_frames[:6]:
        engine.evaluate(frame)
    stale = _quote_at(yxt_frames[5], Decimal("9.20"), age=2.01)
    decision = engine.evaluate(stale)
    assert decision.state == DecisionState.DATA_STALE
    assert "broker-side/manual" in " ".join(decision.what_changes_the_decision).lower()
