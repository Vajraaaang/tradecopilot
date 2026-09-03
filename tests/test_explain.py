from __future__ import annotations

from datetime import timedelta

import pytest

from tradecopilot.config import StrategyConfig
from tradecopilot.explain import (
    DeterministicExplainer,
    Explanation,
    ExplanationConflict,
    SafeExplanationService,
    validate_explanation,
)
from tradecopilot.models import DecisionState, MarketFrame
from tradecopilot.strategy import DecisionEngine


def _buy_decision(frames: list[MarketFrame]):
    engine = DecisionEngine(StrategyConfig())
    decisions = [engine.evaluate(frame) for frame in frames[:5]]
    assert decisions[-1].state == DecisionState.BUY
    return decisions[-1]


class FixedAgent:
    def __init__(self, explanation: Explanation) -> None:
        self.explanation = explanation

    def explain(self, decision, recent_history):
        del decision, recent_history
        return self.explanation


def test_llm_explanation_cannot_override_deterministic_state(
    yxt_frames: list[MarketFrame],
) -> None:
    decision = _buy_decision(yxt_frames)
    fallback = DeterministicExplainer().explain(decision, ())
    malicious = fallback.model_copy(update={"state": DecisionState.SELL})
    conflicts: list[str] = []
    result = SafeExplanationService(FixedAgent(malicious), conflicts.append).explain(decision, ())
    assert result.state == DecisionState.BUY
    assert conflicts


def test_llm_explanation_cannot_invent_price(yxt_frames: list[MarketFrame]) -> None:
    decision = _buy_decision(yxt_frames)
    fallback = DeterministicExplainer().explain(decision, ())
    invented = fallback.model_copy(update={"what_changed": "A new price 99.99 appeared"})
    with pytest.raises(ExplanationConflict, match="unknown price"):
        validate_explanation(invented, decision)


def test_llm_cannot_claim_unavailable_hidden_seller(yxt_frames: list[MarketFrame]) -> None:
    engine = DecisionEngine(StrategyConfig())
    decision = engine.evaluate(yxt_frames[0])
    fallback = DeterministicExplainer().explain(decision, ())
    invented = fallback.model_copy(update={"reasons": ("Confirmed hidden seller",)})
    with pytest.raises(ExplanationConflict, match="hidden-seller"):
        validate_explanation(invented, decision)


def test_buy_explanation_requires_manual_analysis_signal_language(
    yxt_frames: list[MarketFrame],
) -> None:
    decision = _buy_decision(yxt_frames)
    fallback = DeterministicExplainer().explain(decision, ())
    unsafe = fallback.model_copy(update={"one_sentence_action": "Buy now"})
    with pytest.raises(ExplanationConflict, match="manual-execution"):
        validate_explanation(unsafe, decision)


def test_deterministic_explanation_has_expiry(yxt_frames: list[MarketFrame]) -> None:
    decision = _buy_decision(yxt_frames)
    explanation = DeterministicExplainer().explain(decision, ())
    assert explanation.decision_expiry == decision.receipt_timestamp + timedelta(seconds=15)
