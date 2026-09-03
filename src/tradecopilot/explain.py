from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from tradecopilot.models import DecisionState, SignalAvailability, StateTransition, StrategyDecision

LOGGER = logging.getLogger(__name__)
_PRICE_PATTERN = re.compile(
    r"(?:\$|(?:price|trigger|stop|level|entry|resistance)\s+(?:is|at|of)?\s*)(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


class Explanation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state: DecisionState
    one_sentence_action: str
    what_changed: str
    reasons: tuple[str, ...]
    indicator_interpretation: tuple[str, ...]
    risks: tuple[str, ...]
    what_changes_the_decision: tuple[str, ...]
    missing_data: tuple[str, ...]
    decision_expiry: datetime


class ExplanationAgent(Protocol):
    def explain(self, decision: StrategyDecision, recent_history: Sequence[StateTransition]) -> Explanation: ...


class ExplanationConflict(ValueError):
    pass


class DeterministicExplainer:
    def explain(self, decision: StrategyDecision, recent_history: Sequence[StateTransition]) -> Explanation:
        del recent_history
        action = _action(decision)
        indicator_lines: list[str] = []
        if decision.indicator is not None and decision.current_price is not None:
            indicator = decision.indicator
            indicator_lines.extend(
                (
                    f"Price {decision.current_price} versus VWAP {indicator.vwap} ({indicator.session_definition})",
                    f"EMA9 {indicator.ema9_1m}; EMA20 {indicator.ema20_1m}; "
                    f"five-minute structure {indicator.five_minute_structure}",
                    f"ATR {indicator.atr14_1m}; RSI {indicator.rsi14_1m}; MACD {indicator.macd_1m}",
                )
            )
        risks = list(decision.rules_failed)
        risks.extend(
            signal.name
            for signal in decision.exit_signals
            if signal.availability in {SignalAvailability.ACTIVE, SignalAvailability.LIMITED}
        )
        return Explanation(
            state=decision.state,
            one_sentence_action=action,
            what_changed=decision.changed_summary,
            reasons=decision.reasons,
            indicator_interpretation=tuple(indicator_lines),
            risks=tuple(dict.fromkeys(risks)),
            what_changes_the_decision=decision.what_changes_the_decision,
            missing_data=decision.missing_data
            + tuple(
                signal.name for signal in decision.exit_signals if signal.availability == SignalAvailability.UNAVAILABLE
            ),
            decision_expiry=decision.receipt_timestamp + timedelta(seconds=15),
        )


class OpenAIExplanationAgent:
    """Optional structured explainer. It receives no tools and no broker client."""

    def __init__(
        self,
        model: str,
        *,
        effort: str = "high",
        reasoning_mode: str = "pro",
        safety_identifier: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("install tradecopilot[openai] to enable OpenAI explanations") from exc
        self._client = OpenAI(timeout=180.0)
        self._model = model
        self._effort = effort
        self._reasoning_mode = reasoning_mode
        self._safety_identifier = safety_identifier

    @property
    def model_version(self) -> str:
        return self._model

    def explain(self, decision: StrategyDecision, recent_history: Sequence[StateTransition]) -> Explanation:
        sanitized = {
            "decision": decision.model_dump(mode="json"),
            "recent_state_history": [item.model_dump(mode="json") for item in recent_history[-5:]],
        }
        request: dict[str, Any] = {
            "model": self._model,
            "store": False,
            "max_output_tokens": 1_500,
            "reasoning": {"mode": self._reasoning_mode, "effort": self._effort},
            "input": [
                {
                    "role": "system",
                    "content": (
                        "Explain the supplied deterministic trading-analysis decision. "
                        "Never alter state, prices, trigger, stop, risk, quantity, or reward/risk. "
                        "Never claim unavailable Level 2 or tape evidence. BUY and SELL are analysis "
                        "signals only and execution is manual. Return only the requested structure."
                    ),
                },
                {"role": "user", "content": json.dumps(sanitized, separators=(",", ":"))},
            ],
            "text_format": Explanation,
        }
        if self._safety_identifier:
            request["safety_identifier"] = self._safety_identifier
        response = self._client.responses.parse(**request)
        if response.output_parsed is None:
            raise ExplanationConflict("OpenAI response had no parsed explanation")
        return Explanation.model_validate(response.output_parsed)


class SafeExplanationService:
    def __init__(
        self,
        agent: ExplanationAgent | None = None,
        conflict_logger: Callable[[str], None] | None = None,
    ) -> None:
        self._fallback = DeterministicExplainer()
        self._agent = agent
        self._log_conflict = conflict_logger or LOGGER.warning

    @property
    def model_version(self) -> str | None:
        return self._agent.model_version if isinstance(self._agent, OpenAIExplanationAgent) else None

    @property
    def prompt_version(self) -> str:
        return "openai-explanation-v1" if self._agent is not None else "deterministic-explanation-v1"

    def explain(self, decision: StrategyDecision, recent_history: Sequence[StateTransition]) -> Explanation:
        fallback = self._fallback.explain(decision, recent_history)
        if self._agent is None:
            return fallback
        try:
            candidate = self._agent.explain(decision, recent_history)
            validate_explanation(candidate, decision)
        except Exception as exc:
            self._log_conflict(f"Explanation rejected; deterministic fallback used: {exc}")
            return fallback
        return candidate


def validate_explanation(explanation: Explanation, decision: StrategyDecision) -> None:
    if explanation.state != decision.state:
        raise ExplanationConflict("explanation attempted to override deterministic state")
    text = " ".join(
        (
            explanation.one_sentence_action,
            explanation.what_changed,
            *explanation.reasons,
            *explanation.indicator_interpretation,
            *explanation.risks,
            *explanation.what_changes_the_decision,
        )
    )
    known_prices = _known_prices(decision)
    for matched in _PRICE_PATTERN.findall(text):
        value = Decimal(matched)
        if not any(abs(value - known) <= Decimal("0.0001") for known in known_prices):
            raise ExplanationConflict(f"explanation introduced unknown price {value}")
    lower = text.lower()
    unavailable = {
        signal.name for signal in decision.exit_signals if signal.availability == SignalAvailability.UNAVAILABLE
    }
    if "hidden_seller" in unavailable and "confirmed hidden seller" in lower:
        raise ExplanationConflict("explanation claimed unavailable hidden-seller evidence")
    if "red_tape_burst" in unavailable and "confirmed red tape" in lower:
        raise ExplanationConflict("explanation claimed unavailable tape evidence")
    if decision.state in {DecisionState.BUY, DecisionState.SELL}:
        action = explanation.one_sentence_action.lower()
        if "analysis signal" not in action or "manual" not in action:
            raise ExplanationConflict("BUY/SELL explanation omitted manual-execution boundary")


def _known_prices(decision: StrategyDecision) -> set[Decimal]:
    values = {value for value in (decision.current_price,) if value is not None}
    if decision.trade_plan is not None:
        values.update(
            {
                decision.trade_plan.trigger_price,
                decision.trade_plan.structural_stop,
                decision.trade_plan.nearest_resistance,
                decision.trade_plan.invalidation_level,
            }
        )
    if decision.indicator is not None:
        values.update(
            {
                decision.indicator.vwap,
                decision.indicator.ema9_1m,
                decision.indicator.ema20_1m,
                decision.indicator.ema9_5m,
                decision.indicator.ema20_5m,
                decision.indicator.structural_support,
                decision.indicator.structural_resistance,
            }
        )
    return values


def _action(decision: StrategyDecision) -> str:
    plan = decision.trade_plan
    actions = {
        DecisionState.DATA_STALE: "Use broker-side/manual risk controls until fresh data returns",
        DecisionState.DATA_INSUFFICIENT: "Wait; required evidence or market data is missing",
        DecisionState.NO_TRADE: "Stand aside because at least one required setup rule fails",
        DecisionState.WATCH: "Watch for a controlled first pullback; do not anticipate the trigger",
        DecisionState.ARMED: f"Wait for a fresh trade at {plan.trigger_price if plan else 'the trigger'}",
        DecisionState.BUY: "BUY analysis signal only; manual execution after verifying the displayed plan",
        DecisionState.HOLD: "HOLD while the stated thesis and structural stop remain intact",
        DecisionState.EXIT_WARNING: "Prepare to exit manually if the displayed deterioration confirms",
        DecisionState.SELL: "SELL analysis signal only; manual execution because the thesis is invalid",
        DecisionState.REENTRY_WATCH: "Remain flat until the fresh setup reaches its new strict trigger",
        DecisionState.DAY_STOP: "Stop taking new trades for this market session",
    }
    return f"{actions[decision.state]}; analysis signal only; manual execution"
