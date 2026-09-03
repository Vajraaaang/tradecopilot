from __future__ import annotations

import json
import re
from collections.abc import Mapping
from decimal import Decimal
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from tradecopilot.models import DecisionState

_PRICE_PATTERN = re.compile(
    r"(?:\$|(?:price|trigger|stop|level|entry|resistance)\s+(?:is|at|of)?\s*)(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


class ChatAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: str
    state: DecisionState
    manual_execution: Literal[True]


class TradeChatAgent(Protocol):
    @property
    def model_version(self) -> str: ...

    @property
    def reasoning_label(self) -> str: ...

    def answer(self, query: str, context: Mapping[str, Any]) -> ChatAnswer: ...


class OpenAITradeChatAgent:
    """Tool-free GPT chat over a sanitized deterministic decision only."""

    def __init__(
        self,
        model: str = "gpt-5.6",
        *,
        effort: str = "high",
        reasoning_mode: str = "pro",
        safety_identifier: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("install tradecopilot[openai] to enable GPT chat") from exc
        self._client = OpenAI(timeout=180.0)
        self._model = model
        self._effort = effort
        self._reasoning_mode = reasoning_mode
        self._safety_identifier = safety_identifier

    @property
    def model_version(self) -> str:
        return self._model

    @property
    def reasoning_label(self) -> str:
        return f"{self._reasoning_mode} · {self._effort}"

    def answer(self, query: str, context: Mapping[str, Any]) -> ChatAnswer:
        request: dict[str, Any] = {
            "model": self._model,
            "store": False,
            "max_output_tokens": 1_200,
            "reasoning": {"mode": self._reasoning_mode, "effort": self._effort},
            "input": [
                {
                    "role": "system",
                    "content": (
                        "Answer questions about the supplied deterministic trading-analysis decision. "
                        "The deterministic engine is authoritative: never change its state, trigger, stop, "
                        "risk, size, reward/risk, freshness result, or session lock. Do not invent prices or "
                        "claim unavailable Level 2 or tape evidence. Never imply guaranteed profit. Never "
                        "place, review, modify, or cancel an order. BUY and SELL are analysis signals only; "
                        "all execution is manual. Use only the supplied sanitized context and return the "
                        "requested structured answer."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({"question": query, "context": context}, separators=(",", ":")),
                },
            ],
            "text_format": ChatAnswer,
        }
        if self._safety_identifier:
            request["safety_identifier"] = self._safety_identifier
        response = self._client.responses.parse(**request)
        if response.output_parsed is None:
            raise ValueError("OpenAI response had no parsed chat answer")
        return ChatAnswer.model_validate(response.output_parsed)


def sanitized_chat_context(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Exclude positions, account risk, broker fields, and every tool surface."""

    return {
        "meta": snapshot.get("meta"),
        "quote": snapshot.get("quote"),
        "action": snapshot.get("action"),
        "what_changed": snapshot.get("changed"),
        "reasons": snapshot.get("reasons"),
        "what_changes_the_decision": snapshot.get("next"),
        "missing_data": snapshot.get("missing"),
        "quality": snapshot.get("quality"),
        "indicators": snapshot.get("indicators"),
        "trade_plan": snapshot.get("plan"),
        "pattern": snapshot.get("pattern"),
        "level2": snapshot.get("level2"),
        "recent_state_history": snapshot.get("history"),
        "safety": "Analysis only; manual execution. No broker or order tools are available.",
    }


def validate_chat_answer(answer: ChatAnswer, snapshot: Mapping[str, Any]) -> None:
    expected_state = DecisionState(str(snapshot["meta"]["state"]))
    if answer.state != expected_state:
        raise ValueError("chat attempted to override deterministic state")
    lower = answer.answer.casefold()
    if any(phrase in lower for phrase in ("guaranteed profit", "risk-free profit", "i placed", "order was placed")):
        raise ValueError("chat introduced an unsafe certainty or execution claim")
    known_prices = _known_prices(snapshot)
    for matched in _PRICE_PATTERN.findall(answer.answer):
        value = Decimal(matched)
        if not any(abs(value - known) <= Decimal("0.0001") for known in known_prices):
            raise ValueError(f"chat introduced unknown price {value}")
    level2 = snapshot.get("level2") or {}
    if str(level2.get("hidden_seller", "")).startswith("UNAVAILABLE") and "confirmed hidden seller" in lower:
        raise ValueError("chat claimed unavailable hidden-seller evidence")
    if str(level2.get("red_tape_burst", "")).startswith("UNAVAILABLE") and "confirmed red tape" in lower:
        raise ValueError("chat claimed unavailable tape evidence")
    if answer.state in {DecisionState.BUY, DecisionState.SELL} and (
        "analysis signal" not in lower or "manual" not in lower
    ):
        raise ValueError("BUY/SELL chat omitted the manual-execution boundary")


def _known_prices(snapshot: Mapping[str, Any]) -> set[Decimal]:
    values: set[Decimal] = set()
    for section, fields in (
        (snapshot.get("quote") or {}, ("last", "bid", "ask")),
        (
            snapshot.get("indicators") or {},
            ("price", "vwap", "ema9_1m", "ema20_1m", "ema9_5m", "ema20_5m"),
        ),
        (snapshot.get("plan") or {}, ("trigger", "stop", "risk_per_share", "maximum_risk", "resistance")),
        (snapshot.get("pattern") or {}, ("impulse_low", "impulse_high", "pullback_low")),
    ):
        for field in fields:
            value = section.get(field)
            if isinstance(value, int | float):
                values.add(Decimal(str(value)))
    return values
