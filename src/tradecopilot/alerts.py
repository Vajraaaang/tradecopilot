from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from tradecopilot.explain import Explanation
from tradecopilot.models import DecisionState, MarketFrame, StateTransition, StrategyDecision

LOGGER = logging.getLogger(__name__)
ALERT_STATES = frozenset(
    {
        DecisionState.ARMED,
        DecisionState.BUY,
        DecisionState.EXIT_WARNING,
        DecisionState.SELL,
        DecisionState.DAY_STOP,
    }
)


@dataclass(frozen=True)
class AlertMessage:
    symbol: str
    state: DecisionState
    timestamp: datetime
    action: str
    manual_execution: bool = True

    def payload(self) -> Mapping[str, object]:
        return {
            "symbol": self.symbol,
            "state": self.state.value,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "manual_execution": self.manual_execution,
        }


class AlertSink(Protocol):
    async def send(self, message: AlertMessage) -> None: ...


class WebhookAlertSink:
    """Send a sanitized state-change alert to a user-configured HTTPS webhook."""

    def __init__(self, url: str, *, timeout_seconds: float = 5.0) -> None:
        parsed = urlparse(url)
        local_http = parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"}
        if parsed.scheme != "https" and not local_http:
            raise ValueError("alert webhook must use HTTPS (HTTP is allowed only for localhost)")
        if not parsed.netloc:
            raise ValueError("alert webhook URL is incomplete")
        if timeout_seconds <= 0:
            raise ValueError("alert webhook timeout must be positive")
        self._url = url
        self._timeout_seconds = timeout_seconds

    async def send(self, message: AlertMessage) -> None:
        await asyncio.to_thread(self._send_sync, message)

    def _send_sync(self, message: AlertMessage) -> None:
        payload = json.dumps(message.payload(), separators=(",", ":")).encode()
        request = Request(
            self._url,
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "tradecopilot/0.1"},
            method="POST",
        )
        with urlopen(request, timeout=self._timeout_seconds) as response:
            if not 200 <= response.status < 300:
                raise ConnectionError(f"alert webhook returned HTTP {response.status}")


@dataclass
class TransitionAlertDispatcher:
    sink: AlertSink
    _recent: deque[tuple[str, DecisionState, datetime]] = field(default_factory=lambda: deque(maxlen=200))

    async def notify(
        self,
        decision: StrategyDecision,
        frame: MarketFrame,
        explanation: Explanation,
        transition: StateTransition,
    ) -> None:
        stale_while_exposed = (
            decision.state == DecisionState.DATA_STALE and frame.position is not None and frame.position.quantity > 0
        )
        if decision.state not in ALERT_STATES and not stale_while_exposed:
            return
        signature = (decision.symbol, transition.new_state, transition.provider_timestamp)
        if signature in self._recent:
            return
        self._recent.append(signature)
        message = AlertMessage(
            symbol=decision.symbol,
            state=decision.state,
            timestamp=decision.provider_timestamp,
            action=explanation.one_sentence_action,
        )
        try:
            await self.sink.send(message)
        except Exception as exc:
            LOGGER.warning("State-change alert delivery failed (%s)", type(exc).__name__)
