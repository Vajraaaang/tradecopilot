from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import threading
import webbrowser
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from decimal import Decimal
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from rich.console import Console

from tradecopilot.alerts import TransitionAlertDispatcher
from tradecopilot.chat import TradeChatAgent, sanitized_chat_context, validate_chat_answer
from tradecopilot.config import StrategyConfig
from tradecopilot.explain import Explanation, SafeExplanationService
from tradecopilot.indicators import InsufficientIndicators, ema_series, macd, rsi, session_vwap
from tradecopilot.journal import Journal
from tradecopilot.models import DecisionState, MarketFrame, OHLCVBar, StrategyDecision
from tradecopilot.monitor import Monitor
from tradecopilot.providers.base import FrameProvider

if TYPE_CHECKING:
    from tradecopilot.jev import JevAdvisor

EASTERN = ZoneInfo("America/New_York")
PACIFIC = ZoneInfo("America/Los_Angeles")
STATIC_ROOT = Path(__file__).with_name("web")
MAX_CHAT_BYTES = 4_096
LOGGER = logging.getLogger(__name__)
_SYMBOL_PATTERN = re.compile(r"[A-Z][A-Z0-9.-]{0,9}")
_JEV_BLOCKED_STATES = {"DATA_STALE", "DATA_INSUFFICIENT", "NO_TRADE", "DAY_STOP", "SELL"}


class DashboardState:
    """Thread-safe, sanitized application-visible state."""

    def __init__(
        self,
        config: StrategyConfig,
        chat_agent: TradeChatAgent | None = None,
        symbol_selector: Callable[[str], bool] | None = None,
        jev_advisor: JevAdvisor | None = None,
    ) -> None:
        self.config = config
        self._chat_agent = chat_agent
        self._symbol_selector = symbol_selector
        self._jev_advisor = jev_advisor
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._version = 0
        self._snapshot: dict[str, Any] = {
            "ready": False,
            "complete": False,
            "error": None,
            "safety": "Analysis only. All execution is manual.",
            "chat": self._chat_metadata(),
            "jev": self._jev_metadata(),
            "alert": None,
        }
        self._recent_positions: dict[str, dict[str, Any]] = {}
        self._state_history: list[dict[str, Any]] = []
        self._alert_sequence = 0
        self._latest_alert: dict[str, Any] | None = None

    def update(self, decision: StrategyDecision, frame: MarketFrame, explanation: Explanation) -> None:
        position = frame.position
        symbol = decision.symbol
        open_symbols = {item.symbol for item in frame.positions if item.quantity > 0}
        if frame.account_risk is not None:
            for prior_symbol, prior in tuple(self._recent_positions.items()):
                if prior.get("status") == "OPEN" and prior_symbol not in open_symbols and prior_symbol != symbol:
                    self._recent_positions[prior_symbol] = {
                        **prior,
                        "status": "FLAT / NOT PRESENT",
                        "quantity": 0,
                        "unrealized_pnl": None,
                        "feedback": "Position is no longer present in the reconciled Robinhood position read.",
                    }
        for portfolio_position in frame.positions:
            if portfolio_position.quantity <= 0 or portfolio_position.symbol == symbol:
                continue
            self._recent_positions[portfolio_position.symbol] = {
                "symbol": portfolio_position.symbol,
                "account_alias": portfolio_position.account_alias,
                "status": "OPEN",
                "quantity": _number(portfolio_position.quantity),
                "average_entry": _number(portfolio_position.average_entry),
                "unrealized_pnl": _number(portfolio_position.unrealized_pnl),
                "current_price": None,
                "current_r": None,
                "mfe": _number(portfolio_position.mfe),
                "mae": _number(portfolio_position.mae),
                "structural_stop": None,
                "latest_higher_low": _number(portfolio_position.latest_higher_low),
                "vwap": None,
                "ema9": None,
                "nearest_resistance": None,
                "quote_age": None,
                "feedback": (
                    f"Open broker position detected. Select {portfolio_position.symbol} "
                    "for deterministic live feedback."
                ),
                "changes_to_exit": ["Select this ticker to calculate fresh structural exit conditions"],
            }
        state_changed = not self._state_history or self._state_history[-1]["state"] != decision.state.value
        if state_changed:
            self._state_history.append(
                {
                    "state": decision.state.value,
                    "timestamp": decision.provider_timestamp.isoformat(),
                    "reason": decision.reasons[0] if decision.reasons else "No deterministic reason available",
                }
            )
            self._state_history = self._state_history[-10:]
            stale_while_exposed = (
                decision.state == DecisionState.DATA_STALE and position is not None and position.quantity > 0
            )
            if (
                decision.state
                in {
                    DecisionState.ARMED,
                    DecisionState.BUY,
                    DecisionState.EXIT_WARNING,
                    DecisionState.SELL,
                    DecisionState.DAY_STOP,
                }
                or stale_while_exposed
            ):
                self._alert_sequence += 1
                self._latest_alert = {
                    "sequence": self._alert_sequence,
                    "symbol": symbol,
                    "state": decision.state.value,
                    "timestamp": decision.provider_timestamp.isoformat(),
                    "action": explanation.one_sentence_action,
                    "manual_execution": True,
                }
        if position is not None and position.quantity > 0:
            self._recent_positions[symbol] = {
                "symbol": symbol,
                "account_alias": position.account_alias,
                "status": "OPEN",
                "quantity": _number(position.quantity),
                "average_entry": _number(position.average_entry),
                "unrealized_pnl": _number(position.unrealized_pnl),
                "current_price": _number(decision.current_price),
                "current_r": _current_r(decision),
                "mfe": _number(position.mfe),
                "mae": _number(position.mae),
                "structural_stop": _number(decision.trade_plan.structural_stop) if decision.trade_plan else None,
                "latest_higher_low": _number(position.latest_higher_low),
                "vwap": _number(decision.indicator.vwap) if decision.indicator else None,
                "ema9": _number(decision.indicator.ema9_1m) if decision.indicator else None,
                "nearest_resistance": (
                    _number(decision.trade_plan.nearest_resistance) if decision.trade_plan else None
                ),
                "quote_age": frame.quote.age_seconds if frame.quote else None,
                "feedback": _position_feedback(decision),
                "changes_to_exit": list(decision.what_changes_the_decision),
            }
        elif symbol in self._recent_positions and decision.state in {
            DecisionState.SELL,
            DecisionState.REENTRY_WATCH,
        }:
            self._recent_positions[symbol] = {
                **self._recent_positions[symbol],
                "status": "FLAT AFTER EXIT",
                "quantity": 0,
                "unrealized_pnl": None,
                "current_price": _number(decision.current_price),
                "feedback": _position_feedback(decision),
            }

        quote = frame.quote
        indicator = decision.indicator
        plan = decision.trade_plan
        risk = frame.account_risk
        gain = None
        if quote is not None:
            gain = ((quote.last - quote.previous_close) / quote.previous_close) * Decimal(100)
        catalyst = frame.catalyst_evidence
        float_evidence = frame.float_evidence
        l2 = frame.level2_history[-1] if frame.level2_history else None
        snapshot: dict[str, Any] = {
            "ready": True,
            "complete": False,
            "error": None,
            "safety": "Analysis signal only; manual execution. No order controls are exposed.",
            "meta": {
                "symbol": symbol,
                "mode": decision.mode.value,
                "state": decision.state.value,
                "strategy": decision.strategy_version,
                "event_time_et": frame.event_time.astimezone(EASTERN).isoformat(timespec="seconds"),
                "event_time_pt": frame.event_time.astimezone(PACIFIC).isoformat(timespec="seconds"),
                "quote_time": quote.provider_timestamp.isoformat() if quote else None,
                "quote_age": quote.age_seconds if quote else None,
                "position_status": "OPEN" if position and position.quantity > 0 else "FLAT",
            },
            "quote": {
                "last": _number(quote.last) if quote else None,
                "bid": _number(quote.bid) if quote else None,
                "ask": _number(quote.ask) if quote else None,
                "change": _number(quote.last - quote.previous_close) if quote else None,
                "change_percent": _number(gain),
                "total_volume": quote.total_volume if quote else None,
            },
            "action": explanation.one_sentence_action,
            "changed": explanation.what_changed,
            "reasons": list(explanation.reasons),
            "next": list(explanation.what_changes_the_decision),
            "missing": list(explanation.missing_data),
            "pillars": [
                {
                    "name": pillar.name,
                    "passed": pillar.passed,
                    "evidence": pillar.evidence,
                    "required": pillar.required,
                }
                for pillar in decision.pillars
            ],
            "quality": {
                "rvol": _number(indicator.source_style_rvol) if indicator else None,
                "gain_percent": _number(gain),
                "gap_percent": _number(frame.gap_percent),
                "float_shares": float_evidence.shares if float_evidence else None,
                "float_verified": bool(float_evidence and float_evidence.verified),
                "float_source": float_evidence.reference if float_evidence else None,
                "catalyst": catalyst.description if catalyst else None,
                "catalyst_verified": bool(catalyst and catalyst.verified),
                "catalyst_source": catalyst.reference if catalyst else None,
                "market_leader": frame.market_leader,
                "pillars_passed": sum(pillar.passed for pillar in decision.pillars),
            },
            "indicators": _indicator_payload(decision),
            "plan": {
                "trigger": _number(plan.trigger_price) if plan else None,
                "stop": _number(plan.structural_stop) if plan else None,
                "risk_per_share": _number(plan.risk_per_share) if plan else None,
                "maximum_risk": _number(plan.maximum_dollar_risk) if plan else None,
                "maximum_shares": plan.maximum_shares if plan else None,
                "resistance": _number(plan.nearest_resistance) if plan else None,
                "reward_risk": _number(plan.reward_risk) if plan else None,
            },
            "pattern": {
                "impulse_low": _number(decision.impulse.low) if decision.impulse else None,
                "impulse_high": _number(decision.impulse.high) if decision.impulse else None,
                "pullback_low": _number(decision.pullback.low) if decision.pullback else None,
                "retracement_percent": (
                    _number(decision.pullback.retracement_fraction * Decimal(100)) if decision.pullback else None
                ),
                "green_volume": _number(decision.impulse.average_green_volume) if decision.impulse else None,
                "red_volume": _number(decision.pullback.average_red_volume) if decision.pullback else None,
            },
            "level2": {
                "bid": _book_level(l2.bids[0]) if l2 and l2.bids else None,
                "ask": _book_level(l2.asks[0]) if l2 and l2.asks else None,
                "persistent_seller": _exit_status(decision, "persistent_large_seller"),
                "hidden_seller": _exit_status(decision, "hidden_seller"),
                "red_tape_burst": _exit_status(decision, "red_tape_burst"),
                "tape_available": frame.time_and_sales is not None,
            },
            "day_stop": {
                "realized": _number(risk.realized_session_pnl) if risk else None,
                "peak_realized": _number(risk.peak_realized_session_pnl) if risk else None,
                "consecutive_losses": risk.consecutive_losses if risk else None,
                "locked": risk.session_locked if risk else None,
            },
            "positions": list(self._recent_positions.values()),
            "history": list(self._state_history),
            "screener": _screener_rows(decision, frame),
            "chat": self._chat_metadata(),
            "jev": self._jev_metadata(),
            "alert": self._latest_alert,
            "charts": {
                "1m": _chart_series(frame.bars_1m, frame.bars_1m, self.config),
                "5m": _chart_series(frame.bars_5m, frame.bars_1m, self.config),
            },
        }
        with self._condition:
            self._snapshot = snapshot
            self._version += 1
            self._condition.notify_all()

    def mark_complete(self) -> None:
        with self._condition:
            self._snapshot = {**self._snapshot, "complete": True}
            self._version += 1
            self._condition.notify_all()

    def mark_error(self) -> None:
        with self._condition:
            self._snapshot = {
                **self._snapshot,
                "error": "The data provider stopped. Review the terminal for diagnostic details.",
            }
            self._version += 1
            self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return cast(dict[str, Any], json.loads(json.dumps(self._snapshot)))

    def wait_for_snapshot(self, version: int, timeout: float = 15.0) -> tuple[int, dict[str, Any]]:
        with self._condition:
            self._condition.wait_for(lambda: self._version > version, timeout=timeout)
            return self._version, cast(dict[str, Any], json.loads(json.dumps(self._snapshot)))

    def jev_advice(self, symbol: str, source_time: str | None = None) -> dict[str, Any]:
        snapshot = self.snapshot()
        meta = snapshot.get("meta", {})
        if symbol != meta.get("symbol") or (source_time is not None and source_time != meta.get("quote_time")):
            raise ValueError("The selected market context changed; request the current symbol again")
        if self._jev_advisor is None:
            return self._jev_no_advice("disabled", "Jev is not enabled for this session.", snapshot)
        if _jev_context_blocked(snapshot, self.config.maximum_quote_age_seconds):
            return self._jev_no_advice("blocked", "Jev requires fresh live data and an unblocked risk state.", snapshot)
        # Hosted inference runs in this HTTP worker, outside the snapshot lock and monitor loop.
        try:
            result = self._jev_advisor.advise(snapshot)
        except Exception:
            LOGGER.warning("Jev advisory unavailable; the deterministic decision is unchanged")
            result = self._jev_no_advice(
                "unavailable", "Jev is unavailable. The engine remains authoritative.", snapshot,
            )
        current = self.snapshot()
        context_changed = (
            _jev_context_blocked(current, self.config.maximum_quote_age_seconds)
            or _jev_context_identity(current) != _jev_context_identity(snapshot)
        )
        if context_changed or not _jev_result_matches(result, snapshot):
            result = self._jev_no_advice(
                "blocked", "Market context changed or expired. Request a fresh assessment.", current,
            )
        if result.get("status") not in {"available", "uncertain"}:
            result = {
                **result,
                "symbol": result.get("symbol") or meta.get("symbol"),
                "source_time": result.get("source_time") or meta.get("quote_time"),
            }
        metadata = self._jev_metadata()
        with self._condition:
            self._snapshot = {**self._snapshot, "jev": metadata}
            self._version += 1
            self._condition.notify_all()
        return {**result, **metadata}

    def _jev_metadata(self) -> dict[str, Any]:
        if self._jev_advisor is not None:
            return {
                **self._jev_advisor.metadata(),
                "maximum_quote_age_seconds": self.config.maximum_quote_age_seconds,
            }
        return {
            "enabled": False,
            "model": "jev",
            "requests_used": 0,
            "request_limit": 100,
            "estimated_cost_usd": 0.0,
            "maximum_quote_age_seconds": self.config.maximum_quote_age_seconds,
        }

    def _jev_no_advice(self, status: str, message: str, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        meta = snapshot.get("meta", {})
        return {
            **self._jev_metadata(),
            "status": status,
            "action": None,
            "probabilities": {},
            "confidence": None,
            "message": message,
            "symbol": meta.get("symbol"),
            "source_time": meta.get("quote_time"),
            "expires_at": None,
            "input_tokens": None,
            "cached": False,
        }

    def answer(self, query: str) -> dict[str, Any]:
        snapshot = self.snapshot()
        if not snapshot.get("ready"):
            answer = "The first normalized market frame has not arrived yet. No decision is available."
            return {"answer": answer, "state": "DATA_INSUFFICIENT", "manual_execution": True}
        state = str(snapshot["meta"]["state"])
        lowered = query.casefold()
        plan = snapshot["plan"]
        indicators = snapshot["indicators"]
        positions = snapshot["positions"]
        restricted_request = any(word in lowered for word in ("order", "execute", "place", "cancel"))
        if restricted_request:
            answer = (
                "I cannot place, review, change, or cancel orders. "
                f"The deterministic engine is {state}; {snapshot['action']}"
            )
        elif any(word in lowered for word in ("why", "reason", "valid")):
            reasons = "; ".join(snapshot["reasons"][:4])
            answer = f"State {state}: {reasons}. Next change: {_first(snapshot['next'])}."
        elif any(word in lowered for word in ("trigger", "entry", "buy")):
            answer = (
                f"The strict trigger is {_money(plan['trigger'])}; structural stop {_money(plan['stop'])}; "
                f"maximum shares {plan['maximum_shares'] or 'unavailable'}. Current state is {state}. "
                "This is analysis only; any execution is manual."
            )
        elif any(word in lowered for word in ("stop", "risk", "share", "size")):
            answer = (
                f"Structural stop {_money(plan['stop'])}; risk per share {_money(plan['risk_per_share'])}; "
                f"risk cap {_money(plan['maximum_risk'])}; maximum shares {plan['maximum_shares'] or 'unavailable'}."
            )
        elif any(word in lowered for word in ("vwap", "ema", "rsi", "macd", "atr", "indicator")):
            answer = (
                f"Price {indicators['price']} vs VWAP {indicators['vwap']} and EMA9 {indicators['ema9_1m']}; "
                f"EMA20 {indicators['ema20_1m']}, ATR {indicators['atr14_1m']}, "
                f"RSI {indicators['rsi14_1m']}, MACD {indicators['macd_1m']}. "
                "Indicators provide context and cannot independently create BUY or SELL."
            )
        elif any(word in lowered for word in ("position", "holding", "stock")):
            if positions:
                current = positions[0]
                answer = f"{current['symbol']} is {current['status']}. {current['feedback']}"
            else:
                answer = f"No open or recent replay position is visible. Current state is {state}."
        elif any(word in lowered for word in ("screen", "float", "catalyst", "rvol")):
            quality = snapshot["quality"]
            answer = (
                f"The screener has {quality['pillars_passed']}/5 pillars, RVOL {quality['rvol']}, "
                f"verified float {quality['float_shares']}, and catalyst {quality['catalyst'] or 'unavailable'}."
            )
        else:
            answer = (
                f"{snapshot['meta']['symbol']} is {state} at {_money(snapshot['quote']['last'])}. "
                f"{snapshot['changed']} Next change: {_first(snapshot['next'])}."
            )
        fallback = {
            "answer": answer,
            "state": state,
            "manual_execution": True,
            "model": "deterministic-fallback",
            "reasoning": "local",
        }
        if self._chat_agent is None or restricted_request:
            return fallback
        try:
            candidate = self._chat_agent.answer(query, sanitized_chat_context(snapshot))
            validate_chat_answer(candidate, snapshot)
        except Exception as exc:
            LOGGER.warning("GPT chat response rejected; deterministic fallback used: %s", exc)
            return fallback
        return {
            **candidate.model_dump(mode="json"),
            "model": self._chat_agent.model_version,
            "reasoning": self._chat_agent.reasoning_label,
        }

    def select_symbol(self, requested: str) -> dict[str, Any]:
        symbol = requested.strip().upper()
        if not _SYMBOL_PATTERN.fullmatch(symbol):
            return {"accepted": False, "symbol": symbol, "message": "Enter a valid ticker symbol."}
        current = self.snapshot().get("meta", {}).get("symbol")
        if symbol == current:
            return {"accepted": True, "symbol": symbol, "message": f"{symbol} is already loaded."}
        if self._symbol_selector is not None and self._symbol_selector(symbol):
            return {"accepted": True, "symbol": symbol, "message": f"Loading {symbol} from the read-only provider."}
        return {
            "accepted": False,
            "symbol": symbol,
            "message": (
                f"{symbol} is not available from the active replay provider. "
                "A live switch requires the configured read-only market-data adapter."
            ),
        }

    def _chat_metadata(self) -> dict[str, str]:
        if self._chat_agent is None:
            return {"provider": "deterministic", "model": "local", "reasoning": "local"}
        return {
            "provider": "openai",
            "model": self._chat_agent.model_version,
            "reasoning": self._chat_agent.reasoning_label,
        }


class DashboardHTTPServer(ThreadingHTTPServer):
    state: DashboardState
    daemon_threads = True


def _jev_context_blocked(snapshot: Mapping[str, Any], maximum_quote_age_seconds: float) -> bool:
    meta = snapshot.get("meta", {})
    if (
        not snapshot.get("ready")
        or snapshot.get("error")
        or snapshot.get("complete")
        or meta.get("mode") != "live"
        or meta.get("state") in _JEV_BLOCKED_STATES
        or snapshot.get("day_stop", {}).get("locked")
        or snapshot.get("missing")
    ):
        return True
    quote_age = meta.get("quote_age")
    if not isinstance(quote_age, (int, float)) or isinstance(quote_age, bool) or not math.isfinite(quote_age):
        return True
    try:
        quote_time = datetime.fromisoformat(str(meta.get("quote_time")))
        if quote_time.tzinfo is None:
            return True
        wall_age = (datetime.now(UTC) - quote_time).total_seconds()
    except ValueError:
        return True
    return not (0 <= quote_age <= maximum_quote_age_seconds and 0 <= wall_age <= maximum_quote_age_seconds)


def _jev_context_identity(snapshot: Mapping[str, Any]) -> tuple[Any, ...]:
    meta = snapshot.get("meta", {})
    plan = snapshot.get("plan", {})
    positions = tuple(
        (position.get("quantity"), position.get("average_entry"))
        for position in snapshot.get("positions", ())
        if position.get("symbol") == meta.get("symbol")
    )
    return (
        meta.get("symbol"), meta.get("state"), meta.get("position_status"),
        snapshot.get("day_stop"), plan.get("stop"), plan.get("maximum_shares"), positions,
    )


def _jev_result_matches(result: Mapping[str, Any], snapshot: Mapping[str, Any]) -> bool:
    if result.get("status") not in {"available", "uncertain"}:
        return True
    meta = snapshot.get("meta", {})
    if result.get("symbol") != meta.get("symbol"):
        return False
    try:
        source_time = datetime.fromisoformat(str(result.get("source_time")))
        expected_time = datetime.fromisoformat(str(meta.get("quote_time")))
        expiry = datetime.fromisoformat(str(result.get("expires_at")))
        return (
            source_time.tzinfo is not None
            and source_time == expected_time
            and expiry.tzinfo is not None
            and expiry > datetime.now(UTC)
        )
    except ValueError:
        return False


def _handler(state: DashboardState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "Tradecopilot/0.1"

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/snapshot":
                self._json(state.snapshot())
                return
            if path == "/api/health":
                self._json({"ok": True, "write_capabilities": False})
                return
            if path == "/api/events":
                self._events()
                return
            static = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/app.css": ("app.css", "text/css; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/assets/robinhood-feather.svg": ("assets/robinhood-feather.svg", "image/svg+xml"),
                "/assets/technical-indicators.svg": ("assets/technical-indicators.svg", "image/svg+xml"),
                "/assets/drawing-tools.svg": ("assets/drawing-tools.svg", "image/svg+xml"),
                "/assets/intervals.svg": ("assets/intervals.svg", "image/svg+xml"),
                "/assets/trend.svg": ("assets/trend.svg", "image/svg+xml"),
            }.get(path)
            if static is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            relative, content_type = static
            target = STATIC_ROOT / relative
            self._bytes(target.read_bytes(), content_type)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path not in {"/api/chat", "/api/symbol", "/api/jev"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if path == "/api/jev" and not self._valid_jev_origin():
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            if length <= 0 or length > MAX_CHAT_BYTES:
                self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                return
            try:
                raw = json.loads(self.rfile.read(length))
                value = raw.get("query" if path == "/api/chat" else "symbol") if isinstance(raw, Mapping) else None
            except (UnicodeDecodeError, json.JSONDecodeError):
                value = None
            if not isinstance(value, str) or not value.strip():
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            if path == "/api/chat":
                self._json(state.answer(value.strip()[:1_000]))
            elif path == "/api/jev":
                symbol = value.strip().upper()
                source_time = raw.get("source_time")
                if not _SYMBOL_PATTERN.fullmatch(symbol) or (
                    source_time is not None and not isinstance(source_time, str)
                ):
                    self.send_error(HTTPStatus.BAD_REQUEST)
                    return
                try:
                    self._json(state.jev_advice(symbol, source_time))
                except ValueError:
                    self.send_error(HTTPStatus.CONFLICT, "Market context changed")
            else:
                self._json(state.select_symbol(value.strip()[:10]))

        def _valid_jev_origin(self) -> bool:
            if self.headers.get_content_type() != "application/json":
                self.send_error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
                return False
            port = cast(DashboardHTTPServer, self.server).server_port
            host = self.headers.get("Host", "")
            allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            origin = self.headers.get("Origin")
            if (
                len(self.headers.get_all("Host", [])) != 1
                or len(self.headers.get_all("Origin", [])) > 1
                or host not in allowed_hosts
                or (origin is not None and origin != f"http://{host}")
                or self.headers.get("Sec-Fetch-Site") not in {None, "same-origin", "none"}
            ):
                self.send_error(HTTPStatus.FORBIDDEN)
                return False
            return True

        def _json(self, value: Mapping[str, Any]) -> None:
            payload = json.dumps(value, separators=(",", ":")).encode()
            self._bytes(payload, "application/json; charset=utf-8")

        def _events(self) -> None:
            try:
                version = int(self.headers.get("Last-Event-ID", "-1"))
            except ValueError:
                version = -1
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            try:
                while True:
                    next_version, snapshot = state.wait_for_snapshot(version)
                    if next_version == version:
                        self.wfile.write(b": heartbeat\n\n")
                    else:
                        payload = json.dumps(snapshot, separators=(",", ":"))
                        event = f"id: {next_version}\nevent: snapshot\ndata: {payload}\n\n".encode()
                        self.wfile.write(event)
                        version = next_version
                    self.wfile.flush()
                    if snapshot.get("complete"):
                        return
            except (BrokenPipeError, ConnectionResetError):
                return

        def _bytes(self, payload: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'",
            )
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    return Handler


class _MonitorWorker:
    def __init__(
        self,
        provider: FrameProvider,
        state: DashboardState,
        config: StrategyConfig,
        database_path: Path,
        console: Console,
        explanation_service: SafeExplanationService,
        alert_dispatcher: TransitionAlertDispatcher | None,
    ) -> None:
        self.provider = provider
        self.state = state
        self.config = config
        self.database_path = database_path
        self.console = console
        self.explanation_service = explanation_service
        self.alert_dispatcher = alert_dispatcher
        self.loop: asyncio.AbstractEventLoop | None = None
        self.task: asyncio.Task[list[StrategyDecision]] | None = None
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, name="tradecopilot-monitor")

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.ready.wait(timeout=2)
        if self.loop is not None and self.task is not None and not self.task.done():
            self.loop.call_soon_threadsafe(self.task.cancel)
        self.thread.join(timeout=10)

    def _run(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            with Journal(self.database_path, self.config.raw_snapshot_retention_rows) as journal:
                monitor = Monitor(
                    self.config,
                    journal,
                    self.explanation_service,
                    self.console,
                    render_terminal=False,
                    observer=self.state.update,
                    alert_dispatcher=self.alert_dispatcher,
                )
                self.task = self.loop.create_task(monitor.run(self.provider))
                self.ready.set()
                self.loop.run_until_complete(self.task)
            self.state.mark_complete()
        except asyncio.CancelledError:
            pass
        except Exception:
            self.state.mark_error()
            self.console.print_exception(show_locals=False)
        finally:
            self.ready.set()
            self.loop.close()


def serve_dashboard(
    provider: FrameProvider,
    config: StrategyConfig,
    database_path: Path,
    *,
    port: int = 8765,
    open_browser: bool = True,
    console: Console | None = None,
    chat_agent: TradeChatAgent | None = None,
    symbol_selector: Callable[[str], bool] | None = None,
    explanation_service: SafeExplanationService | None = None,
    alert_dispatcher: TransitionAlertDispatcher | None = None,
    jev_advisor: JevAdvisor | None = None,
) -> None:
    if not 0 <= port <= 65_535:
        raise ValueError("port must be between 0 and 65535")
    active_console = console or Console()
    state = DashboardState(config, chat_agent=chat_agent, symbol_selector=symbol_selector, jev_advisor=jev_advisor)
    server = create_dashboard_server(state, port)
    selected_port = server.server_address[1]
    url = f"http://127.0.0.1:{selected_port}/"
    worker = _MonitorWorker(
        provider,
        state,
        config,
        database_path,
        active_console,
        explanation_service or SafeExplanationService(),
        alert_dispatcher,
    )
    worker.start()
    active_console.print(f"[bold green]Visual copilot:[/bold green] {url}")
    active_console.print("Analysis only; no order or account-mutation capability is exposed. Press Ctrl-C to stop.")
    if open_browser:
        browser_timer = threading.Timer(0.2, webbrowser.open, args=(url,))
        browser_timer.daemon = True
        browser_timer.start()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        active_console.print("\nStopping visual copilot…")
    finally:
        server.server_close()
        worker.stop()


def create_dashboard_server(state: DashboardState, port: int = 0) -> DashboardHTTPServer:
    server = DashboardHTTPServer(("127.0.0.1", port), _handler(state))
    server.state = state
    return server


def _chart_series(
    bars: tuple[OHLCVBar, ...],
    vwap_bars: tuple[OHLCVBar, ...],
    config: StrategyConfig,
) -> list[dict[str, Any]]:
    closes = [bar.close for bar in bars]
    ema9_values = ema_series(closes, 9)
    ema20_values = ema_series(closes, 20)
    output: list[dict[str, Any]] = []
    start = max(0, len(bars) - 90)
    for index in range(start, len(bars)):
        bar = bars[index]
        vwap_prefix = tuple(
            candidate for candidate in vwap_bars if candidate.provider_timestamp <= bar.provider_timestamp
        )
        try:
            vwap: Decimal | None = session_vwap(vwap_prefix, config.vwap_session)
        except InsufficientIndicators:
            vwap = None
        try:
            rsi_value = rsi(closes[: index + 1], 14)
        except InsufficientIndicators:
            rsi_value = None
        macd_value: Decimal | None
        macd_signal: Decimal | None
        try:
            macd_value, macd_signal = macd(closes[: index + 1])
        except InsufficientIndicators:
            macd_value = macd_signal = None
        output.append(
            {
                "t": bar.provider_timestamp.isoformat(),
                "o": _number(bar.open),
                "h": _number(bar.high),
                "l": _number(bar.low),
                "c": _number(bar.close),
                "v": bar.volume,
                "vwap": _number(vwap),
                "ema9": _number(ema9_values[index]),
                "ema20": _number(ema20_values[index]),
                "rsi": _number(rsi_value),
                "macd": _number(macd_value),
                "macd_signal": _number(macd_signal),
            }
        )
    return output


def _indicator_payload(decision: StrategyDecision) -> dict[str, Any]:
    indicator = decision.indicator
    if indicator is None:
        return {
            "price": _number(decision.current_price),
            "vwap": None,
            "ema9_1m": None,
            "ema20_1m": None,
            "ema9_5m": None,
            "ema20_5m": None,
            "atr14_1m": None,
            "rsi14_1m": None,
            "macd_1m": None,
            "macd_signal_1m": None,
            "structure_1m": "unavailable",
            "structure_5m": "unavailable",
            "spread_percent": None,
            "vwap_session": None,
        }
    return {
        "price": _number(decision.current_price),
        "vwap": _number(indicator.vwap),
        "ema9_1m": _number(indicator.ema9_1m),
        "ema20_1m": _number(indicator.ema20_1m),
        "ema9_5m": _number(indicator.ema9_5m),
        "ema20_5m": _number(indicator.ema20_5m),
        "atr14_1m": _number(indicator.atr14_1m),
        "rsi14_1m": _number(indicator.rsi14_1m),
        "macd_1m": _number(indicator.macd_1m),
        "macd_signal_1m": _number(indicator.macd_signal_1m),
        "structure_1m": indicator.one_minute_structure,
        "structure_5m": indicator.five_minute_structure,
        "spread_percent": _number(indicator.spread_percent),
        "vwap_session": indicator.session_definition,
    }


def _screener_row(decision: StrategyDecision, frame: MarketFrame) -> dict[str, Any]:
    indicator = decision.indicator
    quote = frame.quote
    return {
        "symbol": decision.symbol,
        "state": decision.state.value,
        "price": _number(decision.current_price),
        "gain_percent": (
            _number(((quote.last - quote.previous_close) / quote.previous_close) * Decimal(100)) if quote else None
        ),
        "rvol": _number(indicator.source_style_rvol) if indicator else None,
        "volume": quote.total_volume if quote else None,
        "float_shares": frame.float_evidence.shares if frame.float_evidence else None,
        "pillars": sum(pillar.passed for pillar in decision.pillars),
        "feedback": decision.reasons[0] if decision.reasons else "No deterministic reason available",
    }


def _screener_rows(decision: StrategyDecision, frame: MarketFrame) -> list[dict[str, Any]]:
    selected = _screener_row(decision, frame)
    rows = [selected]
    for candidate in frame.screener_candidates:
        if candidate.symbol == decision.symbol:
            selected["scan_title"] = candidate.scan_title
            selected["columns"] = candidate.columns
            continue
        rows.append(
            {
                "symbol": candidate.symbol,
                "state": "DATA_INSUFFICIENT",
                "price": None,
                "gain_percent": _column_number(candidate.columns, "% Change", "Change %"),
                "rvol": _column_number(candidate.columns, "Relative volume", "RVOL"),
                "volume": _column_number(candidate.columns, "Volume"),
                "float_shares": candidate.float_shares,
                "pillars": None,
                "feedback": "Select ticker to run the deterministic strategy; no scanner row can issue BUY.",
                "scan_title": candidate.scan_title,
                "columns": candidate.columns,
            }
        )
    return rows


def _column_number(columns: Mapping[str, str], *names: str) -> float | None:
    normalized = {key.casefold(): value for key, value in columns.items()}
    for name in names:
        raw = normalized.get(name.casefold())
        if raw is None:
            continue
        cleaned = raw.replace(",", "").replace("%", "").replace("x", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            continue
    return None


def _current_r(decision: StrategyDecision) -> float | None:
    position = decision.position
    plan = decision.trade_plan
    price = decision.current_price
    if position is None or position.average_entry is None or plan is None or price is None:
        return None
    initial_risk = position.average_entry - plan.structural_stop
    if initial_risk <= 0:
        return None
    return float((price - position.average_entry) / initial_risk)


def _position_feedback(decision: StrategyDecision) -> str:
    if decision.state == DecisionState.HOLD:
        return f"HOLD: {decision.reasons[0]}" if decision.reasons else "HOLD: thesis remains intact"
    if decision.state == DecisionState.EXIT_WARNING:
        return f"EXIT WARNING: {decision.reasons[0]}" if decision.reasons else "EXIT WARNING: deterioration is active"
    if decision.state == DecisionState.SELL:
        return f"SELL analysis signal: {decision.reasons[0]}" if decision.reasons else "SELL analysis signal"
    if decision.state == DecisionState.REENTRY_WATCH:
        return "Flat after exit; wait for a complete new base and strict trigger"
    return decision.reasons[0] if decision.reasons else decision.state.value


def _exit_status(decision: StrategyDecision, name: str) -> str:
    for signal in decision.exit_signals:
        if signal.name == name:
            return f"{signal.availability.value} · {'confirmed' if signal.confirmed else 'not confirmed'}"
    return "UNAVAILABLE"


def _book_level(level: Any) -> dict[str, Any]:
    return {"price": _number(level.price), "size": level.size}


def _number(value: Decimal | None) -> float | None:
    return float(value) if value is not None else None


def _money(value: object) -> str:
    if isinstance(value, int | float):
        return f"${value:.2f}"
    return "unavailable"


def _first(values: list[Any]) -> str:
    return str(values[0]) if values else "wait for a material deterministic feature change"
