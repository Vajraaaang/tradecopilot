from __future__ import annotations

import asyncio
import json
import threading
import urllib.error
import urllib.request
from collections.abc import Mapping
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from conftest import FramesProvider
from tradecopilot.chat import ChatAnswer
from tradecopilot.config import StrategyConfig
from tradecopilot.explain import DeterministicExplainer
from tradecopilot.indicators import ema, session_vwap
from tradecopilot.journal import Journal
from tradecopilot.models import DataQuality, DecisionState, OHLCVBar
from tradecopilot.monitor import Monitor
from tradecopilot.strategy import DecisionEngine
from tradecopilot.webapp import STATIC_ROOT, DashboardState, _chart_series, create_dashboard_server


class FakeChatAgent:
    model_version = "gpt-5.6"
    reasoning_label = "pro · high"

    def __init__(self, state: DecisionState = DecisionState.REENTRY_WATCH) -> None:
        self.state = state
        self.context: dict[str, object] | None = None

    def answer(self, query: str, context: Mapping[str, Any]) -> ChatAnswer:
        self.context = dict(context)
        return ChatAnswer(
            answer=f"The deterministic state remains {self.state.value}; trigger $9.60. Execution is manual.",
            state=self.state,
            manual_execution=True,
        )


def populated_dashboard(yxt_frames) -> DashboardState:
    config = StrategyConfig()
    state = DashboardState(config)
    engine = DecisionEngine(config)
    explainer = DeterministicExplainer()
    for frame in yxt_frames:
        decision = engine.evaluate(frame)
        state.update(decision, frame, explainer.explain(decision, ()))
    return state


def test_dashboard_exposes_charts_positions_screener_and_indicators(yxt_frames) -> None:
    snapshot = populated_dashboard(yxt_frames).snapshot()
    assert snapshot["meta"]["state"] == DecisionState.REENTRY_WATCH
    assert snapshot["positions"][0]["status"] == "FLAT AFTER EXIT"
    assert [item["state"] for item in snapshot["history"]] == [
        "WATCH",
        "ARMED",
        "BUY",
        "HOLD",
        "EXIT_WARNING",
        "SELL",
        "REENTRY_WATCH",
    ]
    assert snapshot["screener"][0]["symbol"] == "YXT"
    assert snapshot["charts"]["1m"]
    assert snapshot["charts"]["5m"]
    assert snapshot["indicators"]["vwap"] is not None
    assert snapshot["indicators"]["ema9_1m"] is not None
    assert snapshot["indicators"]["ema20_1m"] is not None
    assert snapshot["indicators"]["atr14_1m"] is not None
    assert snapshot["indicators"]["rsi14_1m"] is not None
    assert "macd_1m" in snapshot["indicators"]
    assert "macd" in snapshot["charts"]["1m"][-1]


def test_dashboard_lists_other_open_robinhood_positions_with_safe_feedback(yxt_frames) -> None:
    config = StrategyConfig()
    state = DashboardState(config)
    engine = DecisionEngine(config)
    explainer = DeterministicExplainer()
    for frame in yxt_frames[:6]:
        assert frame.position is not None
        other = frame.position.model_copy(
            update={
                "symbol": "OTHER",
                "quantity": Decimal("5"),
                "average_entry": Decimal("2.50"),
                "account_alias": "••••0000",
            }
        )
        frame = frame.model_copy(update={"positions": (frame.position, other)})
        decision = engine.evaluate(frame)
        state.update(decision, frame, explainer.explain(decision, ()))
    positions = {item["symbol"]: item for item in state.snapshot()["positions"]}
    assert positions["OTHER"]["status"] == "OPEN"
    assert positions["OTHER"]["account_alias"] == "••••0000"
    assert "Select OTHER" in positions["OTHER"]["feedback"]
    assert positions["OTHER"]["structural_stop"] is None


def test_dashboard_chart_never_contains_future_bars(yxt_frames) -> None:
    config = StrategyConfig()
    state = DashboardState(config)
    engine = DecisionEngine(config)
    explainer = DeterministicExplainer()
    for frame in yxt_frames:
        decision = engine.evaluate(frame)
        state.update(decision, frame, explainer.explain(decision, ()))
        snapshot = state.snapshot()
        for timeframe in ("1m", "5m"):
            assert all(datetime.fromisoformat(row["t"]) <= frame.event_time for row in snapshot["charts"][timeframe])


def test_chart_overlays_use_full_history_before_display_window() -> None:
    start = datetime.fromisoformat("2026-08-10T13:30:00+00:00")
    bars = tuple(
        OHLCVBar(
            provider_timestamp=start + timedelta(minutes=index),
            receipt_timestamp=start + timedelta(minutes=index),
            age_seconds=0,
            source="test",
            quality=DataQuality.GOOD,
            symbol="TST",
            timeframe="1m",
            open=Decimal("5") + Decimal(index) / Decimal(100),
            high=Decimal("5.1") + Decimal(index) / Decimal(100),
            low=Decimal("4.9") + Decimal(index) / Decimal(100),
            close=Decimal("5") + Decimal(index) / Decimal(100),
            volume=100 + index,
        )
        for index in range(100)
    )
    rows = _chart_series(bars, bars, StrategyConfig())
    assert len(rows) == 90
    assert rows[-1]["ema9"] == float(ema([bar.close for bar in bars], 9))
    assert rows[-1]["vwap"] == float(session_vwap(bars, "extended_hours"))


def test_dashboard_chat_is_analysis_only_and_refuses_order_actions(yxt_frames) -> None:
    state = populated_dashboard(yxt_frames)
    response = state.answer("Place a buy order at the trigger and cancel it if it fails")
    assert response["manual_execution"] is True
    assert "cannot place" in response["answer"]
    assert "cancel" in response["answer"]


def test_gpt_chat_gets_only_sanitized_decision_context(yxt_frames) -> None:
    agent = FakeChatAgent()
    config = StrategyConfig()
    state = DashboardState(config, chat_agent=agent)
    engine = DecisionEngine(config)
    explainer = DeterministicExplainer()
    for frame in yxt_frames:
        decision = engine.evaluate(frame)
        state.update(decision, frame, explainer.explain(decision, ()))
    response = state.answer("Explain this setup")
    assert response["model"] == "gpt-5.6"
    assert response["reasoning"] == "pro · high"
    assert agent.context is not None
    assert "positions" not in agent.context
    assert "day_stop" not in agent.context
    assert "tools" not in agent.context
    assert "account" not in json.dumps(agent.context).lower()


def test_gpt_chat_cannot_override_state(yxt_frames) -> None:
    agent = FakeChatAgent(DecisionState.SELL)
    config = StrategyConfig()
    state = DashboardState(config, chat_agent=agent)
    engine = DecisionEngine(config)
    explainer = DeterministicExplainer()
    for frame in yxt_frames:
        decision = engine.evaluate(frame)
        state.update(decision, frame, explainer.explain(decision, ()))
    response = state.answer("Explain this setup")
    assert response["state"] == DecisionState.REENTRY_WATCH
    assert response["model"] == "deterministic-fallback"


def test_symbol_switcher_fails_closed_without_provider_support(yxt_frames) -> None:
    state = populated_dashboard(yxt_frames)
    assert state.select_symbol("yxt")["accepted"] is True
    unavailable = state.select_symbol("SBFM")
    assert unavailable["accepted"] is False
    assert "read-only market-data adapter" in unavailable["message"]


def test_dashboard_http_surface_has_no_mutation_route(yxt_frames) -> None:
    state = populated_dashboard(yxt_frames)
    server = create_dashboard_server(state)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    host, port = server.server_address
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/health") as response:
            health = json.load(response)
        assert health == {"ok": True, "write_capabilities": False}
        request = urllib.request.Request(
            f"http://{host}:{port}/api/chat",
            data=json.dumps({"query": "Why is this state active?"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            answer = json.load(response)
        assert answer["state"] == DecisionState.REENTRY_WATCH
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(urllib.request.Request(f"http://{host}:{port}/api/order", data=b"{}", method="POST"))
        assert exc.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_dashboard_event_stream_reconnects_with_sanitized_snapshot(yxt_frames) -> None:
    state = populated_dashboard(yxt_frames)
    state.mark_complete()
    server = create_dashboard_server(state)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    host, port = server.server_address
    try:
        for _ in range(2):
            with urllib.request.urlopen(f"http://{host}:{port}/api/events", timeout=5) as response:
                lines: list[str] = []
                while line := response.readline().decode():
                    lines.append(line)
                    if line == "\n":
                        break
                payload = "".join(lines)
            assert "event: snapshot" in payload
            assert '"write_capabilities"' not in payload
            assert "api_key" not in payload
            assert "secret_key" not in payload
            assert "account_number" not in payload
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_primary_visual_controls_are_wired_and_read_only() -> None:
    html = (STATIC_ROOT / "index.html").read_text()
    script = (STATIC_ROOT / "app.js").read_text()
    for control_id in (
        "analysisDeskTab",
        "stockCopilotTab",
        "crosshairToggle",
        "chartTypeToggle",
        "indicatorToggle",
        "drawingToggle",
        "fitChart",
        "logScaleToggle",
        "zoomIn",
        "zoomOut",
        "resetChart",
        "fullscreenChart",
        "rotationToggle",
        "alertToggle",
        "chatForm",
    ):
        assert f'id="{control_id}"' in html
        assert f'$("{control_id}").addEventListener' in script
    assert 'fetch("/api/snapshot"' in script
    assert 'fetch("/api/chat"' in script
    assert 'new EventSource("/api/events")' in script
    assert "`${meta.mode.toUpperCase()} DATA`" in script
    assert "/api/order" not in script


def test_monitor_can_feed_visual_observer_without_terminal_output(tmp_path, yxt_frames) -> None:
    seen: list[DecisionState] = []
    with Journal(tmp_path / "journal.sqlite3") as journal:
        monitor = Monitor(
            StrategyConfig(),
            journal,
            compact=False,
            render_terminal=False,
            observer=lambda decision, frame, explanation: seen.append(decision.state),
        )
        decisions = asyncio.run(monitor.run(FramesProvider(yxt_frames)))
    assert seen == [decision.state for decision in decisions]
