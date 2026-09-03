from __future__ import annotations

import asyncio
import io
from collections.abc import Mapping
from typing import Any

from rich.console import Console

from conftest import FramesProvider
from tradecopilot.config import StrategyConfig
from tradecopilot.explain import DeterministicExplainer
from tradecopilot.journal import Journal
from tradecopilot.monitor import Monitor
from tradecopilot.providers.robinhood import RobinhoodReadAdapter
from tradecopilot.security import DISCOVERED_ROBINHOOD_TOOLS, ReadOnlyToolSurface
from tradecopilot.strategy import DecisionEngine
from tradecopilot.terminal import TerminalRenderer


class ConcurrencyTransport:
    def __init__(self) -> None:
        self.active = 0
        self.maximum_active = 0
        self.calls = 0

    async def call(self, tool_name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        del tool_name, arguments
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        self.calls += 1
        await asyncio.sleep(0.01)
        self.active -= 1
        return {"data": {}}


def test_concurrent_polling_does_not_overlap_same_resource() -> None:
    transport = ConcurrencyTransport()
    surface = ReadOnlyToolSurface.from_transport(transport, DISCOVERED_ROBINHOOD_TOOLS)
    adapter = RobinhoodReadAdapter(surface, StrategyConfig())

    async def run() -> None:
        await asyncio.gather(
            adapter.call("get_equity_quotes", {"symbols": ["YXT"]}),
            adapter.call("get_equity_quotes", {"symbols": ["YXT"]}),
        )

    asyncio.run(run())
    assert transport.calls == 2
    assert transport.maximum_active == 1


def test_monitor_deduplicates_transitions_and_flushes_journal(tmp_path, yxt_frames) -> None:
    path = tmp_path / "journal.sqlite3"
    output = io.StringIO()
    with Journal(path) as journal:
        monitor = Monitor(
            StrategyConfig(),
            journal,
            console=Console(file=output, force_terminal=False, width=160),
            compact=True,
        )
        asyncio.run(monitor.run(FramesProvider(yxt_frames)))
    with Journal(path) as reopened:
        report = reopened.report(yxt_frames[0].event_time.date())
    assert report["transitions"]["WATCH"] == 1
    assert report["transitions"]["ARMED"] == 1
    assert report["transitions"]["SELL"] == 1
    assert "state=BUY" in output.getvalue()


def test_graceful_shutdown_flushes_last_transition(tmp_path, day_stop_frame) -> None:
    path = tmp_path / "journal.sqlite3"
    with Journal(path) as journal:
        monitor = Monitor(StrategyConfig(), journal, compact=True)
        asyncio.run(monitor.run(FramesProvider([day_stop_frame])))
    with Journal(path) as reopened:
        assert reopened.report(day_stop_frame.event_time.date())["transitions"]["DAY_STOP"] == 1


def test_terminal_renders_every_required_section(yxt_frames) -> None:
    output = io.StringIO()
    console = Console(file=output, force_terminal=False, width=180)
    config = StrategyConfig()
    decision = DecisionEngine(config).evaluate(yxt_frames[0])
    explanation = DeterministicExplainer().explain(decision, ())
    console.print(TerminalRenderer(config, console).build(decision, yxt_frames[0], explanation))
    rendered = output.getvalue()
    for section in (
        "TRADECOPILOT",
        "ONE-SENTENCE ACTION",
        "STOCK QUALITY",
        "PRICE ACTION",
        "INDICATORS",
        "VOLUME",
        "LEVEL 2 AND TAPE",
        "TRADE PLAN",
        "POSITION MANAGEMENT",
        "WHAT WOULD CHANGE THE DECISION",
        "DAY-STOP STATUS",
        "MISSING OR UNRELIABLE DATA",
    ):
        assert section in rendered
