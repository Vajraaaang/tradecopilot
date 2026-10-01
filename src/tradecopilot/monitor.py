from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from zoneinfo import ZoneInfo

from rich.console import Console
from rich.live import Live

from tradecopilot.alerts import TransitionAlertDispatcher
from tradecopilot.config import StrategyConfig
from tradecopilot.explain import Explanation, SafeExplanationService
from tradecopilot.journal import Journal
from tradecopilot.models import (
    DecisionState,
    JournalObservation,
    MarketFrame,
    StateTransition,
    StrategyDecision,
)
from tradecopilot.positions import PositionTracker
from tradecopilot.providers.base import FrameProvider
from tradecopilot.state_machine import TradingStateMachine
from tradecopilot.strategy import DecisionEngine
from tradecopilot.terminal import TerminalRenderer

EASTERN = ZoneInfo("America/New_York")


class Monitor:
    def __init__(
        self,
        config: StrategyConfig,
        journal: Journal,
        explanation_service: SafeExplanationService | None = None,
        console: Console | None = None,
        *,
        compact: bool = False,
        render_terminal: bool = True,
        observer: Callable[[StrategyDecision, MarketFrame, Explanation], None] | None = None,
        alert_dispatcher: TransitionAlertDispatcher | None = None,
    ) -> None:
        self.config = config
        self.journal = journal
        self.engine = DecisionEngine(config)
        self.machine = TradingStateMachine()
        self.explanations = explanation_service or SafeExplanationService()
        self.console = console or Console()
        self.renderer = TerminalRenderer(config, self.console)
        self.compact = compact
        self.render_terminal = render_terminal
        self.observer = observer
        self.alert_dispatcher = alert_dispatcher
        self._history: list[StateTransition] = []
        self._last_explanation: Explanation | None = None
        self._last_explanation_time = 0.0
        self._last_bar_timestamp: object | None = None
        self._position_tracker = PositionTracker()
        self.journal.save_strategy(config.strategy_version())
        self.journal.flush()

    async def run(self, provider: FrameProvider) -> list[StrategyDecision]:
        decisions: list[StrategyDecision] = []
        live = (
            Live(console=self.console, auto_refresh=False, transient=False)
            if self.render_terminal and not self.compact
            else None
        )
        if live is not None:
            live.start()
        try:
            async for original_frame in provider.frames():
                frame = self._apply_persisted_session_lock(original_frame)
                for position_change in self._position_tracker.observe_all(frame):
                    self.journal.record_position_change(position_change)
                decision = self.engine.evaluate(frame)
                transition = self.machine.apply(decision)
                if transition is not None:
                    self._history.append(transition)
                    self.journal.record_transition(transition)
                if decision.state == DecisionState.DAY_STOP and frame.account_risk is not None:
                    self.journal.lock_session(
                        frame.account_risk.trading_date,
                        "; ".join(decision.reasons),
                        frame.mode.value,
                    )
                explanation = self._explain_when_needed(decision, frame, transition)
                self.journal.record_observation(
                    _observation(
                        decision,
                        frame,
                        explanation_model=self.explanations.model_version,
                        prompt_version=self.explanations.prompt_version,
                    )
                )
                if transition is not None and self.alert_dispatcher is not None:
                    await self.alert_dispatcher.notify(decision, frame, explanation, transition)
                if self.observer is not None:
                    self.observer(decision, frame, explanation)
                if self.render_terminal and self.compact:
                    if transition is not None:
                        self.console.print(_compact_line(decision))
                elif self.render_terminal:
                    assert live is not None
                    self.renderer.alert(decision, frame, enabled=transition is not None)
                    live.update(self.renderer.build(decision, frame, explanation), refresh=True)
                decisions.append(decision)
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        finally:
            if live is not None:
                live.stop()
            self.journal.flush()
        return decisions

    def _apply_persisted_session_lock(self, frame: MarketFrame) -> MarketFrame:
        risk = frame.account_risk
        if risk is None or not self.journal.is_session_locked(risk.trading_date):
            return frame
        return frame.model_copy(update={"account_risk": risk.model_copy(update={"session_locked": True})})

    def _explain_when_needed(
        self,
        decision: StrategyDecision,
        frame: MarketFrame,
        transition: StateTransition | None,
    ) -> Explanation:
        loop_time = asyncio.get_running_loop().time()
        bar_timestamp = frame.bars_1m[-1].provider_timestamp if frame.bars_1m else None
        bar_closed = bar_timestamp is not None and bar_timestamp != self._last_bar_timestamp
        position_open = bool(frame.position and frame.position.quantity > 0)
        heartbeat_due = position_open and loop_time - self._last_explanation_time >= self.config.heartbeat_seconds
        should_explain = (
            transition is not None
            or (bar_closed and (decision.state == DecisionState.ARMED or position_open))
            or heartbeat_due
        )
        self._last_bar_timestamp = bar_timestamp
        if should_explain or self._last_explanation is None:
            self._last_explanation = self.explanations.explain(decision, self._history)
            self._last_explanation_time = loop_time
        return self._last_explanation


def _observation(
    decision: StrategyDecision,
    frame: MarketFrame,
    *,
    explanation_model: str | None,
    prompt_version: str,
) -> JournalObservation:
    bars = tuple(
        {
            "timestamp": bar.provider_timestamp.isoformat(),
            "timeframe": bar.timeframe,
            "open": str(bar.open),
            "high": str(bar.high),
            "low": str(bar.low),
            "close": str(bar.close),
            "volume": bar.volume,
        }
        for bar in (*frame.bars_1m[-3:], *frame.bars_5m[-2:])
    )
    quote = frame.quote
    price = quote or frame.price_snapshot
    level2 = frame.level2_history[-1] if frame.level2_history else None
    market_snapshot = {
        "quote_timestamp": price.provider_timestamp.isoformat() if price else None,
        "bid": str(quote.bid) if quote else None,
        "ask": str(quote.ask) if quote else None,
        "last": str(price.last) if price else None,
        "total_volume": quote.total_volume if quote else None,
        "gap_percent": str(frame.gap_percent) if frame.gap_percent is not None else None,
        "resistance_levels": [str(value) for value in frame.resistance_levels],
    }
    if quote is None and frame.price_snapshot is not None:
        market_snapshot["price_source"] = frame.price_snapshot.source
        market_snapshot["price_only"] = True
    supplemental = {
        "float": frame.float_evidence.model_dump(mode="json") if frame.float_evidence else None,
        "catalyst": (frame.catalyst_evidence.model_dump(mode="json") if frame.catalyst_evidence else None),
    }
    level2_sample = (
        tuple(
            {
                "side": level.side,
                "price": str(level.price),
                "size": level.size,
                "timestamp": level.provider_timestamp.isoformat(),
            }
            for level in (*level2.bids[:3], *level2.asks[:3])
        )
        if level2
        else ()
    )
    position = frame.position
    risk = frame.account_risk
    return JournalObservation(
        provider_timestamp=decision.provider_timestamp,
        receipt_timestamp=decision.receipt_timestamp,
        age_seconds=decision.age_seconds,
        source="monitor",
        quality=decision.quality,
        symbol=decision.symbol,
        trading_date=frame.event_time.astimezone(EASTERN).date(),
        strategy_version=decision.strategy_version,
        mode=frame.mode,
        decision=decision,
        compact_bars=bars,
        market_snapshot=market_snapshot,
        supplemental_evidence=supplemental,
        risk_snapshot=(
            {
                "trading_date": risk.trading_date.isoformat(),
                "buying_power": str(risk.buying_power),
                "realized_session_pnl": str(risk.realized_session_pnl),
                "peak_realized_session_pnl": str(risk.peak_realized_session_pnl),
                "unrealized_pnl": str(risk.unrealized_pnl),
                "consecutive_losses": risk.consecutive_losses,
                "session_locked": risk.session_locked,
            }
            if risk
            else {}
        ),
        level2_sample=level2_sample,
        outcome_tracking={
            "mfe": str(position.mfe) if position and position.mfe is not None else None,
            "mae": str(position.mae) if position and position.mae is not None else None,
            "realized_pnl": None,
            "realized_r": None,
            "time_to_plus_1r_seconds": None,
            "time_to_minus_1r_seconds": None,
            "outcome_after_minutes": {"1": None, "3": None, "5": None, "15": None},
            "exit_latency_seconds": None,
        },
        data_quality_problems=decision.missing_data,
        explanation_model=explanation_model,
        prompt_version=prompt_version,
    )


def _compact_line(decision: StrategyDecision) -> str:
    plan = decision.trade_plan
    trigger = f" trigger={plan.trigger_price}" if plan else ""
    stop = f" stop={plan.structural_stop}" if plan else ""
    return (
        f"{decision.provider_timestamp.isoformat()} {decision.symbol} "
        f"state={decision.state}{trigger}{stop} reason={decision.reasons[0]}"
    )


def transition_states(decisions: Sequence[StrategyDecision]) -> list[DecisionState]:
    states: list[DecisionState] = []
    for decision in decisions:
        if not states or states[-1] != decision.state:
            states.append(decision.state)
    return states
