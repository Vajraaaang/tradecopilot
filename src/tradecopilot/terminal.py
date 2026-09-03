from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal
from zoneinfo import ZoneInfo

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.text import Text

from tradecopilot.config import StrategyConfig
from tradecopilot.explain import Explanation
from tradecopilot.models import (
    AccountRiskSnapshot,
    DecisionState,
    ExitSignal,
    Level2Level,
    MarketFrame,
    SignalAvailability,
    StrategyDecision,
)

EASTERN = ZoneInfo("America/New_York")
PACIFIC = ZoneInfo("America/Los_Angeles")
ALERT_STATES = frozenset(
    {
        DecisionState.ARMED,
        DecisionState.BUY,
        DecisionState.EXIT_WARNING,
        DecisionState.SELL,
        DecisionState.DAY_STOP,
    }
)


class TerminalRenderer:
    def __init__(self, config: StrategyConfig, console: Console | None = None) -> None:
        self.config = config
        self.console = console or Console()

    def render(
        self,
        decision: StrategyDecision,
        frame: MarketFrame,
        explanation: Explanation,
        *,
        alert: bool,
    ) -> None:
        self.alert(decision, frame, enabled=alert)
        self.console.print(self.build(decision, frame, explanation))

    def alert(self, decision: StrategyDecision, frame: MarketFrame, *, enabled: bool) -> None:
        stale_while_exposed = (
            decision.state == DecisionState.DATA_STALE and frame.position is not None and frame.position.quantity > 0
        )
        if enabled and (decision.state in ALERT_STATES or stale_while_exposed):
            self.console.bell()

    def build(self, decision: StrategyDecision, frame: MarketFrame, explanation: Explanation) -> RenderableType:
        quote = frame.quote
        event_et = frame.event_time.astimezone(EASTERN)
        event_pt = frame.event_time.astimezone(PACIFIC)
        position = frame.position
        header = _lines(
            (
                f"Symbol: {decision.symbol}",
                f"Eastern: {event_et.isoformat(timespec='seconds')}",
                f"Pacific: {event_pt.isoformat(timespec='seconds')}",
                f"Quote timestamp: {quote.provider_timestamp.isoformat() if quote else 'UNAVAILABLE'}",
                f"Quote age: {quote.age_seconds:.2f}s" if quote else "Quote age: UNAVAILABLE",
                f"State: {decision.state}",
                f"Position: {'OPEN' if position and position.quantity > 0 else 'FLAT'}",
                f"Strategy: {decision.strategy_version}",
                f"Mode: {decision.mode}",
            )
        )
        pillar_lines = [
            f"{pillar.name}: {'PASS' if pillar.passed else 'FAIL'} — {pillar.evidence}" for pillar in decision.pillars
        ]
        pillar_lines.extend(
            (
                f"Source-style / time-adjusted RVOL: {_value(decision.indicator, 'source_style_rvol')} / "
                f"{_value(decision.indicator, 'time_adjusted_rvol')}",
                f"Total volume: {quote.total_volume if quote else 'UNAVAILABLE'}",
                f"Percentage gain: {_gain_percent(frame)}",
                f"Gap: {frame.gap_percent if frame.gap_percent is not None else 'UNAVAILABLE'}%",
                f"Catalyst: {_catalyst(frame)}",
                f"Float: {_float(frame)}",
                f"Pillars satisfied: {sum(pillar.passed for pillar in decision.pillars)}/5",
                f"Market leader: {frame.market_leader}",
            )
        )
        stock_quality = _lines(pillar_lines or ["Pillars unavailable until required data is complete"])
        impulse_range = _pair(decision.impulse.low, decision.impulse.high) if decision.impulse else "UNAVAILABLE"
        retracement = _percent(decision.pullback.retracement_fraction) if decision.pullback else "UNAVAILABLE"
        prior_high = decision.pullback.previous_candle_high if decision.pullback else "UNAVAILABLE"
        price_action = _lines(
            (
                f"Five-minute structure: {_value(decision.indicator, 'five_minute_structure')}",
                f"One-minute structure: {_value(decision.indicator, 'one_minute_structure')}",
                f"Impulse low/high: {impulse_range}",
                f"Pullback low: {decision.pullback.low if decision.pullback else 'UNAVAILABLE'}",
                f"Retracement: {retracement}",
                f"Previous candle high / entry trigger: {prior_high}",
                f"Latest higher low: {position.latest_higher_low if position else 'UNAVAILABLE'}",
                f"Premarket high/low: {_value(decision.indicator, 'premarket_high')} / "
                f"{_value(decision.indicator, 'premarket_low')}",
                f"High/low of day: {_value(decision.indicator, 'high_of_day')} / "
                f"{_value(decision.indicator, 'low_of_day')}",
                f"Opening range high/low: {_value(decision.indicator, 'opening_range_high')} / "
                f"{_value(decision.indicator, 'opening_range_low')}",
                f"Structural support/resistance: {_value(decision.indicator, 'structural_support')} / "
                f"{_value(decision.indicator, 'structural_resistance')}",
                f"Half/whole-dollar levels: {_value(decision.indicator, 'nearest_half_dollar')} / "
                f"{_value(decision.indicator, 'nearest_whole_dollar')}",
                f"Topping-tail concern: {'ACTIVE' if 'topping' in ' '.join(decision.reasons).lower() else 'clear'}",
            )
        )
        indicator = decision.indicator
        vwap_session = _value(indicator, "session_definition")
        indicators = _lines(
            (
                f"Price vs VWAP: {decision.current_price} vs {_value(indicator, 'vwap')} ({vwap_session})",
                f"VWAP slope: {_value(indicator, 'vwap_slope')}",
                f"Price vs EMA9: {decision.current_price} vs {_value(indicator, 'ema9_1m')}",
                f"EMA9 slope: {_value(indicator, 'ema9_slope_1m')}",
                f"EMA9 vs EMA20: {_value(indicator, 'ema9_1m')} / {_value(indicator, 'ema20_1m')}",
                f"ATR: {_value(indicator, 'atr14_1m')}",
                f"Distance from VWAP: {_value(indicator, 'distance_from_vwap')} / "
                f"{_value(indicator, 'distance_from_vwap_percent')}% / "
                f"{_value(indicator, 'distance_from_vwap_atr')} ATR",
                f"RSI context: {_value(indicator, 'rsi14_1m')}",
                f"MACD context: {_value(indicator, 'macd_1m')}",
            )
        )
        green_volume = decision.impulse.average_green_volume if decision.impulse else "UNAVAILABLE"
        red_volume = decision.pullback.average_red_volume if decision.pullback else "UNAVAILABLE"
        volume = _lines(
            (
                f"Green impulse average: {green_volume}",
                f"Red pullback average: {red_volume}",
                f"Pullback/impulse ratio: {_volume_ratio(decision)}",
                f"Breakout/current-minute volume: {quote.current_minute_volume if quote else 'UNAVAILABLE'}",
                "Expansion/contraction: deterministic ratio shown above",
            )
        )
        l2 = frame.level2_history[-1] if frame.level2_history else None
        l2_signals = {signal.name: signal for signal in decision.exit_signals}
        level2 = _lines(
            (
                f"Bid support: {_top(l2.bids if l2 else ())}",
                f"Ask resistance: {_top(l2.asks if l2 else ())}",
                f"Spread: {_value(indicator, 'spread_dollars')} / {_value(indicator, 'spread_percent')}%",
                f"Persistent seller: {_signal(l2_signals.get('persistent_large_seller'))}",
                f"Hidden seller: {_signal(l2_signals.get('hidden_seller'))}",
                f"Red tape burst: {_signal(l2_signals.get('red_tape_burst'))}",
                "Buying slowdown: see active deterioration reasons",
                f"Data reliability: {l2.quality if l2 else 'UNAVAILABLE'}",
            )
        )
        plan = decision.trade_plan
        trade_plan = _lines(
            (
                f"Trigger: {_value(plan, 'trigger_price')}",
                f"Current price: {decision.current_price or 'UNAVAILABLE'}",
                f"Structural stop: {_value(plan, 'structural_stop')}",
                f"Risk/share: {_value(plan, 'risk_per_share')}",
                f"Maximum risk: {_value(plan, 'maximum_dollar_risk')}",
                f"Maximum shares: {_value(plan, 'maximum_shares')}",
                f"Nearest resistance: {_value(plan, 'nearest_resistance')}",
                f"Available reward: {_value(plan, 'available_reward')}",
                f"Reward/risk: {_value(plan, 'reward_risk')}",
            )
        )
        active_exit_names = ", ".join(signal.name for signal in decision.exit_signals if signal.confirmed)
        position_panel = _lines(
            (
                f"Average entry: {_value(position, 'average_entry')}",
                f"Quantity: {_value(position, 'quantity')}",
                f"Unrealized P&L: {_value(position, 'unrealized_pnl')}",
                f"Current R: {_current_r(decision)}",
                f"MFE: {_value(position, 'mfe')}",
                f"MAE: {_value(position, 'mae')}",
                f"Latest invalidation: {_value(plan, 'invalidation_level')}",
                f"Active exit indicators: {active_exit_names or 'none confirmed'}",
            )
        )
        risk = frame.account_risk
        day_stop = _lines(
            (
                f"Peak realized P&L: {_value(risk, 'peak_realized_session_pnl')}",
                f"Current realized P&L: {_value(risk, 'realized_session_pnl')}",
                f"Giveback: {_giveback(risk)}",
                f"Remaining daily loss capacity: {_remaining_loss(risk, self.config.maximum_daily_loss_usd)}",
                f"Consecutive losses: {_value(risk, 'consecutive_losses')}",
                f"Trading window: "
                f"{'closed' if event_et.time().replace(tzinfo=None) >= self.config.trading_cutoff_eastern else 'open'}",
                f"Session lock: {_value(risk, 'session_locked')}",
            )
        )
        missing = list(explanation.missing_data)
        missing.extend(
            f"{signal.name}: {signal.availability}"
            for signal in decision.exit_signals
            if signal.availability in {SignalAvailability.UNAVAILABLE, SignalAvailability.LIMITED}
        )
        if frame.time_and_sales is None:
            missing.append("Time-and-sales unavailable; hidden seller and red-tape confirmation are limited")
        missing = list(dict.fromkeys(missing))
        return Group(
            Panel(header, title="TRADECOPILOT", border_style="#1f6feb"),
            Panel(explanation.one_sentence_action, title="ONE-SENTENCE ACTION", border_style="#00c805"),
            Panel(_lines((explanation.what_changed, *explanation.reasons)), title="WHAT CHANGED / WHY"),
            Panel(stock_quality, title="STOCK QUALITY"),
            Panel(price_action, title="PRICE ACTION"),
            Panel(indicators, title="INDICATORS"),
            Panel(volume, title="VOLUME"),
            Panel(level2, title="LEVEL 2 AND TAPE"),
            Panel(trade_plan, title="TRADE PLAN", border_style="#ff8c00"),
            Panel(position_panel, title="POSITION MANAGEMENT"),
            Panel(
                _lines(explanation.what_changes_the_decision),
                title="WHAT WOULD CHANGE THE DECISION",
            ),
            Panel(day_stop, title="DAY-STOP STATUS"),
            Panel(_lines(missing or ("None",)), title="MISSING OR UNRELIABLE DATA", border_style="red"),
        )


def _lines(values: Iterable[object]) -> Text:
    return Text("\n".join(str(value) for value in values))


def _value(value: object | None, attribute: str) -> object:
    if value is None:
        return "UNAVAILABLE"
    result = getattr(value, attribute, None)
    return result if result is not None else "UNAVAILABLE"


def _pair(first: object, second: object) -> str:
    return f"{first} / {second}"


def _percent(value: Decimal) -> str:
    return f"{value * Decimal(100):.2f}%"


def _volume_ratio(decision: StrategyDecision) -> object:
    if not decision.impulse or not decision.pullback or not decision.impulse.average_green_volume:
        return "UNAVAILABLE"
    return decision.pullback.average_red_volume / decision.impulse.average_green_volume


def _top(levels: tuple[Level2Level, ...]) -> str:
    if not levels:
        return "UNAVAILABLE"
    level = levels[0]
    return f"{level.price} x {level.size}"


def _signal(signal: ExitSignal | None) -> object:
    if signal is None:
        return "UNAVAILABLE"
    return f"{signal.availability} (confirmed={signal.confirmed})"


def _current_r(decision: StrategyDecision) -> object:
    if not decision.trade_plan or decision.current_price is None or not decision.position:
        return "UNAVAILABLE"
    if decision.position.average_entry is None:
        return "UNAVAILABLE"
    return (decision.current_price - decision.position.average_entry) / decision.trade_plan.risk_per_share


def _giveback(risk: AccountRiskSnapshot | None) -> object:
    if risk is None:
        return "UNAVAILABLE"
    peak = risk.peak_realized_session_pnl
    current = risk.realized_session_pnl
    if peak <= 0:
        return "not applicable (peak was never positive)"
    return f"{((peak - current) / peak) * Decimal(100):.1f}%"


def _remaining_loss(risk: AccountRiskSnapshot | None, maximum_daily_loss: Decimal) -> object:
    if risk is None:
        return "UNAVAILABLE"
    current = risk.realized_session_pnl
    return max(Decimal(0), maximum_daily_loss + current)


def _gain_percent(frame: MarketFrame) -> object:
    quote = frame.quote
    if quote is None:
        return "UNAVAILABLE"
    return ((quote.last - quote.previous_close) / quote.previous_close) * Decimal(100)


def _catalyst(frame: MarketFrame) -> str:
    evidence = frame.catalyst_evidence
    if evidence is None:
        return "UNAVAILABLE"
    status = "verified" if evidence.verified else "unverified"
    return f"{status} — {evidence.description} ({evidence.source}, {evidence.provider_timestamp.isoformat()})"


def _float(frame: MarketFrame) -> str:
    evidence = frame.float_evidence
    if evidence is None:
        return "UNAVAILABLE"
    status = "verified" if evidence.verified else "unverified"
    return f"{evidence.shares:,} shares, {status} ({evidence.source}, {evidence.provider_timestamp.isoformat()})"
