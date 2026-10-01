from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from tradecopilot.config import StrategyConfig
from tradecopilot.features import ImpulseDetector, PullbackDetector
from tradecopilot.indicators import InsufficientIndicators, calculate_indicators
from tradecopilot.level2 import Level2Analyzer
from tradecopilot.models import (
    DataQuality,
    DecisionState,
    ExitSignal,
    IndicatorSnapshot,
    MarketFrame,
    MomentumImpulse,
    PillarResult,
    PositionSnapshot,
    Pullback,
    StrategyDecision,
    TradePlan,
)
from tradecopilot.risk import estimate_slippage, maximum_shares, risk_per_share

EASTERN = ZoneInfo("America/New_York")


class DecisionEngine:
    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        self.impulses = ImpulseDetector(config)
        self.pullbacks = PullbackDetector(config)
        self.level2 = Level2Analyzer(config)
        self.previous_decision: StrategyDecision | None = None
        self.previous_price: Decimal | None = None
        self.active_plan: TradePlan | None = None
        self.buy_time: datetime | None = None
        self.awaiting_reentry = False

    def evaluate(self, frame: MarketFrame) -> StrategyDecision:
        quote = frame.quote
        previous_state = self.previous_decision.state if self.previous_decision else None
        if quote is None:
            if frame.price_snapshot is not None:
                return self._finish(
                    self._decision(
                        frame,
                        DecisionState.DATA_INSUFFICIENT,
                        ("Price-only feed: bid/ask and minute OHLCV data are unavailable",),
                        failed=("complete_market_data",),
                        missing=("bid/ask quote", "minute OHLCV bars", "account/position risk context"),
                        changes=("Connect complete market and account data before using strategy or Jev advice",),
                    )
                )
            return self._finish(
                self._decision(
                    frame,
                    DecisionState.DATA_INSUFFICIENT,
                    ("Quote is unavailable",),
                    failed=("quote_present",),
                    missing=("quote",),
                )
            )

        position_open = frame.position is not None and frame.position.quantity > 0
        freshness_failures = self._freshness_failures(frame)
        if freshness_failures:
            reasons = list(freshness_failures)
            changes = ["Wait for fresh, internally consistent quote and Level 2 data"]
            if position_open:
                reasons.append("Open position is not labeled HOLD while market data is stale")
                if self.active_plan:
                    reasons.append(f"Last known structural stop: {self.active_plan.structural_stop}")
                changes = ["Broker-side/manual risk controls must govern until fresh data returns"]
            return self._finish(
                self._decision(
                    frame,
                    DecisionState.DATA_STALE,
                    tuple(reasons),
                    failed=("data_fresh",),
                    missing=tuple(freshness_failures),
                    changes=tuple(changes),
                    plan=self.active_plan,
                )
            )

        if frame.account_risk is None:
            return self._finish(
                self._decision(
                    frame,
                    DecisionState.DATA_INSUFFICIENT,
                    ("Account risk snapshot is unavailable",),
                    failed=("account_risk_present",),
                    missing=("account risk snapshot",),
                    plan=self.active_plan,
                )
            )

        day_stop_reasons = self._day_stop_reasons(frame)
        if day_stop_reasons and not position_open:
            return self._finish(
                self._decision(
                    frame,
                    DecisionState.DAY_STOP,
                    tuple(day_stop_reasons),
                    passed=("session_lock_enforced",),
                    failed=("day_risk_open",),
                    changes=("No new entry is allowed until the next market session",),
                )
            )

        try:
            indicator = calculate_indicators(
                quote,
                frame.bars_1m,
                frame.bars_5m,
                frame.resistance_levels,
                self.config,
            )
        except InsufficientIndicators as exc:
            return self._finish(
                self._decision(
                    frame,
                    DecisionState.DATA_INSUFFICIENT,
                    (str(exc),),
                    failed=("indicator_history_complete",),
                    missing=("indicator history",),
                    plan=self.active_plan,
                )
            )

        exit_signals = self.level2.signals(frame.level2_history, frame.time_and_sales)
        if position_open:
            position_decision = self._manage_position(frame, indicator, exit_signals)
            if position_decision.state == DecisionState.SELL:
                return self._finish(position_decision)

        if day_stop_reasons:
            return self._finish(
                self._decision(
                    frame,
                    DecisionState.DAY_STOP,
                    tuple(day_stop_reasons),
                    passed=("session_lock_enforced",),
                    failed=("day_risk_open",),
                    changes=("No new entry is allowed until the next market session",),
                    indicator=indicator,
                    plan=self.active_plan,
                    position=frame.position,
                    exit_signals=exit_signals,
                )
            )

        if position_open:
            return self._finish(position_decision)

        pillars, pillar_missing = self._pillars(frame, indicator)
        if pillar_missing:
            return self._finish(
                self._decision(
                    frame,
                    DecisionState.DATA_INSUFFICIENT,
                    tuple(pillar_missing),
                    failed=("verified_float_present",),
                    missing=tuple(pillar_missing),
                    indicator=indicator,
                    pillars=pillars,
                    exit_signals=exit_signals,
                )
            )
        passed_count = sum(pillar.passed for pillar in pillars)
        float_passed = next(pillar.passed for pillar in pillars if pillar.name == "public_float")
        if not float_passed or passed_count < self.config.minimum_pillars:
            return self._finish(
                self._decision(
                    frame,
                    DecisionState.NO_TRADE,
                    (f"Only {passed_count}/5 stock-selection pillars passed",),
                    passed=tuple(p.name for p in pillars if p.passed),
                    failed=tuple(p.name for p in pillars if not p.passed),
                    changes=(
                        f"Require verified float below {self.config.maximum_float_shares:,} and at least four pillars",
                    ),
                    indicator=indicator,
                    pillars=pillars,
                    exit_signals=exit_signals,
                )
            )

        impulse = self.impulses.detect(frame.bars_1m)
        if impulse is None or not impulse.valid:
            return self._finish(
                self._decision(
                    frame,
                    DecisionState.WATCH,
                    ("Stock qualifies, but no valid momentum impulse is complete",),
                    passed=("stock_selection",),
                    failed=("momentum_impulse",),
                    changes=("Wait for a clear multi-bar momentum impulse",),
                    indicator=indicator,
                    pillars=pillars,
                    impulse=impulse,
                    exit_signals=exit_signals,
                )
            )
        pullback = self.pullbacks.detect(frame.bars_1m, impulse)
        if pullback is None:
            return self._finish(
                self._decision(
                    frame,
                    DecisionState.WATCH,
                    ("Momentum impulse is valid; a two-bar controlled pullback has not formed",),
                    passed=("stock_selection", "momentum_impulse"),
                    failed=("controlled_pullback",),
                    changes=("Wait for at least two controlled red pullback candles",),
                    indicator=indicator,
                    pillars=pillars,
                    impulse=impulse,
                    exit_signals=exit_signals,
                )
            )
        if not pullback.valid:
            return self._finish(
                self._decision(
                    frame,
                    DecisionState.NO_TRADE,
                    pullback.reasons,
                    passed=("stock_selection", "momentum_impulse"),
                    failed=("controlled_pullback",),
                    changes=("A fresh impulse and controlled pullback must form",),
                    indicator=indicator,
                    pillars=pillars,
                    impulse=impulse,
                    pullback=pullback,
                    exit_signals=exit_signals,
                )
            )

        setup_failures = self._setup_failures(frame, indicator, impulse, pullback, exit_signals)
        plan = self._trade_plan(frame, indicator, pullback)
        if plan.reward_risk < self.config.minimum_reward_risk:
            setup_failures.append(f"reward-to-risk {plan.reward_risk:.2f} is below {self.config.minimum_reward_risk}")
        if setup_failures:
            return self._finish(
                self._decision(
                    frame,
                    DecisionState.NO_TRADE,
                    tuple(setup_failures),
                    passed=("stock_selection", "momentum_impulse", "controlled_pullback"),
                    failed=tuple(setup_failures),
                    changes=("All failed setup checks must clear before entry",),
                    indicator=indicator,
                    pillars=pillars,
                    impulse=impulse,
                    pullback=pullback,
                    plan=plan,
                    exit_signals=exit_signals,
                )
            )

        if quote.last < plan.trigger_price:
            self.active_plan = plan
            state = DecisionState.REENTRY_WATCH if self.awaiting_reentry else DecisionState.ARMED
            return self._finish(
                self._decision(
                    frame,
                    state,
                    (
                        "Valid first-pullback setup; strict trigger has not crossed",
                        f"BUY trigger is {plan.trigger_price}",
                    ),
                    passed=(
                        "stock_selection",
                        "momentum_impulse",
                        "controlled_pullback",
                        "above_vwap",
                        "above_ema9",
                        "minimum_2r",
                    ),
                    changes=(f"A fresh trade at or above {plan.trigger_price} would trigger BUY",),
                    indicator=indicator,
                    pillars=pillars,
                    impulse=impulse,
                    pullback=pullback,
                    plan=plan,
                    exit_signals=exit_signals,
                )
            )

        crossing_fresh = (
            previous_state in {DecisionState.ARMED, DecisionState.REENTRY_WATCH}
            and self.previous_price is not None
            and self.previous_price < plan.trigger_price <= quote.last
            and quote.age_seconds <= self.config.maximum_quote_age_seconds
            and self.previous_decision is not None
            and quote.provider_timestamp > self.previous_decision.provider_timestamp
        )
        entry_failures = self._entry_failures(frame, plan, pullback, exit_signals)
        if not crossing_fresh:
            entry_failures.append("strict trigger crossing was not observed fresh")
        if entry_failures:
            return self._finish(
                self._decision(
                    frame,
                    DecisionState.NO_TRADE,
                    tuple(entry_failures),
                    passed=("qualified_pullback",),
                    failed=tuple(entry_failures),
                    changes=("Wait for a new pullback and a newly observed trigger cross",),
                    indicator=indicator,
                    pillars=pillars,
                    impulse=impulse,
                    pullback=pullback,
                    plan=plan,
                    exit_signals=exit_signals,
                )
            )

        self.active_plan = plan
        self.buy_time = frame.event_time
        return self._finish(
            self._decision(
                frame,
                DecisionState.BUY,
                (
                    f"Fresh strict trigger crossing at {plan.trigger_price}",
                    "All deterministic entry checks remain valid",
                    "Analysis signal only; execution is manual",
                ),
                passed=(
                    "fresh_trigger_cross",
                    "above_vwap",
                    "above_ema9",
                    "breakout_volume",
                    "spread",
                    "liquidity",
                    "minimum_2r",
                ),
                changes=(f"Manual entry is invalidated at structural stop {plan.structural_stop}",),
                indicator=indicator,
                pillars=pillars,
                impulse=impulse,
                pullback=pullback,
                plan=plan,
                exit_signals=exit_signals,
                crossing_fresh=True,
            )
        )

    def _manage_position(
        self,
        frame: MarketFrame,
        indicator: IndicatorSnapshot,
        exit_signals: Sequence[ExitSignal],
    ) -> StrategyDecision:
        quote = frame.quote
        position = frame.position
        assert quote is not None and position is not None
        plan = self.active_plan
        if plan is None or position.average_entry is None:
            return self._decision(
                frame,
                DecisionState.SELL,
                ("Open position cannot be reconciled with an immutable risk plan",),
                failed=("position_reconciled",),
                changes=("Manually reconcile or close exposure; analysis signal only",),
                indicator=indicator,
                position=position,
                exit_signals=tuple(exit_signals),
            )
        confirmed = [signal for signal in exit_signals if signal.confirmed]
        if quote.last <= plan.structural_stop:
            return self._position_decision(
                frame,
                DecisionState.SELL,
                (f"Structural stop {plan.structural_stop} was reached",),
                indicator,
                exit_signals,
                failed=("structural_stop_intact",),
            )
        if position.latest_higher_low is not None and quote.last < position.latest_higher_low:
            return self._position_decision(
                frame,
                DecisionState.SELL,
                (f"Latest meaningful higher low {position.latest_higher_low} broke",),
                indicator,
                exit_signals,
                failed=("latest_higher_low_intact",),
            )
        if not frame.tradability_known or frame.halted:
            return self._position_decision(
                frame,
                DecisionState.SELL,
                ("Tradability or halt state is unsafe/uncertain",),
                indicator,
                exit_signals,
                failed=("tradability_known",),
            )
        if confirmed:
            evidence = tuple(f"{signal.name}: {'; '.join(signal.evidence)}" for signal in confirmed)
            return self._position_decision(
                frame,
                DecisionState.SELL,
                (*evidence, "Confirmed exit evidence invalidated the thesis"),
                indicator,
                exit_signals,
                failed=tuple(signal.name for signal in confirmed),
            )
        if self._failed_breakout(frame, plan):
            return self._position_decision(
                frame,
                DecisionState.SELL,
                (f"Breakout lost trigger {plan.trigger_price} and failed to reclaim",),
                indicator,
                exit_signals,
                failed=("trigger_reclaim",),
            )
        if self._strong_vwap_loss(frame, indicator):
            return self._position_decision(
                frame,
                DecisionState.SELL,
                ("VWAP was lost on strong red volume",),
                indicator,
                exit_signals,
                failed=("vwap_intact",),
            )
        if self._no_follow_through(frame, plan):
            return self._position_decision(
                frame,
                DecisionState.SELL,
                (f"Breakout made no meaningful progress within {self.config.no_follow_through_bars} bars",),
                indicator,
                exit_signals,
                failed=("timely_follow_through",),
            )
        if self._bid_disappeared_with_spread_expansion(frame):
            return self._position_decision(
                frame,
                DecisionState.SELL,
                ("Displayed bid support disappeared while the spread expanded sharply",),
                indicator,
                exit_signals,
                failed=("bid_support_and_spread",),
            )
        topping_tail = self._topping_tail(frame)
        slowdown = self._buying_slowdown(frame)
        if topping_tail and slowdown:
            return self._position_decision(
                frame,
                DecisionState.EXIT_WARNING,
                ("First meaningful topping wick is present", "Buying progress is slowing"),
                indicator,
                exit_signals,
                failed=("clean_price_progress",),
                changes=(
                    f"SELL if price loses {position.latest_higher_low or plan.structural_stop} "
                    "or deterioration confirms",
                ),
            )
        hold_reasons = (
            f"Structural stop {plan.structural_stop} remains intact",
            f"Latest higher low {position.latest_higher_low or plan.structural_stop} remains intact",
            "Price remains above VWAP and no confirmed exit signal is active",
        )
        return self._position_decision(
            frame,
            DecisionState.HOLD,
            hold_reasons,
            indicator,
            exit_signals,
            passed=("structural_stop_intact", "thesis_intact", "no_confirmed_exit"),
            changes=(
                f"SELL on a break of {position.latest_higher_low or plan.structural_stop} or confirmed exit signal",
            ),
        )

    def _position_decision(
        self,
        frame: MarketFrame,
        state: DecisionState,
        reasons: tuple[str, ...],
        indicator: IndicatorSnapshot,
        exit_signals: Sequence[ExitSignal],
        *,
        passed: tuple[str, ...] = (),
        failed: tuple[str, ...] = (),
        changes: tuple[str, ...] = ("Continue monitoring deterministic invalidation levels",),
    ) -> StrategyDecision:
        return self._decision(
            frame,
            state,
            reasons + (("Analysis signal only; execution is manual",) if state == DecisionState.SELL else ()),
            passed=passed,
            failed=failed,
            changes=changes,
            indicator=indicator,
            plan=self.active_plan,
            position=frame.position,
            exit_signals=tuple(exit_signals),
        )

    def _pillars(self, frame: MarketFrame, indicator: IndicatorSnapshot) -> tuple[tuple[PillarResult, ...], list[str]]:
        quote = frame.quote
        assert quote is not None
        evidence = frame.float_evidence
        if evidence is None or not evidence.verified:
            return (
                (
                    PillarResult(
                        name="public_float",
                        passed=False,
                        evidence="Verified public float is missing",
                        required=True,
                    ),
                ),
                ["verified public float"],
            )
        float_ok = evidence.shares < self.config.maximum_float_shares
        gain = ((quote.last - quote.previous_close) / quote.previous_close) * Decimal(100)
        catalyst_ok = bool(frame.catalyst_evidence and frame.catalyst_evidence.verified)
        leader_exception = (
            frame.market_leader
            and indicator.source_style_rvol >= self.config.market_leader_exception_rvol
            and quote.total_volume >= self.config.market_leader_exception_volume
            and gain >= self.config.market_leader_exception_gain
        )
        price_ok = (
            self.config.source_price_minimum <= quote.session_origin_price <= self.config.source_price_maximum
            or self.config.source_price_minimum <= quote.last <= self.config.source_price_maximum
        )
        pillars = (
            PillarResult(
                name="relative_volume",
                passed=indicator.source_style_rvol >= self.config.minimum_source_style_rvol,
                evidence=f"source-style RVOL {indicator.source_style_rvol:.2f}x",
            ),
            PillarResult(
                name="percentage_gain",
                passed=gain >= self.config.minimum_percentage_gain,
                evidence=f"session gain {gain:.2f}%",
            ),
            PillarResult(
                name="catalyst",
                passed=catalyst_ok or leader_exception,
                evidence=(
                    "verified catalyst"
                    if catalyst_ok
                    else "explicit market-leader exception"
                    if leader_exception
                    else "no verified catalyst or exception"
                ),
            ),
            PillarResult(
                name="price",
                passed=price_ok,
                evidence=f"origin {quote.session_origin_price}; current {quote.last}",
            ),
            PillarResult(
                name="public_float",
                passed=float_ok,
                evidence=f"verified float {evidence.shares:,} shares from {evidence.source}",
                required=True,
            ),
        )
        return pillars, []

    def _setup_failures(
        self,
        frame: MarketFrame,
        indicator: IndicatorSnapshot,
        impulse: MomentumImpulse,
        pullback: Pullback,
        exit_signals: Sequence[ExitSignal],
    ) -> list[str]:
        quote = frame.quote
        assert quote is not None
        failures: list[str] = []
        if quote.last <= indicator.vwap:
            failures.append("price is not above VWAP")
        if quote.last <= indicator.ema9_1m:
            failures.append("price is not above EMA9")
        if pullback.average_red_volume >= impulse.average_green_volume:
            failures.append("red pullback volume is not below green impulse volume")
        if indicator.spread_percent > self.config.maximum_spread_percentage:
            failures.append("spread exceeds configured maximum")
        support_extension_atr = (
            (quote.last - indicator.structural_support) / indicator.atr14_1m
            if indicator.atr14_1m > 0
            else Decimal("Infinity")
        )
        if support_extension_atr > self.config.maximum_extension_atr:
            failures.append("price extension from structural support exceeds configured ATR maximum")
        if indicator.five_minute_structure == "lower-high/lower-low downtrend":
            failures.append("five-minute lower-high/lower-low downtrend veto")
        if self._topping_tail(frame):
            failures.append("severe topping-tail rejection is directly overhead")
        if any(signal.confirmed for signal in exit_signals):
            failures.append("an exit signal is already confirmed")
        if not frame.tradability_known or frame.halted:
            failures.append("tradability or halt state is uncertain")
        return failures

    def _trade_plan(self, frame: MarketFrame, indicator: IndicatorSnapshot, pullback: Pullback) -> TradePlan:
        quote = frame.quote
        assert quote is not None
        stop = pullback.low - self.config.stop_buffer
        best_sizes = []
        if frame.level2_history:
            latest = frame.level2_history[-1]
            best_sizes = [level.size for level in (*latest.bids[:1], *latest.asks[:1])]
        slippage = estimate_slippage(
            quote.ask - quote.bid,
            min(best_sizes, default=0),
            self.config,
        )
        per_share = risk_per_share(pullback.previous_candle_high, stop, slippage)
        candidates = [level for level in frame.resistance_levels if level > pullback.previous_candle_high]
        if indicator.structural_resistance > pullback.previous_candle_high:
            candidates.append(indicator.structural_resistance)
        resistance = min(candidates, default=pullback.previous_candle_high)
        reward = max(Decimal(0), resistance - pullback.previous_candle_high)
        ratio = reward / per_share if per_share else Decimal(0)
        return TradePlan(
            provider_timestamp=frame.event_time,
            receipt_timestamp=frame.event_time,
            age_seconds=0,
            source="deterministic_risk_engine",
            quality=DataQuality.GOOD,
            symbol=quote.symbol,
            trigger_price=pullback.previous_candle_high,
            structural_stop=stop,
            estimated_slippage=slippage,
            risk_per_share=per_share,
            maximum_dollar_risk=self.config.maximum_risk_per_trade_usd,
            maximum_shares=maximum_shares(self.config.maximum_risk_per_trade_usd, per_share),
            nearest_resistance=resistance,
            available_reward=reward,
            reward_risk=ratio,
            invalidation_level=stop,
        )

    def _entry_failures(
        self,
        frame: MarketFrame,
        plan: TradePlan,
        pullback: Pullback,
        exit_signals: Sequence[ExitSignal],
    ) -> list[str]:
        quote = frame.quote
        assert quote is not None
        failures: list[str] = []
        if quote.last < plan.trigger_price:
            failures.append("price has not crossed trigger")
        if quote.last - plan.trigger_price > plan.risk_per_share / Decimal(2):
            failures.append("entry would be an uncontrolled vertical chase")
        if quote.current_minute_volume < int(pullback.average_red_volume * self.config.breakout_volume_ratio):
            failures.append("breakout volume is not acceptable")
        if any(signal.confirmed for signal in exit_signals):
            failures.append("persistent seller or another exit signal blocks entry")
        return failures

    def _freshness_failures(self, frame: MarketFrame) -> list[str]:
        failures: list[str] = []
        if frame.quote is not None:
            quote_age = max(
                frame.quote.age_seconds,
                (frame.event_time - frame.quote.provider_timestamp).total_seconds(),
            )
            if quote_age > self.config.maximum_quote_age_seconds:
                failures.append(f"quote age {quote_age:.2f}s exceeds limit")
            if frame.quote.quality in {DataQuality.STALE, DataQuality.CONTRADICTORY}:
                failures.append(f"quote quality is {frame.quote.quality}")
        if frame.bars_1m:
            latest_bar = frame.bars_1m[-1]
            bar_age = max(
                latest_bar.age_seconds,
                (frame.event_time - latest_bar.provider_timestamp).total_seconds(),
            )
            if bar_age > self.config.maximum_one_minute_bar_age_seconds:
                failures.append(f"one-minute bar age {bar_age:.2f}s exceeds limit")
            if latest_bar.quality in {DataQuality.STALE, DataQuality.CONTRADICTORY}:
                failures.append(f"one-minute bar quality is {latest_bar.quality}")
        if frame.level2_history:
            latest = frame.level2_history[-1]
            age = max(
                latest.age_seconds,
                (frame.event_time - latest.provider_timestamp).total_seconds(),
            )
            if age > self.config.maximum_level2_age_seconds:
                failures.append(f"Level 2 age {age:.2f}s exceeds limit")
            if latest.quality in {DataQuality.STALE, DataQuality.CONTRADICTORY}:
                failures.append(f"Level 2 quality is {latest.quality}")
        else:
            failures.append("Level 2 snapshot is unavailable")
        if (
            frame.position is not None
            and frame.position.quantity > 0
            and frame.position.quality
            in {
                DataQuality.STALE,
                DataQuality.INSUFFICIENT,
                DataQuality.CONTRADICTORY,
            }
        ):
            failures.append(f"position quality is {frame.position.quality}")
        return failures

    def _day_stop_reasons(self, frame: MarketFrame) -> list[str]:
        risk = frame.account_risk
        if risk is None:
            return []
        reasons: list[str] = []
        if risk.session_locked:
            reasons.append("Session was already locked")
        if risk.peak_realized_session_pnl > 0 and risk.realized_session_pnl <= risk.peak_realized_session_pnl / Decimal(
            2
        ):
            reasons.append("At least half of positive peak realized P&L was given back")
        if risk.realized_session_pnl <= -self.config.maximum_daily_loss_usd:
            reasons.append("Maximum daily loss reached")
        if risk.consecutive_losses >= self.config.maximum_consecutive_losses:
            reasons.append("Maximum consecutive losses reached")
        if frame.event_time.astimezone(EASTERN).timetz().replace(tzinfo=None) >= self.config.trading_cutoff_eastern:
            reasons.append("Configured Eastern trading window closed")
        if frame.no_a_quality_candidates:
            reasons.append("No A-quality candidates remain")
        if frame.bearish_momentum_environment:
            reasons.append("Momentum names are repeatedly popping and rejecting")
        return reasons

    def _failed_breakout(self, frame: MarketFrame, plan: TradePlan) -> bool:
        if not frame.bars_1m or frame.quote is None:
            return False
        latest = frame.bars_1m[-1]
        return (
            latest.high >= plan.trigger_price
            and latest.close < plan.trigger_price
            and frame.quote.last < plan.trigger_price
        )

    def _strong_vwap_loss(self, frame: MarketFrame, indicator: IndicatorSnapshot) -> bool:
        if not frame.bars_1m or frame.quote is None:
            return False
        latest = frame.bars_1m[-1]
        prior_volume = [bar.volume for bar in frame.bars_1m[-6:-1]]
        average = sum(prior_volume) / len(prior_volume) if prior_volume else 0
        return frame.quote.last < indicator.vwap and latest.close < latest.open and latest.volume > average * 1.5

    def _topping_tail(self, frame: MarketFrame) -> bool:
        if not frame.bars_1m:
            return False
        latest = frame.bars_1m[-1]
        candle_range = latest.high - latest.low
        if candle_range <= 0:
            return False
        upper_wick = latest.high - max(latest.open, latest.close)
        return upper_wick / candle_range >= self.config.severe_topping_tail_fraction

    def _buying_slowdown(self, frame: MarketFrame) -> bool:
        if len(frame.bars_1m) < 3:
            return False
        recent = frame.bars_1m[-3:]
        progress = [recent[index].close - recent[index - 1].close for index in range(1, 3)]
        return progress[-1] < progress[0] and recent[-1].volume < recent[-2].volume

    def _no_follow_through(self, frame: MarketFrame, plan: TradePlan) -> bool:
        if self.buy_time is None:
            return False
        completed = [bar for bar in frame.bars_1m if bar.provider_timestamp > self.buy_time]
        if len(completed) < self.config.no_follow_through_bars:
            return False
        window = completed[: self.config.no_follow_through_bars]
        meaningful_progress = plan.risk_per_share / Decimal(2)
        return max(bar.high for bar in window) < plan.trigger_price + meaningful_progress

    def _bid_disappeared_with_spread_expansion(self, frame: MarketFrame) -> bool:
        if len(frame.level2_history) < 2:
            return False
        previous, current = frame.level2_history[-2:]
        if not previous.bids or not previous.asks or not current.bids or not current.asks:
            return False
        previous_bid_size = previous.bids[0].size
        current_bid_size = current.bids[0].size
        if previous_bid_size <= 0:
            return False
        previous_spread = previous.asks[0].price - previous.bids[0].price
        current_spread = current.asks[0].price - current.bids[0].price
        return Decimal(current_bid_size) / Decimal(previous_bid_size) <= Decimal("0.25") and current_spread >= max(
            previous_spread * Decimal(2), Decimal("0.02")
        )

    def _decision(
        self,
        frame: MarketFrame,
        state: DecisionState,
        reasons: tuple[str, ...],
        *,
        passed: tuple[str, ...] = (),
        failed: tuple[str, ...] = (),
        missing: tuple[str, ...] = (),
        changes: tuple[str, ...] = ("Wait for the next deterministic market event",),
        indicator: IndicatorSnapshot | None = None,
        pillars: tuple[PillarResult, ...] = (),
        impulse: MomentumImpulse | None = None,
        pullback: Pullback | None = None,
        plan: TradePlan | None = None,
        position: PositionSnapshot | None = None,
        exit_signals: tuple[ExitSignal, ...] = (),
        crossing_fresh: bool = False,
    ) -> StrategyDecision:
        quote = frame.quote
        price = quote or frame.price_snapshot
        source_timestamps = tuple(
            sorted(
                {
                    *([price.provider_timestamp] if price else []),
                    *(bar.provider_timestamp for bar in frame.bars_1m[-2:]),
                    *(snapshot.provider_timestamp for snapshot in frame.level2_history[-3:]),
                }
            )
        )
        previous_state = self.previous_decision.state if self.previous_decision else None
        changed_parts = [
            f"State changed from {previous_state} to {state}" if previous_state != state else f"State remains {state}"
        ]
        previous = self.previous_decision
        if quote is not None and previous is not None and previous.current_price is not None:
            changed_parts.append(f"last price moved {quote.last - previous.current_price:+}")
        if indicator is not None and previous is not None and previous.indicator is not None:
            prior_above_vwap = previous.current_price is not None and previous.current_price > previous.indicator.vwap
            current_above_vwap = quote is not None and quote.last > indicator.vwap
            if prior_above_vwap != current_above_vwap:
                changed_parts.append("price reclaimed VWAP" if current_above_vwap else "price lost VWAP")
            prior_above_ema9 = (
                previous.current_price is not None and previous.current_price > previous.indicator.ema9_1m
            )
            current_above_ema9 = quote is not None and quote.last > indicator.ema9_1m
            if prior_above_ema9 != current_above_ema9:
                changed_parts.append("price reclaimed EMA9" if current_above_ema9 else "price lost EMA9")
        changed = "; ".join(changed_parts)
        decision_provider_time = price.provider_timestamp if price else frame.event_time
        decision_age = max(0.0, (frame.event_time - decision_provider_time).total_seconds())
        return StrategyDecision(
            provider_timestamp=decision_provider_time,
            receipt_timestamp=frame.event_time,
            age_seconds=decision_age,
            source="deterministic_strategy_engine",
            quality=(
                DataQuality.STALE
                if state == DecisionState.DATA_STALE
                else DataQuality.INSUFFICIENT
                if state == DecisionState.DATA_INSUFFICIENT
                else DataQuality.GOOD
            ),
            symbol=price.symbol if price else "UNKNOWN",
            state=state,
            previous_state=previous_state,
            strategy_id=self.config.strategy_id,
            strategy_version=self.config.version,
            mode=frame.mode,
            reasons=reasons,
            rules_passed=passed,
            rules_failed=failed,
            missing_data=missing,
            what_changes_the_decision=changes,
            source_data_timestamps=source_timestamps,
            pillars=pillars,
            indicator=indicator,
            impulse=impulse,
            pullback=pullback,
            trade_plan=plan,
            position=position if position is not None else frame.position,
            exit_signals=exit_signals,
            current_price=price.last if price else None,
            crossing_fresh=crossing_fresh,
            changed_summary=changed,
        )

    def _finish(self, decision: StrategyDecision) -> StrategyDecision:
        if decision.state == DecisionState.SELL:
            self.awaiting_reentry = True
        elif decision.state == DecisionState.BUY:
            self.awaiting_reentry = False
        self.previous_decision = decision
        self.previous_price = decision.current_price
        return decision
