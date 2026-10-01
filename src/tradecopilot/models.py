from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DataQuality(StrEnum):
    GOOD = "GOOD"
    DEGRADED = "DEGRADED"
    STALE = "STALE"
    INSUFFICIENT = "INSUFFICIENT"
    UNAVAILABLE = "UNAVAILABLE"
    LIMITED = "LIMITED"
    CONTRADICTORY = "CONTRADICTORY"


class DecisionState(StrEnum):
    DATA_STALE = "DATA_STALE"
    DATA_INSUFFICIENT = "DATA_INSUFFICIENT"
    NO_TRADE = "NO_TRADE"
    WATCH = "WATCH"
    ARMED = "ARMED"
    BUY = "BUY"
    HOLD = "HOLD"
    EXIT_WARNING = "EXIT_WARNING"
    SELL = "SELL"
    REENTRY_WATCH = "REENTRY_WATCH"
    DAY_STOP = "DAY_STOP"


class RunMode(StrEnum):
    MOCK = "mock"
    REPLAY = "replay"
    LIVE = "live"


class PositionChangeKind(StrEnum):
    ENTRY = "ENTRY"
    INCREASE = "INCREASE"
    REDUCTION = "REDUCTION"
    EXIT = "EXIT"


class ThresholdOrigin(StrEnum):
    SOURCE_DERIVED = "source_derived"
    IMPLEMENTATION_DEFAULT = "implementation_default"
    LEARNED_CANDIDATE = "learned_candidate"


class SignalAvailability(StrEnum):
    CLEAR = "CLEAR"
    ACTIVE = "ACTIVE"
    LIMITED = "LIMITED"
    UNAVAILABLE = "UNAVAILABLE"


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StampedModel(FrozenModel):
    provider_timestamp: datetime
    receipt_timestamp: datetime
    age_seconds: float = Field(ge=0)
    source: str = Field(min_length=1)
    quality: DataQuality

    @field_validator("provider_timestamp", "receipt_timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_calculated_age(self) -> Self:
        expected = (self.receipt_timestamp - self.provider_timestamp).total_seconds()
        if expected < 0:
            raise ValueError("receipt timestamp cannot precede provider timestamp")
        if abs(self.age_seconds - expected) > 0.001:
            raise ValueError("age_seconds does not match provider and receipt timestamps")
        return self


class PriceSnapshot(StampedModel):
    """A display price and optional session summary, without bid/ask or candles."""

    symbol: str = Field(pattern=r"^[A-Z][A-Z0-9.-]{0,9}$")
    last: Decimal = Field(gt=0)
    previous_close: Decimal = Field(gt=0)
    session_open: Decimal | None = Field(default=None, gt=0)
    session_high: Decimal | None = Field(default=None, gt=0)
    session_low: Decimal | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_session_range(self) -> Self:
        if (
            self.session_high is not None
            and self.session_low is not None
            and self.session_high < self.session_low
        ) or (
            self.session_open is not None
            and self.session_high is not None
            and self.session_open > self.session_high
        ) or (
            self.session_open is not None
            and self.session_low is not None
            and self.session_open < self.session_low
        ):
            raise ValueError("session open/high/low values are contradictory")
        return self


class Quote(StampedModel):
    symbol: str
    bid: Decimal = Field(gt=0)
    ask: Decimal = Field(gt=0)
    last: Decimal = Field(gt=0)
    previous_close: Decimal = Field(gt=0)
    total_volume: int = Field(ge=0)
    current_minute_volume: int = Field(ge=0)
    session_origin_price: Decimal = Field(gt=0)
    average_daily_volume_50d: int | None = Field(default=None, gt=0)
    average_cumulative_volume_same_time: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_book(self) -> Self:
        if self.ask < self.bid:
            raise ValueError("ask cannot be below bid")
        return self


class OHLCVBar(StampedModel):
    symbol: str
    timeframe: str
    open: Decimal = Field(gt=0)
    high: Decimal = Field(gt=0)
    low: Decimal = Field(gt=0)
    close: Decimal = Field(gt=0)
    volume: int = Field(ge=0)
    extended_hours: bool = False

    @model_validator(mode="after")
    def validate_ohlc(self) -> Self:
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError("OHLC values are contradictory")
        if self.high < self.low:
            raise ValueError("bar high cannot be below bar low")
        return self


class Level2Level(StampedModel):
    side: str
    price: Decimal = Field(gt=0)
    size: int = Field(ge=0)


class Level2Snapshot(StampedModel):
    symbol: str
    bids: tuple[Level2Level, ...]
    asks: tuple[Level2Level, ...]


class TimeAndSalesPrint(StampedModel):
    symbol: str
    price: Decimal = Field(gt=0)
    size: int = Field(gt=0)
    side: str


class PositionSnapshot(StampedModel):
    symbol: str
    account_alias: str | None = None
    quantity: Decimal = Field(ge=0)
    average_entry: Decimal | None = Field(default=None, gt=0)
    unrealized_pnl: Decimal | None = None
    latest_higher_low: Decimal | None = Field(default=None, gt=0)
    mfe: Decimal | None = None
    mae: Decimal | None = None


class PositionChange(StampedModel):
    symbol: str
    account_alias: str | None
    kind: PositionChangeKind
    prior_quantity: Decimal = Field(ge=0)
    new_quantity: Decimal = Field(ge=0)
    average_entry: Decimal | None = Field(default=None, gt=0)


class AccountRiskSnapshot(StampedModel):
    trading_date: date
    buying_power: Decimal = Field(ge=0)
    realized_session_pnl: Decimal
    peak_realized_session_pnl: Decimal
    unrealized_pnl: Decimal
    consecutive_losses: int = Field(ge=0)
    session_locked: bool


class CatalystEvidence(StampedModel):
    description: str
    verified: bool
    reference: str


class FloatEvidence(StampedModel):
    shares: int = Field(gt=0)
    verified: bool
    reference: str


class HistoricalContextEvidence(StampedModel):
    symbol: str
    average_daily_volume_50d: int = Field(gt=0)
    latest_daily_date: date
    ownership_snapshot_timestamp: datetime
    latest_filing_timestamp: datetime | None = None
    latest_filing_type: str | None = None
    latest_filing_url: str | None = None

    @field_validator("ownership_snapshot_timestamp", "latest_filing_timestamp")
    @classmethod
    def require_context_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("context timestamps must be timezone-aware")
        return value


class ScreenerCandidate(StampedModel):
    symbol: str
    scan_id: str
    scan_title: str
    columns: dict[str, str]
    float_shares: int | None = Field(default=None, gt=0)
    average_daily_volume_50d: int | None = Field(default=None, gt=0)


class IndicatorSnapshot(StampedModel):
    symbol: str
    session_definition: str
    vwap: Decimal
    vwap_slope: Decimal
    ema9_1m: Decimal
    ema9_slope_1m: Decimal
    ema20_1m: Decimal
    ema9_5m: Decimal
    ema20_5m: Decimal
    atr14_1m: Decimal
    rsi14_1m: Decimal | None
    macd_1m: Decimal | None
    macd_signal_1m: Decimal | None
    premarket_high: Decimal | None
    premarket_low: Decimal | None
    high_of_day: Decimal
    low_of_day: Decimal
    opening_range_high: Decimal | None
    opening_range_low: Decimal | None
    nearest_half_dollar: Decimal
    nearest_whole_dollar: Decimal
    structural_support: Decimal
    structural_resistance: Decimal
    distance_from_vwap: Decimal
    distance_from_vwap_percent: Decimal
    distance_from_vwap_atr: Decimal
    spread_dollars: Decimal
    spread_percent: Decimal
    source_style_rvol: Decimal
    time_adjusted_rvol: Decimal | None
    one_minute_structure: str
    five_minute_structure: str


class MomentumImpulse(StampedModel):
    symbol: str
    start_index: int = Field(ge=0)
    end_index: int = Field(ge=0)
    low: Decimal
    high: Decimal
    average_green_volume: Decimal
    bar_timestamps: tuple[datetime, ...]
    valid: bool
    reasons: tuple[str, ...]


class Pullback(StampedModel):
    symbol: str
    start_index: int = Field(ge=0)
    end_index: int = Field(ge=0)
    low: Decimal
    retracement_fraction: Decimal
    average_red_volume: Decimal
    previous_candle_high: Decimal
    bar_timestamps: tuple[datetime, ...]
    valid: bool
    reasons: tuple[str, ...]


class TradePlan(StampedModel):
    symbol: str
    trigger_price: Decimal
    structural_stop: Decimal
    estimated_slippage: Decimal
    risk_per_share: Decimal
    maximum_dollar_risk: Decimal
    maximum_shares: int = Field(ge=0)
    nearest_resistance: Decimal
    available_reward: Decimal
    reward_risk: Decimal
    invalidation_level: Decimal

    @model_validator(mode="after")
    def validate_risk_plan(self) -> Self:
        if self.trigger_price <= self.structural_stop:
            raise ValueError("long trigger must be above structural stop")
        if Decimal(self.maximum_shares) * self.risk_per_share > self.maximum_dollar_risk:
            raise ValueError("maximum shares exceed the configured dollar risk")
        return self


class ExitSignal(StampedModel):
    name: str
    severity: str
    confirmed: bool
    evidence: tuple[str, ...]
    availability: SignalAvailability


class PillarResult(FrozenModel):
    name: str
    passed: bool
    evidence: str
    required: bool = False


class StrategyDecision(StampedModel):
    symbol: str
    state: DecisionState
    previous_state: DecisionState | None
    strategy_id: str
    strategy_version: str
    mode: RunMode
    reasons: tuple[str, ...]
    rules_passed: tuple[str, ...]
    rules_failed: tuple[str, ...]
    missing_data: tuple[str, ...]
    what_changes_the_decision: tuple[str, ...]
    source_data_timestamps: tuple[datetime, ...]
    pillars: tuple[PillarResult, ...] = ()
    indicator: IndicatorSnapshot | None = None
    impulse: MomentumImpulse | None = None
    pullback: Pullback | None = None
    trade_plan: TradePlan | None = None
    position: PositionSnapshot | None = None
    exit_signals: tuple[ExitSignal, ...] = ()
    current_price: Decimal | None = None
    crossing_fresh: bool = False
    changed_summary: str = "Initial decision"


class StateTransition(StampedModel):
    symbol: str
    prior_state: DecisionState | None
    new_state: DecisionState
    strategy_version: str
    exact_reasons: tuple[str, ...]
    rules_passed: tuple[str, ...]
    rules_failed: tuple[str, ...]
    source_data_timestamps: tuple[datetime, ...]
    invalidation_level: Decimal | None
    trigger_level: Decimal | None
    active_position: PositionSnapshot | None


class JournalObservation(StampedModel):
    symbol: str
    trading_date: date
    strategy_version: str
    mode: RunMode
    decision: StrategyDecision
    compact_bars: tuple[dict[str, Any], ...]
    market_snapshot: dict[str, Any] = Field(default_factory=dict)
    supplemental_evidence: dict[str, Any] = Field(default_factory=dict)
    risk_snapshot: dict[str, Any] = Field(default_factory=dict)
    level2_sample: tuple[dict[str, Any], ...] = ()
    outcome_tracking: dict[str, Any] = Field(default_factory=dict)
    data_quality_problems: tuple[str, ...] = ()
    explanation_model: str | None = None
    prompt_version: str | None = None


class StrategyVersion(StampedModel):
    strategy_id: str
    version: str
    source_video_id: str
    deployment_status: str
    execution_mode: str
    parameters: dict[str, Any]
    parameter_origins: dict[str, ThresholdOrigin]


class ExperimentCandidate(StampedModel):
    candidate_id: str
    base_version: str
    candidate_version: str
    changed_parameter: str
    old_value: str
    new_value: str
    hypothesis: str
    causal_rationale: str
    training_window: str
    validation_window: str
    out_of_sample_window: str
    baseline_metrics: dict[str, float]
    candidate_metrics: dict[str, float]
    regressions: tuple[str, ...]
    data_quality_concerns: tuple[str, ...]
    overfitting_risks: tuple[str, ...]
    shadow_test_plan: str
    rollback_condition: str
    unseen_signals: int = Field(ge=0)
    out_of_sample_sessions: int = Field(ge=0)
    improved_chronological_windows: int = Field(ge=0)
    shadow_sessions: int = Field(ge=0)
    human_approval_status: str
    research_result: str


class MarketFrame(FrozenModel):
    event_time: datetime
    mode: RunMode
    quote: Quote | None
    bars_1m: tuple[OHLCVBar, ...]
    bars_5m: tuple[OHLCVBar, ...]
    level2_history: tuple[Level2Snapshot, ...]
    time_and_sales: tuple[TimeAndSalesPrint, ...] | None
    position: PositionSnapshot | None
    account_risk: AccountRiskSnapshot | None
    float_evidence: FloatEvidence | None
    catalyst_evidence: CatalystEvidence | None
    resistance_levels: tuple[Decimal, ...]
    gap_percent: Decimal | None
    market_leader: bool
    no_a_quality_candidates: bool
    bearish_momentum_environment: bool
    tradability_known: bool
    halted: bool
    historical_context: HistoricalContextEvidence | None = None
    screener_candidates: tuple[ScreenerCandidate, ...] = ()
    positions: tuple[PositionSnapshot, ...] = ()
    price_snapshot: PriceSnapshot | None = None
    price_history: tuple[PriceSnapshot, ...] = Field(default=(), max_length=60)

    @field_validator("event_time")
    @classmethod
    def require_event_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("event_time must be timezone-aware")
        return value
