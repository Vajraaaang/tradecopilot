from __future__ import annotations

from datetime import UTC, datetime, time
from decimal import Decimal
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tradecopilot.models import DataQuality, StrategyVersion, ThresholdOrigin


class StrategyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy_id: str = "ross_first_pullback"
    version: str = "1.0.0-candidate"
    source_video_id: str = "xGIa8Vg0PWM"
    deployment_status: str = "shadow"
    execution_mode: str = "manual_only"

    minimum_source_style_rvol: Decimal = Decimal("5")
    minimum_percentage_gain: Decimal = Decimal("10")
    preferred_percentage_gain: Decimal = Decimal("30")
    minimum_pillars: int = 4
    maximum_float_shares: int = 20_000_000
    source_price_minimum: Decimal = Decimal("2")
    source_price_maximum: Decimal = Decimal("20")
    preferred_price_minimum: Decimal = Decimal("5")
    preferred_price_maximum: Decimal = Decimal("10")
    market_leader_exception_rvol: Decimal = Decimal("10")
    market_leader_exception_volume: int = 2_000_000
    market_leader_exception_gain: Decimal = Decimal("30")

    maximum_retracement_fraction: Decimal = Decimal("0.50")
    minimum_reward_risk: Decimal = Decimal("2.0")
    maximum_risk_per_trade_usd: Decimal = Decimal("25")
    maximum_daily_loss_usd: Decimal = Decimal("75")
    maximum_consecutive_losses: int = 3
    trading_cutoff_eastern: time = time(10, 0)

    maximum_spread_percentage: Decimal = Decimal("1.5")
    maximum_extension_atr: Decimal = Decimal("1.5")
    maximum_quote_age_seconds: float = 2.0
    maximum_level2_age_seconds: float = 2.0
    maximum_one_minute_bar_age_seconds: float = 90.0
    heartbeat_seconds: float = 15.0
    stop_buffer: Decimal = Decimal("0.01")
    slippage_safety_buffer: Decimal = Decimal("0.005")

    impulse_lookback_bars: int = 8
    impulse_minimum_bars: int = 3
    impulse_minimum_gain_percentage: Decimal = Decimal("6")
    pullback_minimum_bars: int = 2
    pullback_maximum_bars: int = 4
    breakout_volume_ratio: Decimal = Decimal("0.80")
    severe_topping_tail_fraction: Decimal = Decimal("0.45")
    no_follow_through_bars: int = 3

    seller_size_multiple_of_recent_median: Decimal = Decimal("4")
    minimum_seller_persistence_snapshots: int = 3
    minimum_seller_persistence_seconds: float = 2.0
    failed_break_attempt_count: int = 3

    opening_range_minutes: int = 5
    vwap_session: str = "extended_hours"
    poll_interval_seconds: float = 1.0
    selected_quote_interval_seconds: float = 1.0
    screener_interval_seconds: float = 2.5
    level2_interval_seconds: float = 1.5
    position_interval_seconds: float = 2.0
    account_pnl_interval_seconds: float = 5.0
    one_minute_bars_interval_seconds: float = 5.0
    five_minute_bars_interval_seconds: float = 60.0
    maximum_poll_retries: int = 4
    circuit_breaker_failures: int = 5
    maximum_backoff_seconds: float = 30.0
    raw_snapshot_retention_rows: int = 10_000

    origins: dict[str, ThresholdOrigin] = Field(
        default_factory=lambda: {
            "strategy_id": ThresholdOrigin.SOURCE_DERIVED,
            "version": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "source_video_id": ThresholdOrigin.SOURCE_DERIVED,
            "deployment_status": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "execution_mode": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "minimum_source_style_rvol": ThresholdOrigin.SOURCE_DERIVED,
            "minimum_percentage_gain": ThresholdOrigin.SOURCE_DERIVED,
            "preferred_percentage_gain": ThresholdOrigin.SOURCE_DERIVED,
            "minimum_pillars": ThresholdOrigin.SOURCE_DERIVED,
            "maximum_float_shares": ThresholdOrigin.SOURCE_DERIVED,
            "source_price_minimum": ThresholdOrigin.SOURCE_DERIVED,
            "source_price_maximum": ThresholdOrigin.SOURCE_DERIVED,
            "preferred_price_minimum": ThresholdOrigin.SOURCE_DERIVED,
            "preferred_price_maximum": ThresholdOrigin.SOURCE_DERIVED,
            "market_leader_exception_rvol": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "market_leader_exception_volume": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "market_leader_exception_gain": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "maximum_retracement_fraction": ThresholdOrigin.SOURCE_DERIVED,
            "minimum_reward_risk": ThresholdOrigin.SOURCE_DERIVED,
            "trading_cutoff_eastern": ThresholdOrigin.SOURCE_DERIVED,
            "maximum_risk_per_trade_usd": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "maximum_daily_loss_usd": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "maximum_consecutive_losses": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "maximum_spread_percentage": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "maximum_extension_atr": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "maximum_quote_age_seconds": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "maximum_level2_age_seconds": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "maximum_one_minute_bar_age_seconds": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "heartbeat_seconds": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "stop_buffer": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "slippage_safety_buffer": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "impulse_lookback_bars": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "impulse_minimum_bars": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "impulse_minimum_gain_percentage": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "pullback_minimum_bars": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "pullback_maximum_bars": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "breakout_volume_ratio": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "severe_topping_tail_fraction": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "no_follow_through_bars": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "seller_size_multiple_of_recent_median": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "minimum_seller_persistence_snapshots": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "minimum_seller_persistence_seconds": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "failed_break_attempt_count": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "opening_range_minutes": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "vwap_session": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "poll_interval_seconds": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "selected_quote_interval_seconds": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "screener_interval_seconds": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "level2_interval_seconds": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "position_interval_seconds": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "account_pnl_interval_seconds": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "one_minute_bars_interval_seconds": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "five_minute_bars_interval_seconds": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "maximum_poll_retries": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "circuit_breaker_failures": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "maximum_backoff_seconds": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
            "raw_snapshot_retention_rows": ThresholdOrigin.IMPLEMENTATION_DEFAULT,
        }
    )

    @model_validator(mode="after")
    def require_parameter_provenance(self) -> Self:
        expected = set(type(self).model_fields) - {"origins"}
        if set(self.origins) != expected:
            missing = sorted(expected - set(self.origins))
            unknown = sorted(set(self.origins) - expected)
            raise ValueError(f"parameter provenance mismatch; missing={missing}, unknown={unknown}")
        return self

    def strategy_version(self) -> StrategyVersion:
        now = datetime.now(UTC)
        parameters: dict[str, Any] = self.model_dump(exclude={"origins"}, mode="json")
        return StrategyVersion(
            provider_timestamp=now,
            receipt_timestamp=now,
            age_seconds=0,
            source="local_configuration",
            quality=DataQuality.GOOD,
            strategy_id=self.strategy_id,
            version=self.version,
            source_video_id=self.source_video_id,
            deployment_status=self.deployment_status,
            execution_mode=self.execution_mode,
            parameters=parameters,
            parameter_origins=self.origins,
        )
