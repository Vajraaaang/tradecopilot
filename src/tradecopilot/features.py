from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

from tradecopilot.config import StrategyConfig
from tradecopilot.models import DataQuality, MomentumImpulse, OHLCVBar, Pullback


def retracement_fraction(impulse_low: Decimal, impulse_high: Decimal, pullback_low: Decimal) -> Decimal:
    impulse_range = impulse_high - impulse_low
    if impulse_range <= 0:
        raise ValueError("impulse high must be greater than impulse low")
    return (impulse_high - pullback_low) / impulse_range


class ImpulseDetector:
    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    def detect(self, bars: Sequence[OHLCVBar]) -> MomentumImpulse | None:
        if len(bars) < self.config.impulse_minimum_bars:
            return None
        pullback_count = _trailing_red_count(bars, self.config.pullback_maximum_bars)
        end_exclusive = len(bars) - pullback_count
        if end_exclusive < self.config.impulse_minimum_bars:
            return None
        start_window = max(0, end_exclusive - self.config.impulse_lookback_bars)
        window = bars[start_window:end_exclusive]
        low_offset = min(range(len(window)), key=lambda index: window[index].low)
        after_low = window[low_offset:]
        high_offset = low_offset + max(range(len(after_low)), key=lambda index: after_low[index].high)
        start_index = start_window + low_offset
        end_index = start_window + high_offset
        impulse_bars = bars[start_index : end_index + 1]
        if len(impulse_bars) < self.config.impulse_minimum_bars:
            return None
        low = impulse_bars[0].low
        high = max(bar.high for bar in impulse_bars)
        gain_percentage = ((high - low) / low) * Decimal(100)
        green_volumes = [Decimal(bar.volume) for bar in impulse_bars if bar.close > bar.open]
        valid = gain_percentage >= self.config.impulse_minimum_gain_percentage and len(green_volumes) >= 2
        reasons = (
            f"{len(impulse_bars)} bars classified as impulse",
            f"impulse gain {gain_percentage:.2f}%",
            f"{len(green_volumes)} green impulse bars",
        )
        latest = impulse_bars[-1]
        return MomentumImpulse(
            provider_timestamp=latest.provider_timestamp,
            receipt_timestamp=latest.receipt_timestamp,
            age_seconds=latest.age_seconds,
            source="deterministic_impulse_detector",
            quality=DataQuality.GOOD,
            symbol=latest.symbol,
            start_index=start_index,
            end_index=end_index,
            low=low,
            high=high,
            average_green_volume=(
                sum(green_volumes, Decimal(0)) / Decimal(len(green_volumes)) if green_volumes else Decimal(0)
            ),
            bar_timestamps=tuple(bar.provider_timestamp for bar in impulse_bars),
            valid=valid,
            reasons=reasons,
        )


class PullbackDetector:
    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    def detect(self, bars: Sequence[OHLCVBar], impulse: MomentumImpulse | None) -> Pullback | None:
        if impulse is None or not impulse.valid:
            return None
        count = _trailing_red_count(bars, self.config.pullback_maximum_bars)
        if count < self.config.pullback_minimum_bars:
            return None
        start_index = len(bars) - count
        pullback_bars = bars[start_index:]
        pullback_low = min(bar.low for bar in pullback_bars)
        if impulse.high <= impulse.low:
            return None
        retracement = retracement_fraction(impulse.low, impulse.high, pullback_low)
        red_volumes = [Decimal(bar.volume) for bar in pullback_bars if bar.close < bar.open]
        average_red = sum(red_volumes, Decimal(0)) / Decimal(len(red_volumes)) if red_volumes else Decimal(0)
        volume_contracts = average_red < impulse.average_green_volume
        valid = retracement <= self.config.maximum_retracement_fraction and volume_contracts
        reasons = (
            f"retracement {retracement * Decimal(100):.2f}%",
            "pullback volume contracted" if volume_contracts else "pullback volume did not contract",
            f"{count} red pullback bars",
        )
        latest = pullback_bars[-1]
        return Pullback(
            provider_timestamp=latest.provider_timestamp,
            receipt_timestamp=latest.receipt_timestamp,
            age_seconds=latest.age_seconds,
            source="deterministic_pullback_detector",
            quality=DataQuality.GOOD,
            symbol=latest.symbol,
            start_index=start_index,
            end_index=len(bars) - 1,
            low=pullback_low,
            retracement_fraction=retracement,
            average_red_volume=average_red,
            previous_candle_high=latest.high,
            bar_timestamps=tuple(bar.provider_timestamp for bar in pullback_bars),
            valid=valid,
            reasons=reasons,
        )


def _trailing_red_count(bars: Sequence[OHLCVBar], maximum: int) -> int:
    count = 0
    for bar in reversed(bars):
        if bar.close >= bar.open or count >= maximum:
            break
        count += 1
    return count
