from __future__ import annotations

from collections.abc import Sequence
from datetime import time, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from tradecopilot.config import StrategyConfig
from tradecopilot.models import DataQuality, IndicatorSnapshot, OHLCVBar, Quote

EASTERN = ZoneInfo("America/New_York")
ZERO = Decimal("0")
ONE_HUNDRED = Decimal("100")


class InsufficientIndicators(ValueError):
    pass


def ema_series(values: Sequence[Decimal], period: int) -> list[Decimal | None]:
    """Return an EMA series aligned to the inputs, with no look-ahead values."""
    if period <= 0:
        raise ValueError("period must be positive")
    output: list[Decimal | None] = [None] * len(values)
    if len(values) < period:
        return output
    seed = sum(values[:period], ZERO) / Decimal(period)
    output[period - 1] = seed
    multiplier = Decimal(2) / Decimal(period + 1)
    current = seed
    for index in range(period, len(values)):
        current = (values[index] - current) * multiplier + current
        output[index] = current
    return output


def ema(values: Sequence[Decimal], period: int) -> Decimal:
    result = ema_series(values, period)
    if not result or result[-1] is None:
        raise InsufficientIndicators(f"EMA({period}) needs at least {period} values")
    return result[-1]


def session_vwap(bars: Sequence[OHLCVBar], session_definition: str) -> Decimal:
    selected = [bar for bar in bars if _in_vwap_session(bar, session_definition)]
    total_volume = sum(bar.volume for bar in selected)
    if not selected or total_volume <= 0:
        raise InsufficientIndicators("VWAP needs positive session volume")
    numerator = sum(((bar.high + bar.low + bar.close) / Decimal(3)) * Decimal(bar.volume) for bar in selected)
    return numerator / Decimal(total_volume)


def _in_vwap_session(bar: OHLCVBar, session_definition: str) -> bool:
    local = bar.provider_timestamp.astimezone(EASTERN).timetz().replace(tzinfo=None)
    if session_definition == "regular_hours":
        return time(9, 30) <= local < time(16, 0)
    if session_definition == "extended_hours":
        return time(4, 0) <= local < time(20, 0)
    raise ValueError(f"unknown VWAP session: {session_definition}")


def atr(bars: Sequence[OHLCVBar], period: int = 14) -> Decimal:
    if len(bars) < period:
        raise InsufficientIndicators(f"ATR({period}) needs at least {period} bars")
    true_ranges: list[Decimal] = []
    for index, bar in enumerate(bars):
        if index == 0:
            true_ranges.append(bar.high - bar.low)
            continue
        previous_close = bars[index - 1].close
        true_ranges.append(
            max(
                bar.high - bar.low,
                abs(bar.high - previous_close),
                abs(bar.low - previous_close),
            )
        )
    current = sum(true_ranges[:period], ZERO) / Decimal(period)
    for value in true_ranges[period:]:
        current = ((current * Decimal(period - 1)) + value) / Decimal(period)
    return current


def rsi(values: Sequence[Decimal], period: int = 14) -> Decimal:
    if len(values) < period + 1:
        raise InsufficientIndicators(f"RSI({period}) needs at least {period + 1} values")
    deltas = [values[index] - values[index - 1] for index in range(1, len(values))]
    gains = [max(delta, ZERO) for delta in deltas]
    losses = [max(-delta, ZERO) for delta in deltas]
    average_gain = sum(gains[:period], ZERO) / Decimal(period)
    average_loss = sum(losses[:period], ZERO) / Decimal(period)
    for gain, loss in zip(gains[period:], losses[period:], strict=True):
        average_gain = ((average_gain * Decimal(period - 1)) + gain) / Decimal(period)
        average_loss = ((average_loss * Decimal(period - 1)) + loss) / Decimal(period)
    if average_loss == 0:
        return ONE_HUNDRED
    relative_strength = average_gain / average_loss
    return ONE_HUNDRED - (ONE_HUNDRED / (Decimal(1) + relative_strength))


def macd(values: Sequence[Decimal], fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[Decimal, Decimal]:
    fast_values = ema_series(values, fast)
    slow_values = ema_series(values, slow)
    macd_values = [
        fast_value - slow_value
        for fast_value, slow_value in zip(fast_values, slow_values, strict=True)
        if fast_value is not None and slow_value is not None
    ]
    if len(macd_values) < signal:
        raise InsufficientIndicators("MACD needs at least slow + signal - 1 values")
    return macd_values[-1], ema(macd_values, signal)


def _structure(bars: Sequence[OHLCVBar]) -> str:
    if len(bars) < 3:
        return "insufficient"
    first, second, third = bars[-3:]
    if second.high > first.high and third.high > second.high and second.low > first.low and third.low > second.low:
        return "higher-high/higher-low uptrend"
    if second.high < first.high and third.high < second.high and second.low < first.low and third.low < second.low:
        return "lower-high/lower-low downtrend"
    return "mixed"


def _nearest_half(price: Decimal) -> Decimal:
    return (price * Decimal(2)).to_integral_value() / Decimal(2)


def _nearest_whole(price: Decimal) -> Decimal:
    return price.to_integral_value()


def _opening_range(bars: Sequence[OHLCVBar], minutes: int) -> tuple[Decimal | None, Decimal | None]:
    regular = []
    for bar in bars:
        local = bar.provider_timestamp.astimezone(EASTERN)
        start = local.replace(hour=9, minute=30, second=0, microsecond=0)
        end = start + timedelta(minutes=minutes)
        if start <= local < end:
            regular.append(bar)
    if not regular:
        return None, None
    return max(bar.high for bar in regular), min(bar.low for bar in regular)


def calculate_indicators(
    quote: Quote,
    bars_1m: Sequence[OHLCVBar],
    bars_5m: Sequence[OHLCVBar],
    resistance_levels: Sequence[Decimal],
    config: StrategyConfig,
) -> IndicatorSnapshot:
    if len(bars_1m) < 20 or len(bars_5m) < 20:
        raise InsufficientIndicators("at least 20 one-minute and five-minute bars are required")
    closes_1m = [bar.close for bar in bars_1m]
    closes_5m = [bar.close for bar in bars_5m]
    current_vwap = session_vwap(bars_1m, config.vwap_session)
    prior_vwap = session_vwap(bars_1m[:-1], config.vwap_session)
    current_ema9 = ema(closes_1m, 9)
    prior_ema9 = ema(closes_1m[:-1], 9)
    atr14 = atr(bars_1m, 14)
    try:
        current_rsi = rsi(closes_1m, 14)
    except InsufficientIndicators:
        current_rsi = None
    current_macd: Decimal | None
    current_macd_signal: Decimal | None
    try:
        current_macd, current_macd_signal = macd(closes_1m)
    except InsufficientIndicators:
        current_macd = current_macd_signal = None

    local_date = quote.provider_timestamp.astimezone(EASTERN).date()
    premarket = [
        bar
        for bar in bars_1m
        if bar.provider_timestamp.astimezone(EASTERN).date() == local_date
        and time(4, 0) <= bar.provider_timestamp.astimezone(EASTERN).timetz().replace(tzinfo=None) < time(9, 30)
    ]
    same_day = [bar for bar in bars_1m if bar.provider_timestamp.astimezone(EASTERN).date() == local_date]
    if not same_day:
        same_day = list(bars_1m)
    opening_high, opening_low = _opening_range(same_day, config.opening_range_minutes)
    support_candidates = [
        level
        for level in (*resistance_levels, current_vwap, current_ema9, _nearest_half(quote.last))
        if level < quote.last
    ]
    resistance_candidates = [level for level in resistance_levels if level > quote.last]
    support = max(support_candidates, default=min(bar.low for bar in same_day))
    resistance = min(
        resistance_candidates,
        default=(quote.last.to_integral_value() + Decimal(1)),
    )
    distance = quote.last - current_vwap
    spread = quote.ask - quote.bid
    if quote.average_daily_volume_50d is None:
        raise InsufficientIndicators("Shibui 50-day average volume is unavailable")
    source_rvol = Decimal(quote.total_volume) / Decimal(quote.average_daily_volume_50d)
    time_adjusted = (
        Decimal(quote.total_volume) / Decimal(quote.average_cumulative_volume_same_time)
        if quote.average_cumulative_volume_same_time
        else None
    )
    receipt = max(quote.receipt_timestamp, bars_1m[-1].receipt_timestamp)

    return IndicatorSnapshot(
        provider_timestamp=receipt,
        receipt_timestamp=receipt,
        age_seconds=0,
        source="deterministic_feature_engine",
        quality=DataQuality.GOOD,
        symbol=quote.symbol,
        session_definition=config.vwap_session,
        vwap=current_vwap,
        vwap_slope=current_vwap - prior_vwap,
        ema9_1m=current_ema9,
        ema9_slope_1m=current_ema9 - prior_ema9,
        ema20_1m=ema(closes_1m, 20),
        ema9_5m=ema(closes_5m, 9),
        ema20_5m=ema(closes_5m, 20),
        atr14_1m=atr14,
        rsi14_1m=current_rsi,
        macd_1m=current_macd,
        macd_signal_1m=current_macd_signal,
        premarket_high=max((bar.high for bar in premarket), default=None),
        premarket_low=min((bar.low for bar in premarket), default=None),
        high_of_day=max(bar.high for bar in same_day),
        low_of_day=min(bar.low for bar in same_day),
        opening_range_high=opening_high,
        opening_range_low=opening_low,
        nearest_half_dollar=_nearest_half(quote.last),
        nearest_whole_dollar=_nearest_whole(quote.last),
        structural_support=support,
        structural_resistance=resistance,
        distance_from_vwap=distance,
        distance_from_vwap_percent=(distance / current_vwap) * ONE_HUNDRED,
        distance_from_vwap_atr=distance / atr14 if atr14 else ZERO,
        spread_dollars=spread,
        spread_percent=(spread / quote.last) * ONE_HUNDRED,
        source_style_rvol=source_rvol,
        time_adjusted_rvol=time_adjusted,
        one_minute_structure=_structure(bars_1m),
        five_minute_structure=_structure(bars_5m),
    )
