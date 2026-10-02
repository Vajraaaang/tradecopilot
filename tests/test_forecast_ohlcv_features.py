from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from tradecopilot.forecast.bars import HistoricalBar
from tradecopilot.forecast.contracts import FEATURE_NAMES, ForecastConfig, ForecastExample


def bars():
    result = []
    for day in (14, 15):
        start = datetime(2026, 9, day, 13, 30, tzinfo=UTC)
        for index in range(90):
            close = Decimal("100") + Decimal(index) / 100
            result.append(
                HistoricalBar(
                    symbol="AAPL",
                    start_time=start + timedelta(minutes=index),
                    end_time=start + timedelta(minutes=index + 1),
                    available_at=start + timedelta(minutes=index + 1),
                    opening=close - Decimal("0.01"),
                    high=close + Decimal("0.2"),
                    low=close - Decimal("0.2"),
                    close=close,
                    volume=Decimal(100 + index),
                )
            )
    return result


def example(rows, offset=70):
    anchor = rows[90 + offset - 1]
    return ForecastExample(
        config_id=ForecastConfig(symbols=("AAPL",)).config_id,
        symbol="AAPL",
        as_of=anchor.end_time,
        target_time=anchor.end_time + timedelta(minutes=15),
        session_date=date(2026, 9, 15),
        anchor_price=anchor.close,
        features=dict.fromkeys(FEATURE_NAMES, 0.0),
        observation_ids=(anchor.bar_id,),
        provenance="historical",
    )


def test_v2_uses_real_high_low_volume_and_longer_returns():
    from tradecopilot.forecast.ohlcv_features import OhlcvFeatureBuilder

    rows = bars()
    row = example(rows)
    features = OhlcvFeatureBuilder(rows).build(row)
    assert features.feature_version == "causal-ohlcv-v2"
    assert features.values["high_low_range_5m_bps"] > 30
    assert features.values["return_60m_bps"] > 50
    assert features.values["log_volume_1m"] > 0
    assert features.values["session_minute"] == 70
    assert features.values["missing_window_60m"] == 0
    assert features.values["relative_volume_same_minute"] == pytest.approx(1)


def test_future_bars_and_late_receipts_never_change_features():
    from tradecopilot.forecast.ohlcv_features import OhlcvFeatureBuilder

    rows = bars()
    row = example(rows)
    original = OhlcvFeatureBuilder(rows).build(row)
    changed = [
        item.model_copy(
            update={
                "opening": Decimal("9999"),
                "high": Decimal("10000"),
                "low": Decimal("9998"),
                "close": Decimal("9999"),
                "volume": Decimal("9999"),
            }
        )
        if item.end_time > row.as_of
        else item
        for item in rows
    ]
    assert OhlcvFeatureBuilder(changed).build(row) == original
    late = rows[90].model_copy(update={"available_at": row.as_of + timedelta(minutes=1)})
    late_rows = [late if item == rows[90] else item for item in rows]
    poisoned = late.model_copy(
        update={"opening": Decimal("9999"), "high": Decimal("10000"), "low": Decimal("9998"), "close": Decimal("9999")}
    )
    changed_late = [poisoned if item == late else item for item in late_rows]
    assert OhlcvFeatureBuilder(late_rows).build(row) == OhlcvFeatureBuilder(changed_late).build(row)


def test_short_history_is_explicitly_missing_not_forward_filled():
    from tradecopilot.forecast.ohlcv_features import OhlcvFeatureBuilder

    rows = bars()
    row = example(rows, offset=10)
    values = OhlcvFeatureBuilder(rows).build(row).values
    assert values["return_60m_bps"] is None and values["missing_window_60m"] == 1
    assert values["return_5m_bps"] is not None


def test_current_volume_is_not_in_its_own_trailing_reference():
    from tradecopilot.forecast.ohlcv_features import OhlcvFeatureBuilder

    rows = bars()
    row = example(rows)
    original = OhlcvFeatureBuilder(rows).build(row).values
    changed = [
        item.model_copy(update={"volume": item.volume * 10})
        if item.end_time == row.as_of and item.start_time.date() == row.session_date
        else item
        for item in rows
    ]
    revised = OhlcvFeatureBuilder(changed).build(row).values
    assert revised["relative_volume_prior_20m"] == pytest.approx(original["relative_volume_prior_20m"] * 10)
