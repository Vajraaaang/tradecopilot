from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tradecopilot.config import StrategyConfig
from tradecopilot.features import retracement_fraction
from tradecopilot.risk import (
    estimate_slippage,
    maximum_shares,
    risk_per_share,
    validate_no_averaging_down,
    validate_stop_not_widened,
)


@given(
    maximum_risk=st.decimals(min_value="0.01", max_value="10000", places=2),
    per_share=st.decimals(min_value="0.0001", max_value="100", places=4),
)
def test_position_sizing_never_exceeds_configured_dollar_risk(maximum_risk: Decimal, per_share: Decimal) -> None:
    shares = maximum_shares(maximum_risk, per_share)
    assert Decimal(shares) * per_share <= maximum_risk


def test_slippage_is_included_in_risk_per_share() -> None:
    slippage = estimate_slippage(Decimal("0.02"), 500, StrategyConfig())
    risk = risk_per_share(Decimal("9.15"), Decimal("8.89"), slippage)
    assert slippage > 0
    assert risk == Decimal("0.26") + slippage


def test_stop_cannot_be_widened_after_entry() -> None:
    with pytest.raises(ValueError, match="widening"):
        validate_stop_not_widened(Decimal("8.90"), Decimal("8.80"))
    validate_stop_not_widened(Decimal("8.90"), Decimal("9.00"))


def test_averaging_down_is_prohibited() -> None:
    with pytest.raises(ValueError, match="averaging down"):
        validate_no_averaging_down(Decimal("9.15"), Decimal("9.00"), Decimal("80"))


@given(
    impulse_low=st.decimals(min_value="0.01", max_value="100", places=2),
    impulse_range=st.decimals(min_value="0.01", max_value="100", places=2),
    fraction=st.decimals(min_value="0", max_value="1", places=4),
)
def test_retracement_fraction_preserves_bounded_fraction(
    impulse_low: Decimal, impulse_range: Decimal, fraction: Decimal
) -> None:
    impulse_high = impulse_low + impulse_range
    pullback_low = impulse_high - impulse_range * fraction
    assert retracement_fraction(impulse_low, impulse_high, pullback_low) == fraction
