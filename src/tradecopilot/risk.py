from __future__ import annotations

from decimal import ROUND_FLOOR, Decimal

from tradecopilot.config import StrategyConfig


def estimate_slippage(spread: Decimal, top_of_book_size: int, config: StrategyConfig) -> Decimal:
    """Conservative spread, liquidity, and configured-buffer estimate."""
    if spread < 0:
        raise ValueError("spread cannot be negative")
    liquidity_penalty = spread if top_of_book_size < 1_000 else spread / Decimal(2)
    return (spread / Decimal(2)) + liquidity_penalty + config.slippage_safety_buffer


def risk_per_share(planned_entry: Decimal, structural_stop: Decimal, estimated_slippage: Decimal) -> Decimal:
    risk = planned_entry - structural_stop + estimated_slippage
    if risk <= 0:
        raise ValueError("risk per share must be positive")
    return risk


def maximum_shares(maximum_dollar_risk: Decimal, per_share_risk: Decimal) -> int:
    if maximum_dollar_risk < 0:
        raise ValueError("maximum dollar risk cannot be negative")
    if per_share_risk <= 0:
        raise ValueError("per-share risk must be positive")
    return int((maximum_dollar_risk / per_share_risk).to_integral_value(rounding=ROUND_FLOOR))


def validate_stop_not_widened(original_stop: Decimal, proposed_stop: Decimal) -> None:
    if proposed_stop < original_stop:
        raise ValueError("widening a long-position stop is prohibited")


def validate_no_averaging_down(
    current_average_entry: Decimal, proposed_entry: Decimal, current_quantity: Decimal
) -> None:
    if current_quantity > 0 and proposed_entry < current_average_entry:
        raise ValueError("averaging down is prohibited")
