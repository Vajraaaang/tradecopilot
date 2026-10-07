"""Synthetic delayed opening fills bounded by decision-known volume."""

from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from .contracts import Fill, LedgerState, Order, SimConfig


def floor_quantity(value: Decimal, quantum: Decimal) -> Decimal:
    return (value / quantum).to_integral_value(rounding=ROUND_FLOOR) * quantum


def execute(
    order: Order,
    state: LedgerState,
    config: SimConfig,
    *,
    opening_price: Decimal,
    known_volume: Decimal,
    execution_volume: Decimal,
    available_at: int,
) -> tuple[Fill | None, str | None]:
    capacity = floor_quantity(known_volume * config.capacity_fraction, config.quantity_quantum)
    if execution_volume <= 0 or capacity <= 0:
        return None, "no_liquidity"
    sign = Decimal(1) if order.side == "buy" else Decimal(-1)
    price = opening_price * (1 + sign * config.cost_bps / 10000)
    fee = config.fee_per_fill.quantize(Decimal(".01"), rounding=ROUND_CEILING)
    if order.side == "buy":
        dollars = min(
            order.buy_notional,
            max(Decimal(0), state.cash - fee),
            max(Decimal(0), config.notional_cap - state.shares * opening_price),
        )
        requested = order.buy_notional / price
        quantity = floor_quantity(min(dollars / price, capacity), config.quantity_quantum)
    else:
        requested = order.sell_shares
        quantity = floor_quantity(min(requested, state.shares, capacity), config.quantity_quantum)
        if state.cash + quantity * price < fee:
            return None, "fee_unaffordable"
    if quantity <= 0:
        return None, "cash_or_cap_limit"
    return Fill(order.side, quantity, price, opening_price, fee, order.execution_at, available_at), (
        "partial_fill" if quantity < requested else None
    )
