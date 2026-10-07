"""Exact internal cash arithmetic; display rounding is a reporting concern."""

from dataclasses import replace
from decimal import Decimal

from .contracts import Fill, LedgerState


def apply_fill(state: LedgerState, fill: Fill) -> LedgerState:
    gross = fill.quantity * fill.price
    if fill.side == "buy":
        result = replace(
            state,
            cash=state.cash - gross - fill.fee,
            shares=state.shares + fill.quantity,
            cost_basis=state.cost_basis + gross,
        )
    else:
        removed = state.cost_basis if fill.quantity == state.shares else state.cost_basis * fill.quantity / state.shares
        result = replace(
            state,
            cash=state.cash + gross - fill.fee,
            shares=state.shares - fill.quantity,
            cost_basis=state.cost_basis - removed,
            realized_pnl=state.realized_pnl + gross - removed,
        )
    if result.cash < 0 or result.shares < 0:
        raise ValueError("negative cash or shares")
    return replace(
        result,
        fees=state.fees + fill.fee,
        execution_drag=state.execution_drag + abs(fill.price - fill.reference_price) * fill.quantity,
    )


def mark_to_market(state: LedgerState, price: Decimal) -> LedgerState:
    equity = state.equity(price)
    peak = max(state.peak_equity, equity)
    drawdown = (peak - equity) / peak if peak else Decimal(0)
    return replace(state, peak_equity=peak, max_drawdown=max(state.max_drawdown, drawdown))
