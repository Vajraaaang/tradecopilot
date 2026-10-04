"""Immutable, label-free simulator inputs and exact accounting contracts."""

import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from hashlib import sha256
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SimConfig:
    initial_cash: Decimal = Decimal("10000")
    notional_cap: Decimal = Decimal("1000")
    daily_loss: Decimal = Decimal("100")
    cost_bps: Decimal = Decimal("2")
    fee_per_fill: Decimal = Decimal("0")
    capacity_fraction: Decimal = Decimal("0.01")
    quantity_quantum: Decimal = Decimal("0.000001")
    warmup_minutes: int = 60
    latency_minutes: int = 1
    forced_close_buffer_minutes: int = 5

    def __post_init__(self) -> None:
        for name in (
            "initial_cash",
            "notional_cap",
            "daily_loss",
            "cost_bps",
            "fee_per_fill",
            "capacity_fraction",
            "quantity_quantum",
        ):
            value = Decimal(str(getattr(self, name)))
            object.__setattr__(self, name, value)
            if not value.is_finite() or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if min(self.initial_cash, self.notional_cap, self.daily_loss, self.quantity_quantum) <= 0:
            raise ValueError("capital, limits and quantity quantum must be positive")
        if self.notional_cap > self.initial_cash or self.daily_loss > self.initial_cash:
            raise ValueError("notional cap and daily loss must not exceed initial cash")
        if self.capacity_fraction > 1 or self.cost_bps >= 10000:
            raise ValueError("invalid capacity or cost")
        for name in ("warmup_minutes", "latency_minutes", "forced_close_buffer_minutes"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"invalid {name}")
        if self.latency_minutes < 1 or self.forced_close_buffer_minutes < self.latency_minutes + 1:
            raise ValueError("latency must include one minute; close buffer must allow execution")

    @property
    def content_hash(self) -> str:
        return sha256(json.dumps({k: str(v) for k, v in vars(self).items()}, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class EpisodeData:
    symbol: str
    session_date: date
    session_open: int
    session_close: int
    starts: NDArray[np.int64]
    ends: NDArray[np.int64]
    available_at: NDArray[np.int64]
    opens: NDArray[np.float64]
    closes: NDArray[np.float64]
    volumes: NDArray[np.float64]
    features: NDArray[np.float32]
    feature_names: tuple[str, ...]
    data_id: str
    episode_id: str = ""

    def __post_init__(self) -> None:
        n = len(self.starts)
        if not self.symbol or not self.data_id or self.session_open >= self.session_close or n == 0:
            raise ValueError("invalid episode metadata")
        for name in ("starts", "ends", "available_at", "opens", "closes", "volumes", "features"):
            dtype = (
                np.int64
                if name in ("starts", "ends", "available_at")
                else (np.float32 if name == "features" else np.float64)
            )
            original = np.asarray(getattr(self, name))
            if name in ("starts", "ends", "available_at") and not np.issubdtype(original.dtype, np.integer):
                raise ValueError(f"{name} must use integer timestamps")
            arr = np.array(original, dtype=dtype, copy=True)
            if (name != "features" and arr.shape != (n,)) or not np.isfinite(arr).all():
                raise ValueError(f"invalid {name}")
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)
        if self.features.shape != (n, len(self.feature_names)) or (
            len(set(self.feature_names)) != len(self.feature_names)
        ):
            raise ValueError("invalid feature shape or names")
        if (np.diff(self.starts) <= 0).any() or (self.ends != self.starts + 60).any():
            raise ValueError("bars must be ordered one-minute intervals")
        if (
            (self.available_at < self.ends).any()
            or (self.starts < self.session_open).any()
            or (self.ends > self.session_close).any()
        ):
            raise ValueError("invalid bar availability or session range")
        if (self.opens <= 0).any() or (self.closes <= 0).any() or (self.volumes < 0).any():
            raise ValueError("invalid prices or volume")
        if not self.episode_id:
            h = sha256(
                f"{self.symbol}|{self.session_date}|{self.data_id}|{self.session_open}|{self.session_close}".encode()
            )
            for name in ("starts", "ends", "available_at", "opens", "closes", "volumes", "features"):
                h.update(getattr(self, name).tobytes())
            h.update(repr(self.feature_names).encode())
            object.__setattr__(self, "episode_id", h.hexdigest())


@dataclass(frozen=True)
class LedgerState:
    cash: Decimal
    shares: Decimal = Decimal(0)
    cost_basis: Decimal = Decimal(0)
    realized_pnl: Decimal = Decimal(0)
    fees: Decimal = Decimal(0)
    execution_drag: Decimal = Decimal(0)
    peak_equity: Decimal = Decimal(0)
    max_drawdown: Decimal = Decimal(0)
    last_committed_target_level: int = 0
    risk_locked: bool = False

    def __post_init__(self) -> None:
        for name in (
            "cash",
            "shares",
            "cost_basis",
            "realized_pnl",
            "fees",
            "execution_drag",
            "peak_equity",
            "max_drawdown",
        ):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite():
                raise ValueError(f"{name} must be a finite Decimal")
            if name != "realized_pnl" and value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.max_drawdown > 1 or self.last_committed_target_level not in (0, 1, 2):
            raise ValueError("invalid drawdown or target level")

    def equity(self, price: Decimal) -> Decimal:
        if not price.is_finite() or price <= 0:
            raise ValueError("mark price must be finite and positive")
        return self.cash + self.shares * price


@dataclass(frozen=True)
class Order:
    side: Literal["buy", "sell"]
    decision_at: int
    execution_at: int
    buy_notional: Decimal = Decimal(0)
    sell_shares: Decimal = Decimal(0)

    def __post_init__(self) -> None:
        if self.side not in ("buy", "sell"):
            raise ValueError("invalid order side")
        if any(not value.is_finite() or value < 0 for value in (self.buy_notional, self.sell_shares)):
            raise ValueError("order amounts must be finite and nonnegative")
        if self.execution_at <= self.decision_at:
            raise ValueError("execution must follow decision")
        if self.side == "buy" and (self.buy_notional <= 0 or self.sell_shares != 0):
            raise ValueError("buy commits positive notional only")
        if self.side == "sell" and (self.sell_shares <= 0 or self.buy_notional != 0):
            raise ValueError("sell commits positive shares only")


@dataclass(frozen=True)
class Fill:
    side: Literal["buy", "sell"]
    quantity: Decimal
    price: Decimal
    reference_price: Decimal
    fee: Decimal
    timestamp: int
    available_at: int
