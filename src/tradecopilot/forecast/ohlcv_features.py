"""A separate causal OHLCV representation; the published price-only v1 remains unchanged."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from itertools import pairwise
from statistics import mean, pstdev
from typing import Literal, Self

from pydantic import field_validator, model_validator

from tradecopilot.forecast.bars import HistoricalBar
from tradecopilot.forecast.contracts import ForecastExample, content_hash, utc
from tradecopilot.forecast.sessions import session_bounds, session_for
from tradecopilot.models import FrozenModel

OHLCV_FEATURE_VERSION: Literal["causal-ohlcv-v2"] = "causal-ohlcv-v2"
WINDOWS = (1, 2, 3, 5, 10, 15, 30, 60)
OHLCV_FEATURE_NAMES = (
    *(
        name
        for window in WINDOWS
        for name in (
            f"return_{window}m_bps",
            f"high_low_range_{window}m_bps",
            f"log_volume_sum_{window}m",
            f"realized_volatility_{window}m_bps",
            f"missing_window_{window}m",
        )
    ),
    "body_1m_bps",
    "close_location_1m",
    "true_range_1m_bps",
    "log_volume_1m",
    "relative_volume_prior_20m",
    "relative_volume_same_minute",
    "same_minute_reference_count",
    "session_minute",
    "minutes_to_close",
    "day_of_week",
    "opening_gap_proxy_bps",
    "change_from_previous_close_proxy_bps",
    "distance_previous_high_bps",
    "distance_previous_low_bps",
    "distance_session_vwap_proxy_bps",
)


class OhlcvFeatureRecord(FrozenModel):
    feature_version: Literal["causal-ohlcv-v2"] = OHLCV_FEATURE_VERSION
    base_example_id: str
    config_id: str
    symbol: str
    as_of: datetime
    values: dict[str, float | None]
    input_bars_hash: str

    @field_validator("as_of")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        return utc(value)

    @model_validator(mode="after")
    def valid_feature_schema(self) -> Self:
        if set(self.values) != set(OHLCV_FEATURE_NAMES) or any(
            value is not None and not math.isfinite(value) for value in self.values.values()
        ):
            raise ValueError("invalid OHLCV feature schema")
        return self

    @property
    def feature_id(self) -> str:
        return content_hash(self.model_dump(mode="json"))


def _bps(last: float, first: float) -> float:
    return 10_000 * (last / first - 1)


@dataclass(frozen=True)
class _MarketAnchor:
    symbol: str
    as_of: datetime
    session_date: date
    anchor_price: Decimal
    config_id: str
    example_id: str
    provenance: Literal["historical"] = "historical"


class OhlcvFeatureBuilder:
    def __init__(self, bars: Sequence[HistoricalBar]) -> None:
        self._sessions: dict[tuple[str, date], list[HistoricalBar]] = defaultdict(list)
        self._minutes: dict[tuple[str, float], list[HistoricalBar]] = defaultdict(list)
        self._ids: dict[tuple[str, datetime], str] = {}
        for item in bars:
            bar = HistoricalBar.model_validate(item.model_dump(mode="json"))
            key = (bar.symbol, bar.end_time)
            if key in self._ids:
                raise ValueError("duplicate OHLCV bar")
            day = session_for(bar.end_time)
            bounds = session_bounds(day) if day is not None else None
            if bounds is None or bar.start_time < bounds[0] or bar.end_time > bounds[1]:
                raise ValueError("bar is outside a regular session")
            self._ids[key] = bar.bar_id
            assert day is not None
            self._sessions[(bar.symbol, day)].append(bar)
            minute = (bar.end_time - bounds[0]).total_seconds() / 60
            self._minutes[(bar.symbol, minute)].append(bar)
        for rows in [*self._sessions.values(), *self._minutes.values()]:
            rows.sort(key=lambda bar: (bar.end_time, bar.available_at))

    def build(self, example: ForecastExample) -> OhlcvFeatureRecord:
        return self._build(example)

    def build_as_of(self, symbol: str, as_of: datetime, config_id: str) -> OhlcvFeatureRecord:
        """Build market features without creating or passing an outcome-bearing example."""
        as_of = utc(as_of)
        day = session_for(as_of)
        if day is None or not config_id:
            raise ValueError("regular-session as-of anchor and configuration are required")
        matches = [bar for bar in self._sessions.get((symbol, day), ())
                   if bar.end_time == as_of and bar.available_at <= as_of]
        if len(matches) != 1:
            raise ValueError("OHLCV anchor is missing or late")
        anchor = matches[0]
        identity = content_hash({"schema": "label-free-market-anchor-v1", "symbol": symbol,
                                 "as_of": as_of.isoformat(), "config_id": config_id, "bar_id": anchor.bar_id})
        return self._build(_MarketAnchor(symbol, as_of, day, anchor.close, config_id, identity))

    def _build(self, example: ForecastExample | _MarketAnchor) -> OhlcvFeatureRecord:
        day = session_for(example.as_of)
        if day != example.session_date or example.provenance != "historical":
            raise ValueError("OHLCV features require a historical regular-session example")
        bounds = session_bounds(example.session_date)
        assert bounds is not None
        current = [
            bar
            for bar in self._sessions.get((example.symbol, example.session_date), ())
            if bar.end_time <= example.as_of and bar.available_at <= example.as_of
        ]
        if not current or current[-1].end_time != example.as_of or current[-1].close != example.anchor_price:
            raise ValueError("OHLCV anchor is missing, late or differs from the frozen input")
        anchor = current[-1]
        price = float(anchor.close)
        by_end = {bar.end_time: bar for bar in current}
        values: dict[str, float | None] = {}
        for window in WINDOWS:
            endpoint = example.as_of - timedelta(minutes=window)
            predecessor = by_end.get(endpoint)
            rows = [by_end.get(endpoint + timedelta(minutes=index)) for index in range(1, window + 1)]
            complete = predecessor is not None and all(bar is not None for bar in rows)
            values[f"missing_window_{window}m"] = float(not complete)
            if not complete:
                for name in ("return", "high_low_range", "log_volume_sum", "realized_volatility"):
                    suffix = "" if name == "log_volume_sum" else "_bps"
                    values[f"{name}_{window}m{suffix}"] = None
                continue
            assert predecessor is not None
            available = [bar for bar in rows if bar is not None]
            closes = [float(predecessor.close), *[float(bar.close) for bar in available]]
            returns = [_bps(right, left) for left, right in pairwise(closes)]
            values[f"return_{window}m_bps"] = _bps(price, closes[0])
            values[f"high_low_range_{window}m_bps"] = (
                10_000 * float(max(bar.high for bar in available) - min(bar.low for bar in available)) / price
            )
            values[f"log_volume_sum_{window}m"] = math.log1p(sum(float(bar.volume) for bar in available))
            values[f"realized_volatility_{window}m_bps"] = pstdev(returns)
        previous_minute = by_end.get(example.as_of - timedelta(minutes=1))
        prior20 = [bar for bar in current if example.as_of - timedelta(minutes=20) <= bar.end_time < example.as_of]
        previous_day = example.session_date - timedelta(days=1)
        while session_bounds(previous_day) is None:
            previous_day -= timedelta(days=1)
        previous = [
            bar for bar in self._sessions.get((example.symbol, previous_day), ()) if bar.available_at <= example.as_of
        ]
        if not previous:
            raise ValueError("preceding session OHLCV reference unavailable")
        previous_close = float(previous[-1].close)
        minute = (example.as_of - bounds[0]).total_seconds() / 60
        same_minute = [
            bar
            for bar in self._minutes.get((example.symbol, minute), ())
            if bar.end_time.date() < example.session_date and bar.available_at <= example.as_of
        ][-20:]
        range_one = float(anchor.high - anchor.low)
        total_volume = sum(float(bar.volume) for bar in current)
        vwap_numerator = sum(float((bar.high + bar.low + bar.close) / 3) * float(bar.volume) for bar in current)
        vwap_proxy: float | None = vwap_numerator / total_volume if total_volume else None
        volume = float(anchor.volume)
        prior_volume = mean(float(bar.volume) for bar in prior20) if len(prior20) >= 5 else 0
        same_volume = mean(float(bar.volume) for bar in same_minute) if same_minute else 0
        true_range = (
            max(
                range_one,
                abs(float(anchor.high) - float(previous_minute.close)),
                abs(float(anchor.low) - float(previous_minute.close)),
            )
            if previous_minute
            else None
        )
        values |= {
            "body_1m_bps": _bps(price, float(anchor.opening)),
            "close_location_1m": float(anchor.close - anchor.low) / range_one if range_one else 0.5,
            "true_range_1m_bps": 10_000 * true_range / price if true_range is not None else None,
            "log_volume_1m": math.log1p(volume),
            "relative_volume_prior_20m": volume / prior_volume if prior_volume else None,
            "relative_volume_same_minute": volume / same_volume if same_volume else None,
            "same_minute_reference_count": float(len(same_minute)),
            "session_minute": minute,
            "minutes_to_close": (bounds[1] - example.as_of).total_seconds() / 60,
            "day_of_week": float(example.session_date.weekday()),
            "opening_gap_proxy_bps": _bps(float(current[0].opening), previous_close),
            "change_from_previous_close_proxy_bps": _bps(price, previous_close),
            "distance_previous_high_bps": _bps(price, float(max(bar.high for bar in previous))),
            "distance_previous_low_bps": _bps(price, float(min(bar.low for bar in previous))),
            "distance_session_vwap_proxy_bps": _bps(price, vwap_proxy) if vwap_proxy is not None else None,
        }
        dependencies = {
            *[(bar.symbol, bar.end_time) for bar in current],
            *[(bar.symbol, bar.end_time) for bar in previous],
            *[(bar.symbol, bar.end_time) for bar in same_minute],
        }
        return OhlcvFeatureRecord(
            base_example_id=example.example_id,
            config_id=example.config_id,
            symbol=example.symbol,
            as_of=example.as_of,
            values=values,
            input_bars_hash=content_hash(sorted(self._ids[key] for key in dependencies)),
        )
