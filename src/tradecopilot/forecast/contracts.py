from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from tradecopilot.models import FrozenModel, PriceSnapshot

LABELS = ("DOWN", "FLAT", "UP")
ForecastLabel = Literal["DOWN", "FLAT", "UP"]
Provenance = Literal["synthetic", "market"]
FEATURE_VERSION = "causal-price-v1"
LABEL_VERSION = "forward-return-v1"
FEATURE_NAMES = (
    "return_1m_bps",
    "return_5m_bps",
    "range_5m_bps",
    "volatility_5m_bps",
    "change_from_previous_close_bps",
    "source_age_seconds",
    "history_points",
)


def content_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC)


class ForecastConfig(FrozenModel):
    schema_version: Literal["forecast-config-v1"] = "forecast-config-v1"
    symbols: tuple[str, ...] = ("AAPL", "MSFT", "AMZN", "GOOGL", "NVDA")
    horizon_minutes: int = Field(default=15, ge=1, le=120)
    lookback_minutes: int = Field(default=5, ge=5, le=30)
    anchor_seconds: int = Field(default=60, ge=60, le=900)
    flat_threshold_bps: float = Field(default=10, ge=0, le=1000, allow_inf_nan=False)
    max_source_age_seconds: float = Field(default=30, ge=0, le=300, allow_inf_nan=False)
    max_outcome_delay_seconds: int = Field(default=60, ge=0, le=300)
    min_history_points: int = Field(default=6, ge=2)
    max_history_gap_seconds: int = Field(default=60, ge=1, le=300)
    abstention_threshold: float = Field(default=0.6, ge=0, le=1, allow_inf_nan=False)
    seed: int = 42

    @field_validator("symbols")
    @classmethod
    def valid_symbols(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        import re

        if not values or len(values) > 20 or len(set(values)) != len(values):
            raise ValueError("provide 1-20 unique symbols")
        if any(not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", value) for value in values):
            raise ValueError("symbols must be normalized US stock identifiers")
        return values

    @property
    def config_id(self) -> str:
        return content_hash(self.model_dump(mode="json"))


class Observation(PriceSnapshot):
    provenance: Provenance

    @field_validator("provider_timestamp", "receipt_timestamp")
    @classmethod
    def normalize_utc(cls, value: datetime) -> datetime:
        return utc(value)

    @property
    def observation_id(self) -> str:
        return content_hash(self.model_dump(mode="json"))


class ForecastExample(FrozenModel):
    config_id: str = Field(min_length=1)
    symbol: str
    as_of: datetime
    target_time: datetime
    session_date: date
    anchor_price: Decimal = Field(gt=0)
    features: dict[str, float]
    observation_ids: tuple[str, ...]
    provenance: Provenance
    label: ForecastLabel | None = None
    target_price: Decimal | None = Field(default=None, gt=0)
    label_observed_at: datetime | None = None
    target_return_bps: float | None = Field(default=None, allow_inf_nan=False)
    exclusion_reason: str | None = None

    @field_validator("as_of", "target_time", "label_observed_at")
    @classmethod
    def normalize_utc(cls, value: datetime | None) -> datetime | None:
        return utc(value) if value is not None else None

    @model_validator(mode="after")
    def valid_example(self) -> Self:
        if self.target_time <= self.as_of:
            raise ValueError("forecast target must follow its as-of time")
        if set(self.features) != set(FEATURE_NAMES) or not all(math.isfinite(v) for v in self.features.values()):
            raise ValueError("feature schema is invalid")
        if not self.observation_ids:
            raise ValueError("input provenance is required")
        label_fields = (self.label, self.target_price, self.label_observed_at, self.target_return_bps)
        if any(v is not None for v in label_fields) and not all(v is not None for v in label_fields):
            raise ValueError("an outcome must include label, price, timestamp, and return")
        if self.label_observed_at is not None and self.label_observed_at < self.target_time:
            raise ValueError("outcome cannot be observed before the target")
        return self

    @property
    def example_id(self) -> str:
        # Label arrival never changes the identity of the prediction input.
        return content_hash(
            self.model_dump(
                mode="json",
                exclude={
                    "label",
                    "target_price",
                    "label_observed_at",
                    "target_return_bps",
                    "exclusion_reason",
                },
            )
        )


class DatasetManifest(FrozenModel):
    dataset_id: str
    observations_hash: str
    examples_hash: str
    config: ForecastConfig
    feature_version: str = FEATURE_VERSION
    label_version: str = LABEL_VERSION
    calendar: str = "XNYS"
    calendar_version: str
    provenance: Provenance
    observation_count: int = Field(ge=0)
    example_count: int = Field(ge=0)
    labeled_count: int = Field(ge=0)
    sessions: tuple[date, ...]
    exclusions: dict[str, int] = Field(default_factory=dict)


class ForecastPrediction(FrozenModel):
    example_id: str
    dataset_id: str
    model_id: str
    generated_at: datetime
    status: Literal["ok", "abstained", "error"]
    execution: Literal["local", "fixture", "live_api"]
    probabilities: dict[str, float] | None = None
    reason: str | None = None
    model_version: str | None = None
    prompt_version: str | None = None
    calibration_version: str | None = None
    model_confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    input_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    latency_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    request_id: str | None = None
    cached: bool = False

    @field_validator("generated_at")
    @classmethod
    def normalize_utc(cls, value: datetime) -> datetime:
        return utc(value)

    @model_validator(mode="after")
    def valid_prediction(self) -> Self:
        if self.status == "error":
            if self.probabilities is not None or not self.reason:
                raise ValueError("failed inference requires a reason and no probability claim")
            return self
        values = self.probabilities
        if (
            values is None
            or set(values) != set(LABELS)
            or not all(math.isfinite(v) and 0 <= v <= 1 for v in values.values())
            or not math.isclose(sum(values.values()), 1, abs_tol=1e-6)
        ):
            raise ValueError("forecast probabilities must cover DOWN/FLAT/UP and sum to one")
        return self

    @property
    def prediction_id(self) -> str:
        return content_hash(self.model_dump(mode="json"))
