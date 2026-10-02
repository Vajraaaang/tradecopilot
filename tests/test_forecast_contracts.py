from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError


def test_config_identity_tracks_target_and_rejects_ambiguous_symbols():
    from tradecopilot.forecast.contracts import ForecastConfig

    config = ForecastConfig()
    assert config.horizon_minutes == 15 and len(config.symbols) == 5
    assert config.config_id != ForecastConfig(flat_threshold_bps=20).config_id
    with pytest.raises(ValidationError):
        ForecastConfig(symbols=("AAPL", "AAPL"))


def test_forecast_probabilities_require_the_exact_outcome_contract():
    from tradecopilot.forecast.contracts import ForecastPrediction

    values = dict(
        example_id="x",
        dataset_id="dataset",
        model_id="baseline",
        generated_at=datetime.now(UTC),
        status="ok",
        execution="local",
        probabilities={"DOWN": 0.2, "FLAT": 0.3, "UP": 0.5},
    )
    assert ForecastPrediction(**values).probabilities["UP"] == 0.5
    for probabilities in (
        {"BUY": 0.5, "HOLD": 0.5},
        {"DOWN": 0.2, "FLAT": 0.3, "UP": 0.8},
        {"DOWN": 0.2, "FLAT": 0.3, "UP": float("nan")},
    ):
        with pytest.raises(ValidationError):
            ForecastPrediction(**{**values, "probabilities": probabilities})


def test_example_validates_causality_and_label_maturity():
    from tradecopilot.forecast.contracts import FEATURE_NAMES, ForecastExample

    now = datetime.now(UTC)
    values = dict(
        config_id="test-config",
        symbol="AAPL",
        as_of=now,
        target_time=now + timedelta(minutes=15),
        session_date=now.date(),
        anchor_price="100",
        features={key: 0.0 for key in FEATURE_NAMES},
        observation_ids=("obs",),
        provenance="synthetic",
    )
    example = ForecastExample(**values)
    assert example.label is None and example.example_id == ForecastExample(**values).example_id
    with pytest.raises(ValidationError):
        ForecastExample(**{**values, "target_time": now - timedelta(seconds=1)})
    with pytest.raises(ValidationError):
        ForecastExample(**{**values, "label": "UP", "target_price": "101", "label_observed_at": now})
