from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from tradecopilot.forecast.contracts import FEATURE_NAMES, ForecastConfig, ForecastExample


def examples(count=90):
    config = ForecastConfig()
    start = datetime(2026, 9, 14, 14, 0, tzinfo=UTC)
    result = []
    for index in range(count):
        direction = index % 3 - 1
        values = {name: 0.0 for name in FEATURE_NAMES}
        values.update(
            return_1m_bps=direction * 15 + index / 1000,
            return_5m_bps=direction * 40,
            range_5m_bps=50,
            volatility_5m_bps=3,
            change_from_previous_close_bps=direction * 100,
            history_points=20,
        )
        as_of = start + timedelta(minutes=index)
        result.append(
            ForecastExample(
                config_id=config.config_id,
                symbol="AAPL",
                as_of=as_of,
                target_time=as_of + timedelta(minutes=15),
                session_date=date(2026, 9, 14),
                anchor_price="100",
                features=values,
                observation_ids=(str(index),),
                provenance="synthetic",
                label=("DOWN", "FLAT", "UP")[index % 3],
                target_price=str(100 + direction),
                label_observed_at=as_of + timedelta(minutes=15),
                target_return_bps=direction * 100,
            )
        )
    return result


def test_local_baselines_fit_predict_and_round_trip_without_pickle(tmp_path):
    from tradecopilot.forecast.baselines import fit_model, load_model, predict_model, save_model

    rows = examples()
    for name in ("prior", "momentum", "logistic"):
        model = fit_model(name, rows[:60], ForecastConfig())
        path = save_model(model, tmp_path / (name + ".json"))
        restored = load_model(path)
        np.testing.assert_allclose(model.probabilities(rows[60:]), restored.probabilities(rows[60:]))
        predictions = predict_model(restored, rows[60:], "test-data", ForecastConfig())
        assert len(predictions) == 30
        assert all(p.execution == "local" and set(p.probabilities) == {"DOWN", "FLAT", "UP"} for p in predictions)
        assert model.training_ids == tuple(item.example_id for item in rows[:60])


def test_held_out_changes_do_not_change_fitted_scaling_or_weights():
    from tradecopilot.forecast.baselines import fit_model

    rows = examples()
    first = fit_model("logistic", rows[:60], ForecastConfig())
    rows[-1] = rows[-1].model_copy(update={"features": {name: 99999.0 for name in FEATURE_NAMES}, "label": "DOWN"})
    second = fit_model("logistic", rows[:60], ForecastConfig())
    assert first.model_id == second.model_id
    assert first.parameters == second.parameters


def test_calibration_uses_only_supplied_validation_rows():
    from tradecopilot.forecast.baselines import calibrate_model, fit_model

    rows = examples()
    original = fit_model("logistic", rows[:60], ForecastConfig())
    calibrated = calibrate_model(original, rows[60:75])
    assert calibrated.validation_ids == tuple(row.example_id for row in rows[60:75])
    assert calibrated.training_ids == original.training_ids
    assert calibrated.parameters == original.parameters
    assert calibrated.temperature > 0


def test_model_rejects_changed_feature_target_contract():
    from tradecopilot.forecast.baselines import fit_model

    rows = examples()
    model = fit_model("logistic", rows[:60], ForecastConfig())
    wrong = rows[60].model_copy(update={"config_id": "different"})
    with pytest.raises(ValueError, match="config"):
        model.probabilities([wrong])


def test_weighting_ablation_preserves_balanced_default_and_records_none():
    from tradecopilot.forecast.baselines import fit_model

    rows = examples()
    default = fit_model("logistic", rows[:60], ForecastConfig())
    balanced = fit_model("logistic", rows[:60], ForecastConfig(), class_weight="balanced")
    unweighted = fit_model("logistic", rows[:60], ForecastConfig(), class_weight=None)
    assert default.model_id == balanced.model_id
    assert unweighted.parameters["fit_settings"]["class_weight"] is None
    assert unweighted.training_ids == default.training_ids
