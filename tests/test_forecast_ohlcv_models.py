from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from tradecopilot.forecast.contracts import FEATURE_NAMES, ForecastConfig, ForecastExample
from tradecopilot.forecast.ohlcv_features import OHLCV_FEATURE_NAMES, OhlcvFeatureRecord


def rows(count=90):
    config = ForecastConfig(symbols=("AAPL", "MSFT"))
    examples, features = [], []
    for index in range(count):
        direction = index % 3 - 1
        at = datetime(2026, 9, 15, 14, tzinfo=UTC) + timedelta(minutes=index)
        row = ForecastExample(
            config_id=config.config_id,
            symbol=config.symbols[index % 2],
            as_of=at,
            target_time=at + timedelta(minutes=15),
            session_date=date(2026, 9, 15),
            anchor_price="100",
            features=dict.fromkeys(FEATURE_NAMES, 0.0),
            observation_ids=(str(index),),
            provenance="historical",
            label=("DOWN", "FLAT", "UP")[index % 3],
            target_price=str(100 + direction),
            target_return_bps=direction * 100,
            label_observed_at=at + timedelta(minutes=15),
        )
        values = dict.fromkeys(OHLCV_FEATURE_NAMES, 0.0)
        values.update(
            return_5m_bps=direction * 20,
            return_60m_bps=None if index % 7 == 0 else direction * 70,
            log_volume_1m=5 + index / 100,
        )
        examples.append(row)
        features.append(
            OhlcvFeatureRecord(
                base_example_id=row.example_id,
                config_id=config.config_id,
                symbol=row.symbol,
                as_of=at,
                values=values,
                input_bars_hash=str(index),
            )
        )
    return config, examples, features


@pytest.mark.parametrize("kind", ["ohlcv_logistic", "ohlcv_hist_gradient_boosting"])
def test_portable_models_roundtrip_without_pickle_and_probability_order(kind, tmp_path):
    from tradecopilot.forecast.ohlcv_models import fit_ohlcv_model, load_ohlcv_model, save_ohlcv_model

    config, examples, features = rows(600)
    model, trained = fit_ohlcv_model(kind, examples[:450], features[:450], config)
    path = save_ohlcv_model(model, tmp_path / "model.json")
    restored = load_ohlcv_model(path)
    held = features[450:]
    held[0] = held[0].model_copy(update={"values": held[0].values | {"return_5m_bps": None}})
    expected = np.asarray(trained.predict_proba(model.training_matrix(held)))
    np.testing.assert_allclose(restored.probabilities(held), expected, atol=1e-10)
    assert model.training_ids == tuple(row.example_id for row in examples[:450])
    assert np.allclose(restored.probabilities(held).sum(axis=1), 1)
    if kind == "ohlcv_hist_gradient_boosting":
        assert any(not node["leaf"] for tree in model.parameters["trees"] for node in tree)


def test_preprocessing_uses_training_only_and_rejects_mismatched_records():
    from tradecopilot.forecast.ohlcv_models import fit_ohlcv_model

    config, examples, features = rows()
    first, _ = fit_ohlcv_model("ohlcv_logistic", examples[:60], features[:60], config)
    features[-1] = features[-1].model_copy(update={"values": dict.fromkeys(OHLCV_FEATURE_NAMES, 99999.0)})
    second, _ = fit_ohlcv_model("ohlcv_logistic", examples[:60], features[:60], config)
    assert first.model_id == second.model_id and first.parameters == second.parameters
    with pytest.raises(ValueError, match="align"):
        fit_ohlcv_model("ohlcv_logistic", examples[:60], features[1:61], config)
    with pytest.raises(ValueError, match="config"):
        first.probabilities([features[60].model_copy(update={"config_id": "wrong-horizon"})])


def test_model_artifact_rejects_cyclic_tree_and_unknown_schema():
    from tradecopilot.forecast.ohlcv_models import OhlcvModelArtifact, fit_ohlcv_model

    config, examples, features = rows()
    model, _ = fit_ohlcv_model("ohlcv_hist_gradient_boosting", examples, features, config)
    data = model.model_dump(mode="json")
    data["feature_version"] = "unknown"
    with pytest.raises(ValueError):
        OhlcvModelArtifact.model_validate(data)
    data = model.model_dump(mode="json")
    node = data["parameters"]["trees"][0][0]
    node.update(leaf=False, left=0, right=0)
    with pytest.raises(ValueError, match="tree"):
        OhlcvModelArtifact.model_validate(data)


def test_histogram_missing_only_splits_roundtrip_in_finite_json(tmp_path):
    from tradecopilot.forecast.ohlcv_models import fit_ohlcv_model, load_ohlcv_model, save_ohlcv_model

    config, examples, features = rows(900)
    features = [
        record.model_copy(
            update={"values": dict.fromkeys(OHLCV_FEATURE_NAMES, None if index % 3 == 2 else float(index % 3))}
        )
        for index, record in enumerate(features)
    ]
    model, native = fit_ohlcv_model("ohlcv_hist_gradient_boosting", examples, features, config)
    assert any(node.get("missing_only") for tree in model.parameters["trees"] for node in tree)
    path = save_ohlcv_model(model, tmp_path / "missing-model.json")
    assert "Infinity" not in path.read_text() and "NaN" not in path.read_text()
    restored = load_ohlcv_model(path)
    expected = native.predict_proba(model.training_matrix(features))
    np.testing.assert_allclose(restored.probabilities(features), expected, atol=1e-10)


def test_bounded_candidate_settings_preserve_defaults_and_portability():
    from tradecopilot.forecast.ohlcv_models import fit_ohlcv_model

    config, examples, features = rows(120)
    model, native = fit_ohlcv_model("ohlcv_logistic", examples, features, config, settings={"C": 0.1})
    assert model.parameters["fit_settings"]["C"] == 0.1
    np.testing.assert_allclose(
        model.probabilities(features), native.predict_proba(model.training_matrix(features)), atol=1e-10
    )
    boost, _ = fit_ohlcv_model(
        "ohlcv_hist_gradient_boosting", examples, features, config, settings={"l2_regularization": 50.0}
    )
    assert boost.parameters["fit_settings"]["l2_regularization"] == 50
    for settings in ({"early_stopping": True}, {"C": -1}, {"C": float("nan")}, {"C": True}):
        with pytest.raises(ValueError, match="settings"):
            fit_ohlcv_model("ohlcv_logistic", examples, features, config, settings=settings)
