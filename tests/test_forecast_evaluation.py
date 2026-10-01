from datetime import UTC, date, datetime, timedelta

import pytest

from tradecopilot.forecast.contracts import FEATURE_NAMES, ForecastConfig, ForecastExample, ForecastPrediction


def row(day, label="UP"):
    timestamp = datetime(2026, 9, day, 15, 0, tzinfo=UTC)
    direction = {"UP": 1, "FLAT": 0, "DOWN": -1}[label]
    return ForecastExample(
        config_id=ForecastConfig().config_id,
        symbol="AAPL",
        as_of=timestamp,
        target_time=timestamp + timedelta(minutes=15),
        session_date=date(2026, 9, day),
        anchor_price="100",
        features={name: 0.0 for name in FEATURE_NAMES},
        observation_ids=(f"obs-{day}",),
        provenance="synthetic",
        label=label,
        target_price=str(100 + direction),
        label_observed_at=timestamp + timedelta(minutes=15),
        target_return_bps=direction * 100,
    )


def test_split_keeps_whole_sessions_and_purges_future_labels():
    from tradecopilot.forecast.evaluation import split_examples

    rows = [row(day) for day in range(14, 22)]
    # A malformed/overlapping label cannot leak through even if an importer supplies one.
    rows[0] = rows[0].model_copy(update={"label_observed_at": rows[5].as_of})
    split = split_examples(rows)
    assert rows[0].example_id not in {item.example_id for item in split["train"]}
    assert max(item.label_observed_at for item in split["train"]) < min(item.as_of for item in split["validation"])
    assert max(item.label_observed_at for item in split["validation"]) < min(item.as_of for item in split["test"])
    assert not ({item.session_date for item in split["train"]} & {item.session_date for item in split["test"]})


def test_single_session_capture_is_not_an_experiment():
    from tradecopilot.forecast.evaluation import split_examples

    with pytest.raises(ValueError, match="five"):
        split_examples([row(14)])


def test_metrics_score_abstained_probabilities_and_report_coverage_separately():
    from tradecopilot.forecast.evaluation import score_predictions

    examples = [row(14, "UP"), row(15, "DOWN")]
    predictions = [
        ForecastPrediction(
            example_id=item.example_id,
            dataset_id="dataset",
            model_id="test",
            generated_at=datetime.now(UTC),
            execution="fixture",
            status="ok" if index == 0 else "abstained",
            reason=None if index == 0 else "low_confidence",
            probabilities={"DOWN": 0.1, "FLAT": 0.1, "UP": 0.8},
            model_confidence=0.8 if index == 0 else 0.2,
            estimated_cost_usd=0,
            latency_ms=2 + index,
        )
        for index, item in enumerate(examples)
    ]
    metrics = score_predictions(examples, predictions)
    assert metrics["accuracy"] == 0.5
    assert metrics["brier_score"] == pytest.approx(0.76)
    assert metrics["coverage"] == 0.5 and metrics["selective_accuracy"] == 1.0
    assert metrics["scored"] == 2 and metrics["abstained"] == 1
    assert metrics["latency_ms"]["p95"] == pytest.approx(2.95)
    assert sum(point["count"] for point in metrics["reliability"]["UP"]) == 2


def test_missing_predictions_and_errors_do_not_look_like_perfect_performance():
    from tradecopilot.forecast.evaluation import score_predictions

    examples = [row(14), row(15)]
    prediction = ForecastPrediction(
        example_id=examples[0].example_id,
        dataset_id="d",
        model_id="test",
        generated_at=datetime.now(UTC),
        execution="fixture",
        status="error",
        reason="timeout",
    )
    metrics = score_predictions(examples, [prediction])
    assert metrics["accuracy"] is None and metrics["brier_score"] is None
    assert metrics["coverage"] == 0 and metrics["errors"] == 1 and metrics["missing_predictions"] == 1


def test_unknown_or_duplicate_prediction_ids_are_rejected():
    from tradecopilot.forecast.evaluation import score_predictions

    example = row(14)
    prediction = ForecastPrediction(
        example_id="unknown",
        dataset_id="d",
        model_id="test",
        generated_at=datetime.now(UTC),
        execution="fixture",
        status="error",
        reason="test",
    )
    with pytest.raises(ValueError, match="unknown"):
        score_predictions([example], [prediction])
    prediction = prediction.model_copy(update={"example_id": example.example_id})
    with pytest.raises(ValueError, match="duplicate"):
        score_predictions([example], [prediction, prediction])


def test_reliability_bins_do_not_lose_exact_decimal_boundary_values():
    from tradecopilot.forecast.evaluation import score_predictions

    example = row(14)
    prediction = ForecastPrediction(
        example_id=example.example_id,
        dataset_id="d",
        model_id="test",
        generated_at=datetime.now(UTC),
        execution="fixture",
        status="ok",
        probabilities={"DOWN": 0.2, "FLAT": 0.3, "UP": 0.5},
    )
    metrics = score_predictions([example], [prediction])
    assert all(sum(point["count"] for point in bins) == 1 for bins in metrics["reliability"].values())
