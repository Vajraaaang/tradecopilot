import json
from datetime import timedelta

import pytest

from tradecopilot.forecast.contracts import ForecastConfig, ForecastPrediction, content_hash
from tradecopilot.forecast.dataset import build_dataset
from tradecopilot.forecast.demo import synthetic_observations


@pytest.fixture(scope="module")
def dataset():
    config = ForecastConfig(symbols=("AAPL",))
    return build_dataset(synthetic_observations(config), config)


def test_experiment_keeps_test_separate_and_reproduces_quality_metrics(tmp_path, dataset):
    from tradecopilot.forecast.experiment import load_report, run_experiment

    manifest, examples = dataset
    first = run_experiment(manifest, examples, tmp_path / "first", include_jev_fixture=False)
    second = run_experiment(manifest, examples, tmp_path / "second", include_jev_fixture=False)
    one, two = load_report(first), load_report(second)
    assert one["evaluation_kind"] == "synthetic_demo"
    assert one["experiment_id"] == two["experiment_id"]
    assert len(one["models"]) == 4
    for a, b in zip(one["models"], two["models"], strict=True):
        assert a["metrics"]["log_loss"] == b["metrics"]["log_loss"]
        assert a["metrics"]["eligible"] == one["split"]["test"]["examples"]
    splits = json.loads((first.parent / "splits.json").read_text())
    assert not set(splits["test"]) & (set(splits["train"]) | set(splits["validation"]))
    model = json.loads((first.parent / "logistic-calibrated.json").read_text())
    assert set(model["training_ids"]) == set(splits["train"])
    assert set(model["validation_ids"]) == set(splits["validation"])
    assert one["source"]["source_sha256"]
    with pytest.raises(FileExistsError):
        run_experiment(manifest, examples, first.parent, include_jev_fixture=False)


def test_report_verifies_predictions_and_refuses_path_traversal(tmp_path, dataset):
    from tradecopilot.forecast.experiment import load_report, run_experiment

    manifest, examples = dataset
    path = run_experiment(manifest, examples, tmp_path / "run", include_jev_fixture=False)
    predictions = path.parent / "predictions.jsonl"
    predictions.write_text(predictions.read_text() + "{}\n")
    with pytest.raises(ValueError, match="integrity"):
        load_report(path)
    payload = json.loads(path.read_text())
    payload["artifacts"]["../secret"] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="integrity"):
        load_report(path)


def test_benchmark_refuses_single_session_capture(tmp_path, dataset):
    from tradecopilot.forecast.experiment import run_experiment

    manifest, examples = dataset
    day = examples[0].session_date
    with pytest.raises(ValueError):
        run_experiment(manifest, [row for row in examples if row.session_date == day], tmp_path / "bad")


@pytest.fixture
def pilot(dataset):
    from tradecopilot.forecast.jev import FORECAST_PROMPT_VERSION
    from tradecopilot.jev import JEV_MODEL

    manifest, rows = dataset
    original = rows[0].model_copy(update={
        "label": None, "target_price": None, "label_observed_at": None, "target_return_bps": None,
        "provenance": "market",
    })
    prediction = ForecastPrediction(
        example_id=original.example_id,
        dataset_id=content_hash({"config_id": manifest.config.config_id, "example_id": original.example_id}),
        model_id=JEV_MODEL, model_version=JEV_MODEL, prompt_version=FORECAST_PROMPT_VERSION,
        generated_at=original.as_of + timedelta(seconds=1), execution="live_api", status="ok",
        probabilities={"DOWN": 0.1, "FLAT": 0.7, "UP": 0.2}, request_id="1", estimated_cost_usd=0.001,
    )
    return manifest.config, original, prediction


def test_pilot_keeps_pending_outcomes_unscored(tmp_path, pilot):
    from tradecopilot.forecast.experiment import load_report, write_pilot_report

    config, original, prediction = pilot
    path = write_pilot_report(config, [original], [prediction], [], tmp_path / "pilot",
                              recorded_predictions=[prediction])
    report = load_report(path)
    metrics = report["models"][0]["metrics"]
    assert report["evaluation_kind"] == "prospective_pilot"
    assert metrics["pending_outcomes"] == 1 and metrics["scored"] == 0 and metrics["accuracy"] is None


@pytest.mark.parametrize("update", [
    {"dataset_id": "wrong"}, {"model_id": "wrong"}, {"prompt_version": "wrong"},
    {"reason": "retrospective_api_inference"}, {"request_id": None},
])
def test_pilot_refuses_mismatched_inference_provenance(tmp_path, pilot, update):
    from tradecopilot.forecast.experiment import write_pilot_report

    config, original, prediction = pilot
    changed = prediction.model_copy(update=update)
    with pytest.raises(ValueError, match="prospectively"):
        write_pilot_report(config, [original], [changed], [], tmp_path / "pilot", recorded_predictions=[changed])


def test_pilot_requires_matching_record_in_durable_store(tmp_path, pilot):
    from tradecopilot.forecast.experiment import write_pilot_report

    config, original, prediction = pilot
    with pytest.raises(ValueError, match="prospectively"):
        write_pilot_report(config, [original], [prediction], [], tmp_path / "pilot", recorded_predictions=[])
