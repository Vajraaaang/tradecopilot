from datetime import UTC, datetime, timedelta

import pytest

from tradecopilot.forecast.contracts import ForecastConfig
from tradecopilot.forecast.dataset import build_dataset
from tradecopilot.forecast.demo import synthetic_observations


def historical_dataset():
    config = ForecastConfig(symbols=("AAPL",))
    rows = [row.model_copy(update={"provenance": "historical"}) for row in synthetic_observations(config)]
    manifest, examples = build_dataset(rows, config)
    source = {
        "dataset_id": manifest.dataset_id,
        "observations_hash": manifest.observations_hash,
        "availability_assumption": "unit_test_bar_end_replay",
    }
    return manifest, examples, source


def test_historical_run_requires_source_metadata(tmp_path):
    from tradecopilot.forecast.experiment import load_report, run_experiment

    manifest, examples, source = historical_dataset()
    with pytest.raises(ValueError, match="source"):
        run_experiment(manifest, examples, tmp_path / "missing", include_jev_fixture=False)
    path = run_experiment(manifest, examples, tmp_path / "valid", include_jev_fixture=False, historical_source=source)
    assert load_report(path)["data_source"] == source
    assert (path.parent / "historical-source.json").is_file()


def test_selection_is_deterministic_and_does_not_read_labels():
    from tradecopilot.forecast.retrospective import select_cohort

    _, rows, _ = historical_dataset()
    selected = select_cohort(rows, 10)
    changed = [row.model_copy(update={"label": "DOWN"}) for row in reversed(rows)]
    assert [r.example_id for r in selected] == [r.example_id for r in select_cohort(changed, 10)]
    assert len({r.example_id for r in selected}) == 10


def test_retrospective_run_preserves_complete_selection_and_equal_comparison_cohort(tmp_path):
    from tradecopilot.forecast.experiment import load_report, run_experiment
    from tradecopilot.forecast.jev import JevForecaster
    from tradecopilot.forecast.retrospective import run_retrospective
    from tradecopilot.jev import JEV_MODEL

    manifest, examples, source = historical_dataset()
    base = run_experiment(
        manifest, examples, tmp_path / "baseline", include_jev_fixture=False, historical_source=source
    )
    now = [datetime(2026, 10, 1, 18, tzinfo=UTC)]
    requests = []

    def transport(payload, key):
        requests.append(payload)
        if len(requests) == 2:
            raise TimeoutError("private secret")
        return {
            "model": JEV_MODEL,
            "answers": {
                "direction": {
                    "type": "choice",
                    "choice": "UP",
                    "probabilities": {"DOWN": 0.1, "FLAT": 0.2, "UP": 0.7},
                    "confidence": 0.8,
                }
            },
            "usage": {"input_tokens": 100},
        }

    def sleep(seconds):
        now[0] += timedelta(seconds=seconds)

    forecaster = JevForecaster("unit-test", tmp_path / "ledger.sqlite3", clock=lambda: now[0], transport=transport)
    path = run_retrospective(manifest, examples, base, tmp_path / "sample", forecaster, count=3, sleep=sleep)
    report = load_report(path)
    assert report["evaluation_kind"] == "retrospective_historical_sample"
    assert report["models"][-1]["metrics"]["errors"] == 1
    assert all(m["metrics"]["eligible"] == 3 and m["metrics"]["attempted"] == 3 for m in report["models"])
    assert len(report["cases"]) == len(requests) == 3
    assert "label" not in str(requests) and "target_price" not in str(requests)
    assert (path.parent / "selection.json").is_file()
    with pytest.raises(FileExistsError):
        run_retrospective(manifest, examples, base, path.parent, forecaster, count=3, sleep=sleep)
