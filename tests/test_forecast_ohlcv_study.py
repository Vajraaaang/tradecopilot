import csv
import hashlib
import io
import json
from datetime import UTC, datetime, timedelta
from zipfile import ZipFile
from zoneinfo import ZoneInfo

import pytest

from tradecopilot.forecast.contracts import ForecastConfig
from tradecopilot.forecast.dataset import build_dataset, write_dataset
from tradecopilot.forecast.demo import synthetic_observations
from tradecopilot.forecast.experiment import load_report, run_experiment
from tradecopilot.forecast.historical import import_frd_samples


def test_offline_study_has_identical_cohort_and_no_test_or_api_access(tmp_path, monkeypatch):
    from tradecopilot.forecast.evaluation import split_examples
    from tradecopilot.forecast.jev import JevForecaster
    from tradecopilot.forecast.ohlcv_study import run_ohlcv_validation

    def forbidden(*args, **kwargs):
        raise AssertionError("OHLCV CPU milestone cannot make a paid request")

    monkeypatch.setattr(JevForecaster, "predict", forbidden)
    config = ForecastConfig(symbols=("AAPL",))
    raw = synthetic_observations(config, minutes_per_session=90)
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    text = io.StringIO()
    writer = csv.writer(text)
    writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
    for row in raw:
        if row.provider_timestamp.second or (row.provider_timestamp.hour == 13 and row.provider_timestamp.minute == 30):
            continue
        start = row.provider_timestamp - timedelta(minutes=1)
        writer.writerow(
            [
                start.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S"),
                row.last,
                row.last + 1,
                row.last - 1,
                row.last,
                100,
            ]
        )
    path = archive_dir / "AAPL_sample.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("AAPL_1min_sample.csv", text.getvalue())
    data = path.read_bytes()
    (archive_dir / "downloads.json").write_text(
        json.dumps(
            [
                {
                    "symbol": "AAPL",
                    "url": "https://frd001.s3.us-east-2.amazonaws.com/frd_sample_stock_AAPL.zip",
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                    "retrieved_at": datetime(2026, 10, 1, tzinfo=UTC).isoformat(),
                }
            ]
        )
    )
    observations, source = import_frd_samples(archive_dir, config.symbols)
    manifest, examples = build_dataset(observations, config)
    source |= {"dataset_id": manifest.dataset_id, "observations_hash": manifest.observations_hash}
    dataset = write_dataset(tmp_path / "dataset", manifest, examples)
    baseline = run_experiment(
        manifest, examples, tmp_path / "baseline", include_jev_fixture=False, historical_source=source
    )
    result = run_ohlcv_validation(dataset, archive_dir, baseline, tmp_path / "study")
    report = load_report(result)
    split = split_examples(examples)
    assert report["evaluation_kind"] == "ohlcv_validation_development"
    assert len(report["models"]) == 5
    assert all(m["metrics"]["eligible"] == len(split["validation"]) for m in report["models"])
    assert {row["example_id"] for row in report["cases"]} == {row.example_id for row in split["validation"]}
    assert not {row["example_id"] for row in report["cases"]} & {row.example_id for row in split["test"]}
    assert report["protocol"]["paid_api_calls"] == 0
    with pytest.raises(FileExistsError):
        run_ohlcv_validation(dataset, archive_dir, baseline, tmp_path / "study")
