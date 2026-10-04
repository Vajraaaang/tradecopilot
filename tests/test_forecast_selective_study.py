from datetime import date

import pytest

from test_forecast_selective import cases
from tradecopilot.forecast.contracts import DatasetManifest, ForecastConfig, content_hash
from tradecopilot.forecast.ohlcv_features import OHLCV_FEATURE_NAMES, OhlcvFeatureRecord
from tradecopilot.forecast.sessions import CALENDAR_VERSION


def prepared():
    rows = cases(100, 30)
    records = []
    for row in rows:
        values = dict.fromkeys(OHLCV_FEATURE_NAMES, 0.0)
        values["return_5m_bps"] = {"DOWN": -100.0, "FLAT": 0.0, "UP": 100.0}[row.label]
        records.append(
            OhlcvFeatureRecord(
                base_example_id=row.example_id,
                config_id=row.config_id,
                symbol=row.symbol,
                as_of=row.as_of,
                values=values,
                input_bars_hash=row.example_id,
            )
        )
    return rows, records


def test_offline_study_freezes_selection_before_test_and_rejects_reuse(tmp_path):
    from tradecopilot.forecast.experiment import load_report
    from tradecopilot.forecast.selective_study import run_prepared_study

    rows, records = prepared()
    config = ForecastConfig(symbols=("AAPL",))
    manifest = DatasetManifest(
        dataset_id="fixture-forecast-dataset",
        observations_hash="f" * 64,
        examples_hash=content_hash([r.model_dump(mode="json") for r in rows]),
        config=config,
        calendar_version=CALENDAR_VERSION,
        provenance="historical",
        observation_count=len(rows),
        example_count=len(rows),
        labeled_count=len(rows),
        sessions=tuple(sorted({r.session_date for r in rows})),
    )
    first = run_prepared_study(
        rows,
        records,
        config,
        {"data_id": "fixture", "source_metadata": {"fixture": True}},
        tmp_path / "one",
        dataset_manifest=manifest,
    )
    report = load_report(first)
    assert report["evaluation_kind"] == "synthetic_selective_demo"
    assert report["protocol"]["paid_api_calls"] == 0
    assert report["dataset"]["dataset_id"] == manifest.dataset_id
    assert report["bar_dataset_id"] == "fixture"
    import json

    predictions = [json.loads(line) for line in (first.parent / "predictions.jsonl").read_text().splitlines()]
    assert all(p["dataset_id"] == manifest.dataset_id for p in predictions)
    assert report["split"]["test"]["sessions"] == sorted({str(r.session_date) for r in rows})[-10:]
    assert (first.parent / "frozen-selection.json").is_file()
    assert (tmp_path / "one" / "frozen-model.json").is_file()
    assert "UP" in report["selection_metrics"]["by_predicted_label"]
    # Deliberately planted fixture: it demonstrates execution only, never market performance.
    changed = [
        r.model_copy(update={"label": "DOWN"}) if str(r.session_date) in report["split"]["test"]["sessions"] else r
        for r in rows
    ]
    second = run_prepared_study(
        changed, records, config, {"data_id": "fixture", "source_metadata": {"fixture": True}}, tmp_path / "two"
    )
    later = load_report(second)
    assert report["selection"] == later["selection"]
    assert report["selection_metrics"] != later["selection_metrics"]
    with pytest.raises(FileExistsError):
        run_prepared_study(rows, records, config, {"data_id": "fixture"}, tmp_path / "one")


def test_inspected_dates_and_wrong_feature_alignment_fail_before_fitting(tmp_path):
    from tradecopilot.forecast.selective_study import run_prepared_study

    rows, records = prepared()
    config = ForecastConfig(symbols=("AAPL",))
    bad = rows[0].model_copy(update={"session_date": date(2026, 9, 25)})
    with pytest.raises(ValueError, match="inspected"):
        run_prepared_study([bad, *rows[1:]], records, config, {"data_id": "fixture"}, tmp_path / "bad")
    with pytest.raises(ValueError, match="align"):
        run_prepared_study(rows, records[1:] + records[:1], config, {"data_id": "fixture"}, tmp_path / "wrong")


def test_bar_replay_prior_close_is_available_and_missing_prior_is_skipped():
    from datetime import UTC, datetime, timedelta

    from tradecopilot.forecast.bars import HistoricalBar
    from tradecopilot.forecast.selective_study import replay_observations

    at = datetime(2026, 8, 3, 14, tzinfo=UTC)

    def bar(when, close="100", available=None):
        end = when + timedelta(minutes=1)
        return HistoricalBar(
            symbol="AAPL",
            start_time=when,
            end_time=end,
            available_at=available or end,
            opening=close,
            high=close,
            low=close,
            close=close,
            volume="10",
            source="alpaca_sip_1min_bar",
        )

    first = bar(at)
    delayed = bar(at + timedelta(minutes=1), "200", at + timedelta(days=2))
    current = bar(at + timedelta(days=1), "101")
    rows = replay_observations([first, delayed, current])
    assert len(rows) == 1 and rows[0].previous_close == first.close
    assert rows[0].last == current.close and rows[0].receipt_timestamp == current.end_time


def test_bar_entry_point_fails_closed_on_insufficient_history_and_preserves_input(tmp_path):
    from datetime import UTC, datetime, timedelta

    from tradecopilot.forecast.bars import HistoricalBar, write_bar_dataset
    from tradecopilot.forecast.selective_study import run_selective_study

    bars = []
    for day in range(2):
        for minute in range(45):
            at = datetime(2026, 8, 3 + day, 14, tzinfo=UTC) + timedelta(minutes=minute)
            bars.append(
                HistoricalBar(
                    symbol="AAPL",
                    start_time=at,
                    end_time=at + timedelta(minutes=1),
                    available_at=at + timedelta(minutes=1),
                    opening="100",
                    high="100",
                    low="100",
                    close="100",
                    volume="10",
                    source="alpaca_sip_1min_bar",
                )
            )
    bars = [
        bar.model_copy(update={"symbol": symbol}) for bar in bars for symbol in ("AAPL", "MSFT", "AMZN", "NFLX", "TSLA")
    ]
    source = tmp_path / "bars"
    write_bar_dataset(source, bars, {"fixture": True})
    before = (source / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="100"):
        run_selective_study(source, tmp_path / "study")
    assert not (tmp_path / "study").exists()
    assert (source / "manifest.json").read_bytes() == before


def test_inspected_primer_bar_is_rejected_before_dataset_construction(tmp_path):
    from datetime import UTC, datetime, timedelta

    from tradecopilot.forecast.bars import HistoricalBar, write_bar_dataset
    from tradecopilot.forecast.selective_study import run_selective_study

    at = datetime(2026, 9, 30, 14, tzinfo=UTC)
    bars = [
        HistoricalBar(
            symbol=symbol,
            start_time=at,
            end_time=at + timedelta(minutes=1),
            available_at=at + timedelta(minutes=1),
            opening="100",
            high="100",
            low="100",
            close="100",
            volume="10",
            source="alpaca_sip_1min_bar",
        )
        for symbol in ("AAPL", "MSFT", "AMZN", "NFLX", "TSLA")
    ]
    write_bar_dataset(tmp_path / "bars", bars, {"fixture": True})
    with pytest.raises(ValueError, match="inspected"):
        run_selective_study(tmp_path / "bars", tmp_path / "study")
    assert not (tmp_path / "study").exists()
