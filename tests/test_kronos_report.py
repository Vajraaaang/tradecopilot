import pytest


def test_private_report_is_immutable_and_every_inventory_item_is_checked(tmp_path):
    from tradecopilot.forecast.kronos_report import load_report, write_report

    path = write_report(
        tmp_path / "report",
        {
            "evidence_mode": "retrospective_development_pilot",
            "connection": {"mode": "paper"},
            "models": {},
            "cases": [],
        },
        {"inputs.json": b"[]\n"},
    )
    first = load_report(path)
    assert first["schema_version"] == "kronos-paper-report-v1"
    with pytest.raises(FileExistsError):
        write_report(path.parent, {}, {})
    (path.parent / "inputs.json").write_bytes(b"[1]\n")
    with pytest.raises(ValueError):
        load_report(path)


def test_report_rejects_nonfinite_data_and_path_escape_before_writing(tmp_path):
    from tradecopilot.forecast.kronos_report import write_report

    with pytest.raises(ValueError):
        write_report(tmp_path / "bad", {"accuracy": float("nan")}, {})
    with pytest.raises(ValueError):
        write_report(tmp_path / "escape", {}, {"../credentials.json": b"private"})
    assert not (tmp_path / "bad").exists()
    assert not (tmp_path / "escape").exists()


def test_future_outcome_join_retains_original_forecast_and_requires_observed_receipt(tmp_path):
    import copy
    from datetime import UTC, datetime, timedelta

    from tradecopilot.forecast.bars import HistoricalBar
    from tradecopilot.forecast.kronos_report import attach_outcomes

    start = datetime(2026, 10, 7, 13, 30, tzinfo=UTC)
    future = [start + timedelta(minutes=i) for i in range(1, 16)]
    row = {
        "group": "prospective",
        "symbol": "AAPL",
        "future_times": [t.isoformat() for t in future],
        "actual_close": None,
        "forecasts": {
            "mini": {"status": "ok", "generated_at": "2026-10-06T22:00:00+00:00", "mean_close": [100.0] * 15}
        },
    }
    report = {"connection": {"feed": "sip"}, "cases": [row]}
    originals = copy.deepcopy(report)
    bars = [
        HistoricalBar(
            symbol="AAPL",
            start_time=t - timedelta(minutes=1),
            end_time=t,
            available_at=t,
            opening=101,
            high=101,
            low=101,
            close=101,
            volume=1,
            source="alpaca_sip_1min_bar",
        )
        for t in future
    ]
    with pytest.raises(ValueError):
        attach_outcomes(report, bars, {"feed": "sip", "receipt_at": "2026-10-07T13:40:00+00:00"})
    result = attach_outcomes(report, bars, {"feed": "sip", "receipt_at": "2026-10-07T13:46:00+00:00"})
    assert result["cases"][0]["actual_close"] == [101.0] * 15
    assert result["cases"][0]["forecasts"] == originals["cases"][0]["forecasts"]
    assert report == originals
