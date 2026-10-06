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
    report = {"connection": {"feed": "sip"}, "cases": [row], "published_at": "2026-10-06T22:01:00+00:00"}
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
    late = report | {"published_at": "2026-10-07T13:46:00+00:00"}
    with pytest.raises(ValueError, match="publication"):
        attach_outcomes(late, bars, {"feed": "sip", "receipt_at": "2026-10-07T13:46:00+00:00"})
    missing = {k: v for k, v in report.items() if k != "published_at"}
    with pytest.raises(ValueError, match="publication"):
        attach_outcomes(missing, bars, {"feed": "sip", "receipt_at": "2026-10-07T13:46:00+00:00"})


def test_publication_is_measured_and_expired_forecasts_never_form_a_bundle(tmp_path):
    from datetime import UTC, datetime, timedelta

    from tradecopilot.forecast.kronos_report import load_report, write_report

    now = datetime.now(UTC)
    row = {
        "group": "prospective",
        "future_times": [(now + timedelta(days=1)).isoformat()],
        "forecasts": {"mini": {"status": "ok", "generated_at": now.isoformat()}},
    }
    report = {"cases": [row], "published_at": "2000-01-01T00:00:00+00:00"}
    result = load_report(write_report(tmp_path / "fresh", report, {}))
    assert now <= datetime.fromisoformat(result["published_at"]) < now + timedelta(days=1)
    row["future_times"] = ["2000-01-01T00:00:00+00:00"]
    row["forecasts"]["mini"]["generated_at"] = "1999-12-31T23:00:00+00:00"
    with pytest.raises(ValueError, match="publication"):
        write_report(tmp_path / "expired", report, {})
    assert not (tmp_path / "expired").exists()


def test_grading_preserves_original_publication_and_deadline_crossing_is_rejected(tmp_path, monkeypatch):
    from datetime import UTC, datetime, timedelta

    import tradecopilot.forecast.kronos_report as module

    now = datetime.now(UTC)
    target = now + timedelta(seconds=1)
    report = {
        "cases": [{"group": "prospective", "future_times": [target.isoformat()],
                   "forecasts": {"mini": {"status": "ok", "generated_at": now.isoformat()}}}],
    }
    first = module.load_report(module.write_report(tmp_path / "first", report, {}))
    reads = iter((target + timedelta(seconds=1),))

    class Clock:
        @staticmethod
        def now(zone):
            return next(reads)

        fromisoformat = datetime.fromisoformat

    monkeypatch.setattr(module, "datetime", Clock)
    derived = module.load_report(module.write_report(
        tmp_path / "graded", first | {"parent_report_id": first["report_id"]}, {},
    ))
    assert derived["published_at"] == first["published_at"]
    assert derived["bundle_created_at"] > derived["published_at"]
    reads = iter((now, now, target))
    with pytest.raises(ValueError, match="publication"):
        module.write_report(tmp_path / "crossed", report, {})
    assert not (tmp_path / "crossed").exists()
