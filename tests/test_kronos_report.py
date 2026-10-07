import copy
import json
from datetime import UTC, datetime, timedelta

import pytest

OPEN = datetime(2026, 10, 7, 13, 30, tzinfo=UTC)
BEFORE_OPEN = datetime(2026, 10, 6, 22, tzinfo=UTC)


def _prospective_report():
    return {
        "connection": {"feed": "sip"},
        "protocol": {"protocol_id": "a" * 64},
        "source_data_id": "b" * 64,
        "cases": [{
            "group": "prospective", "symbol": "AAPL", "as_of": BEFORE_OPEN.isoformat(),
            "history_close": [100.0] * 60,
            "future_times": [(OPEN + timedelta(minutes=i)).isoformat() for i in range(1, 16)],
            "actual_close": None,
            "forecasts": {"mini": {"status": "ok", "generated_at": BEFORE_OPEN.isoformat(),
                                    "mean_close": [100.0] * 15}},
        }],
    }


def _freeze_publication(monkeypatch, *moments):
    import tradecopilot.forecast.kronos_report as module

    reads = iter(moments)

    class Clock(datetime):
        @classmethod
        def now(cls, zone):
            return next(reads)

    monkeypatch.setattr(module, "datetime", Clock)


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
    incomplete = attach_outcomes(result, [], {"feed": "sip", "receipt_at": "2026-10-07T13:47:00+00:00"})
    assert incomplete["cases"][0]["actual_close"] is None
    assert "outcomes_received_at" not in incomplete["cases"][0]
    assert incomplete["cases"][0]["outcome_status"] == "pending_missing_exact_candles"
    assert result["cases"][0]["actual_close"] == [101.0] * 15
    late = report | {"published_at": "2026-10-07T13:46:00+00:00"}
    with pytest.raises(ValueError, match="publication"):
        attach_outcomes(late, bars, {"feed": "sip", "receipt_at": "2026-10-07T13:46:00+00:00"})
    missing = {k: v for k, v in report.items() if k != "published_at"}
    with pytest.raises(ValueError, match="publication"):
        attach_outcomes(missing, bars, {"feed": "sip", "receipt_at": "2026-10-07T13:46:00+00:00"})


def test_publication_is_measured_and_expired_forecasts_never_form_a_bundle(tmp_path):
    from tradecopilot.forecast.kronos_report import load_report, write_report
    from tradecopilot.forecast.sessions import session_bounds

    now = datetime.now(UTC)
    day = now.date() + timedelta(days=1)
    while (bounds := session_bounds(day)) is None:
        day += timedelta(days=1)
    row = {
        "group": "prospective",
        "future_times": [(bounds[0] + timedelta(minutes=i)).isoformat() for i in range(1, 16)],
        "forecasts": {"mini": {"status": "ok", "generated_at": now.isoformat()}},
    }
    report = {"cases": [row], "published_at": "2000-01-01T00:00:00+00:00"}
    result = load_report(write_report(tmp_path / "fresh", report, {}))
    assert now <= datetime.fromisoformat(result["published_at"]) < bounds[0]
    day = now.date() - timedelta(days=1)
    while (past_bounds := session_bounds(day)) is None:
        day -= timedelta(days=1)
    row["future_times"] = [(past_bounds[0] + timedelta(minutes=i)).isoformat() for i in range(1, 16)]
    row["forecasts"]["mini"]["generated_at"] = "1999-12-31T23:00:00+00:00"
    with pytest.raises(ValueError, match="publication"):
        write_report(tmp_path / "expired", report, {})
    assert not (tmp_path / "expired").exists()


def test_grading_preserves_original_publication_and_deadline_crossing_is_rejected(tmp_path, monkeypatch):
    import tradecopilot.forecast.kronos_report as module

    report = _prospective_report()
    target = OPEN + timedelta(minutes=1)
    _freeze_publication(monkeypatch, BEFORE_OPEN, BEFORE_OPEN, BEFORE_OPEN)
    first_path = module.write_report(tmp_path / "first", report, {})
    first = module.load_report(first_path)
    _freeze_publication(monkeypatch, target + timedelta(seconds=1))
    derived = module.load_report(module.write_report(
        tmp_path / "graded", first | {"parent_report_id": first["report_id"]}, {},
        original_report_path=first_path,
    ))
    assert derived["published_at"] == first["published_at"]
    assert derived["bundle_created_at"] > derived["published_at"]
    _freeze_publication(monkeypatch, BEFORE_OPEN, BEFORE_OPEN, target)
    with pytest.raises(ValueError, match="publication"):
        module.write_report(tmp_path / "crossed", report, {})
    assert not (tmp_path / "crossed").exists()


@pytest.mark.parametrize("source,late", [("alpaca_iex_1min_bar", False), ("firstratedata_1min_bar", False),
                                          ("alpaca_sip_1min_bar", True)])
def test_outcome_join_rejects_wrong_source_and_unreceived_matched_bars(source, late):
    from tradecopilot.forecast.bars import HistoricalBar
    from tradecopilot.forecast.kronos_report import attach_outcomes

    report = _prospective_report() | {"published_at": BEFORE_OPEN.isoformat()}
    original = copy.deepcopy(report)
    receipt = OPEN + timedelta(minutes=16)
    bars = [HistoricalBar(
        symbol="AAPL", start_time=OPEN + timedelta(minutes=i - 1), end_time=OPEN + timedelta(minutes=i),
        available_at=receipt + timedelta(seconds=1) if late else OPEN + timedelta(minutes=i),
        opening=101, high=101, low=101, close=101, volume=1, source=source,
    ) for i in range(1, 16)]
    with pytest.raises(ValueError, match="outcome"):
        attach_outcomes(report, bars, {"feed": "sip", "receipt_at": receipt.isoformat()})
    assert report == original


@pytest.mark.parametrize("invalid", ["short", "empty", "duplicate", "gap", "reversed", "overnight",
                                    "weekend", "nonutc", "seconds", "before_open", "after_close"])
def test_prospective_grids_are_validated_even_without_successful_forecasts(tmp_path, invalid):
    from tradecopilot.forecast.kronos_report import write_report

    report = _prospective_report()
    row = report["cases"][0]
    row["forecasts"] = {"mini": {"status": "error", "error": "TimeoutError"}}
    times = row["future_times"]
    if invalid == "short":
        times.pop()
    elif invalid == "empty":
        times.clear()
    elif invalid == "duplicate":
        times[1] = times[0]
    elif invalid == "gap":
        times[-1] = (OPEN + timedelta(minutes=16)).isoformat()
    elif invalid == "reversed":
        times.reverse()
    elif invalid == "overnight":
        times[-1] = (OPEN + timedelta(days=1, minutes=15)).isoformat()
    elif invalid == "weekend":
        row["future_times"] = [(OPEN + timedelta(days=3, minutes=i)).isoformat() for i in range(1, 16)]
    elif invalid == "nonutc":
        times[0] = "2026-10-07T09:31:00-04:00"
    elif invalid == "seconds":
        times[0] = "2026-10-07T13:31:01+00:00"
    else:
        start = OPEN - timedelta(minutes=1) if invalid == "before_open" else OPEN.replace(hour=19, minute=50)
        row["future_times"] = [(start + timedelta(minutes=i)).isoformat() for i in range(1, 16)]
    with pytest.raises(ValueError, match="grid"):
        write_report(tmp_path / invalid, report, {})
    assert not (tmp_path / invalid).exists()


def test_successful_publication_before_last_but_after_first_target_is_rejected(tmp_path, monkeypatch):
    from tradecopilot.forecast.kronos_report import write_report

    _freeze_publication(monkeypatch, OPEN + timedelta(minutes=2))
    with pytest.raises(ValueError, match="first target"):
        write_report(tmp_path / "late", _prospective_report(), {})
    assert not (tmp_path / "late").exists()


def test_loading_and_grading_reject_integrity_valid_but_invalid_prospective_grid(tmp_path, monkeypatch):
    from tradecopilot.forecast.contracts import content_hash
    from tradecopilot.forecast.kronos_report import attach_outcomes, load_report, write_report

    _freeze_publication(monkeypatch, BEFORE_OPEN, BEFORE_OPEN, BEFORE_OPEN)
    path = write_report(tmp_path / "report", _prospective_report(), {})
    value = load_report(path)
    value["cases"][0]["future_times"][1] = value["cases"][0]["future_times"][0]
    value["report_id"] = content_hash({k: v for k, v in value.items() if k != "report_id"})
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="integrity"):
        load_report(path)
    with pytest.raises(ValueError, match="grid"):
        attach_outcomes(value, [], {"feed": "sip", "receipt_at": (OPEN + timedelta(minutes=16)).isoformat()})


def test_parent_id_without_verified_original_path_cannot_backdate_publication(tmp_path):
    from tradecopilot.forecast.kronos_report import write_report

    forged = _prospective_report() | {"parent_report_id": "c" * 64, "published_at": BEFORE_OPEN.isoformat()}
    with pytest.raises(ValueError, match="original"):
        write_report(tmp_path / "forged", forged, {})
    assert not (tmp_path / "forged").exists()


@pytest.mark.parametrize("changed", ["forecast", "history", "protocol", "source", "connection", "parent",
                                    "publication", "artifact"])
def test_derived_reports_bind_the_verified_original_forecast_and_context(tmp_path, monkeypatch, changed):
    from tradecopilot.forecast.kronos_report import load_report, write_report

    _freeze_publication(monkeypatch, BEFORE_OPEN, BEFORE_OPEN, BEFORE_OPEN)
    original_path = write_report(tmp_path / "original", _prospective_report(), {"inputs.json": b"original"})
    first = load_report(original_path)
    derived = copy.deepcopy(first) | {"parent_report_id": first["report_id"]}
    artifacts = {"inputs.json": b"original", "outcome-source.json": b"new outcomes"}
    if changed == "forecast":
        derived["cases"][0]["forecasts"]["mini"]["mean_close"][0] = 101
    elif changed == "history":
        derived["cases"][0]["history_close"][0] = 101
    elif changed == "protocol":
        derived["protocol"]["protocol_id"] = "d" * 64
    elif changed == "source":
        derived["source_data_id"] = "d" * 64
    elif changed == "connection":
        derived["connection"]["feed"] = "iex"
    elif changed == "parent":
        derived["parent_report_id"] = "d" * 64
    elif changed == "publication":
        derived["published_at"] = (BEFORE_OPEN - timedelta(days=1)).isoformat()
    else:
        artifacts["inputs.json"] = b"different"
    with pytest.raises(ValueError, match="original"):
        write_report(tmp_path / "derived", derived, artifacts, original_report_path=original_path)
    assert not (tmp_path / "derived").exists()


def test_regrading_changes_only_outcome_evidence_and_keeps_original_publication(tmp_path, monkeypatch):
    from tradecopilot.forecast.kronos_report import load_report, write_report

    _freeze_publication(monkeypatch, BEFORE_OPEN, BEFORE_OPEN, BEFORE_OPEN)
    original_path = write_report(tmp_path / "original", _prospective_report(), {"paths.npz": b"fixed paths"})
    original = load_report(original_path)
    first = copy.deepcopy(original) | {"parent_report_id": original["report_id"], "outcome_source_id": "e" * 64}
    first["cases"][0] |= {"actual_close": [101.0] * 15, "outcome_status": "observed",
                         "outcomes_received_at": (OPEN + timedelta(minutes=16)).isoformat()}
    _freeze_publication(monkeypatch, OPEN + timedelta(minutes=16))
    first_path = write_report(tmp_path / "graded", first,
                              {"paths.npz": b"fixed paths", "outcome-source.json": b"first source"},
                              original_report_path=original_path)
    graded = load_report(first_path)
    correction = copy.deepcopy(graded) | {"parent_report_id": graded["report_id"], "outcome_source_id": "f" * 64}
    correction["cases"][0] |= {"actual_close": None, "outcome_status": "pending_missing_exact_candles"}
    correction["cases"][0].pop("outcomes_received_at")
    _freeze_publication(monkeypatch, OPEN + timedelta(minutes=17))
    corrected = load_report(write_report(
        tmp_path / "corrected", correction,
        {"paths.npz": b"fixed paths", "outcome-source.json": b"corrected source"}, original_report_path=first_path,
    ))
    assert corrected["published_at"] == original["published_at"]
    assert corrected["cases"][0]["forecasts"] == original["cases"][0]["forecasts"]
    assert corrected["cases"][0]["actual_close"] is None
    assert load_report(first_path)["cases"][0]["actual_close"] == [101.0] * 15
