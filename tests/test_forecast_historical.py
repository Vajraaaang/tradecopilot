"""Tiny fabricated fixtures; provider data must remain local."""

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest


def sample(directory: Path, rows: list[str], symbol: str = "AAPL", *, filename: str | None = None) -> None:
    path = directory / f"{symbol}_sample.zip"
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr(
            filename or f"{symbol}_1min_sample.csv",
            "timestamp,open,high,low,close,volume\n" + "\n".join(rows) + "\n",
        )
        archive.writestr("_readme_documentation.txt", "Synthetic test fixture")
    manifest = directory / "downloads.json"
    records = json.loads(manifest.read_text()) if manifest.exists() else []
    records = [record for record in records if record["symbol"] != symbol]
    records.append({
        "symbol": symbol,
        "url": f"https://frd001.s3.us-east-2.amazonaws.com/frd_sample_stock_{symbol}.zip",
        "retrieved_at": "2026-10-02T02:35:09.119873+00:00",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    })
    manifest.write_text(json.dumps(records))


def bar(at: str, close: str = "100") -> str:
    return f"{at},{close},{close},{close},{close},1"


def test_bar_close_is_available_at_end_with_causal_previous_close(tmp_path):
    from tradecopilot.forecast.historical import import_frd_samples

    sample(tmp_path, [
        bar("2026-09-17 09:30:00", "101"),
        bar("2026-09-16 15:59:00", "90"),
        bar("2026-09-17 15:59:00", "999"),
        bar("2026-09-17 04:00:00", "888"),
    ])
    observations, metadata = import_frd_samples(tmp_path, ["AAPL"])
    assert len(observations) == 2
    first, last = observations
    assert first.provider_timestamp == datetime(2026, 9, 17, 13, 31, tzinfo=UTC)
    assert first.receipt_timestamp == first.provider_timestamp
    assert first.age_seconds == 0 and first.last == Decimal("101")
    assert first.previous_close == last.previous_close == Decimal("90")
    assert first.provenance == "historical" and first.quality == "LIMITED"
    assert first.source == "firstratedata_1min_close"
    assert first.session_high is first.session_low is first.session_open is None
    assert metadata["adjustment_policy"] == "unspecified_in_free_sample"
    assert metadata["exclusions"] == {"outside_regular_session": 1, "previous_session_unavailable": 1}
    assert metadata["csv_files"][0]["rows"] == 4
    assert metadata["downloads"] == json.loads((tmp_path / "downloads.json").read_text())
    assert import_frd_samples(tmp_path, ["AAPL"]) == (observations, metadata)


def test_prior_session_gap_is_skipped_and_weekend_is_not_a_gap(tmp_path):
    from tradecopilot.forecast.historical import import_frd_samples

    sample(tmp_path, [
        bar("2026-09-18 15:59:00", "90"),
        bar("2026-09-21 09:30:00", "100"),
        bar("2026-09-23 09:30:00", "110"),
        bar("2026-09-24 09:30:00", "120"),
    ])
    observations, metadata = import_frd_samples(tmp_path, ["AAPL"])
    assert [row.provider_timestamp.day for row in observations] == [21, 24]
    assert [row.previous_close for row in observations] == [Decimal("90"), Decimal("110")]
    assert metadata["exclusions"]["previous_session_unavailable"] == 2


@pytest.mark.parametrize(("prior", "current", "expected"), [
    ("2026-03-06 15:59:00", "2026-03-09 09:30:00", datetime(2026, 3, 9, 13, 31, tzinfo=UTC)),
    ("2026-10-30 15:59:00", "2026-11-02 09:30:00", datetime(2026, 11, 2, 14, 31, tzinfo=UTC)),
])
def test_dst_changes_preserve_exchange_local_start_times(tmp_path, prior, current, expected):
    from tradecopilot.forecast.historical import import_frd_samples

    sample(tmp_path, [bar(prior, "90"), bar(current)])
    observations, _ = import_frd_samples(tmp_path, ["AAPL"])
    assert observations[0].provider_timestamp == expected


def test_early_close_and_exact_regular_session_edges(tmp_path):
    from tradecopilot.forecast.historical import import_frd_samples

    sample(tmp_path, [
        bar("2026-11-25 15:59:00", "90"),
        bar("2026-11-26 09:30:00"),  # Thanksgiving.
        bar("2026-11-27 09:29:00"),
        bar("2026-11-27 09:30:00"),
        bar("2026-11-27 12:59:00"),
        bar("2026-11-27 13:00:00"),
        bar("2026-11-30 09:30:00"),
        bar("2026-11-30 15:59:00"),
        bar("2026-11-30 16:00:00"),
    ])
    observations, metadata = import_frd_samples(tmp_path, ["AAPL"])
    assert len(observations) == 4
    assert observations[1].provider_timestamp == datetime(2026, 11, 27, 18, tzinfo=UTC)
    assert observations[-1].provider_timestamp == datetime(2026, 11, 30, 21, tzinfo=UTC)
    assert metadata["exclusions"]["outside_regular_session"] == 4


@pytest.mark.parametrize("field", ["sha256", "bytes", "retrieved_at"])
def test_invalid_download_manifest_fails_closed(tmp_path, field):
    from tradecopilot.forecast.historical import import_frd_samples

    sample(tmp_path, [bar("2026-09-16 15:59:00")])
    path = tmp_path / "downloads.json"
    records = json.loads(path.read_text())
    records[0][field] = {"sha256": "0" * 64, "bytes": 1, "retrieved_at": "2026-10-01"}[field]
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match=r"manifest|retriev"):
        import_frd_samples(tmp_path, ["AAPL"])


def test_archive_tampering_is_rejected(tmp_path):
    from tradecopilot.forecast.historical import import_frd_samples

    sample(tmp_path, [bar("2026-09-16 15:59:00")])
    with (tmp_path / "AAPL_sample.zip").open("ab") as file:
        file.write(b"tampered")
    with pytest.raises(ValueError, match="manifest"):
        import_frd_samples(tmp_path, ["AAPL"])


@pytest.mark.parametrize("row", [
    "2026-09-16 15:59:00,0,100,90,95,1",
    "2026-09-16 15:59:00,100,90,80,95,1",
    "2026-09-16 15:59:00,100,110,105,109,1",
    "2026-09-16 15:59:00,NaN,110,90,100,1",
    "2026-09-16 15:59:00,100,Infinity,90,100,1",
    "2026-09-16 15:59:00,100,110,90,100,-1",
    "2026-09-16 15:59:00,100,110,90,100,1.5",
    "2026-09-16 15:59:00,100,110,90,100",
    "2026-09-16 15:59:30,100,110,90,100,1",
    "2026-03-08 02:30:00,100,110,90,100,1",
    "2026-11-01 01:30:00,100,110,90,100,1",
])
def test_malformed_bars_are_rejected_even_outside_sessions(tmp_path, row):
    from tradecopilot.forecast.historical import import_frd_samples

    sample(tmp_path, [row])
    with pytest.raises(ValueError, match=r"bar|timestamp|volume|OHLC"):
        import_frd_samples(tmp_path, ["AAPL"])


def test_duplicate_conflicts_fail_and_identical_duplicates_are_counted(tmp_path):
    from tradecopilot.forecast.historical import import_frd_samples

    prior = bar("2026-09-16 15:59:00", "90")
    sample(tmp_path, [prior, prior, bar("2026-09-17 09:30:00")])
    observations, metadata = import_frd_samples(tmp_path, ["AAPL"])
    assert len(observations) == 1
    assert metadata["exclusions"]["duplicate_identical"] == 1
    sample(tmp_path, [prior, bar("2026-09-16 15:59:00", "91")])
    with pytest.raises(ValueError, match="conflict"):
        import_frd_samples(tmp_path, ["AAPL"])


def test_exact_csv_member_is_required_and_symbols_are_sorted(tmp_path):
    from tradecopilot.forecast.historical import import_frd_samples

    sample(tmp_path, [bar("2026-09-16 15:59:00")], filename="../AAPL_1min_sample.csv")
    with pytest.raises(ValueError, match=r"member|filename"):
        import_frd_samples(tmp_path, ["AAPL"])
    rows = [bar("2026-09-16 15:59:00"), bar("2026-09-17 09:30:00")]
    sample(tmp_path, rows)
    sample(tmp_path, rows, "MSFT")
    observations, metadata = import_frd_samples(tmp_path, ["MSFT", "AAPL"])
    assert [row.symbol for row in observations] == ["AAPL", "MSFT"]
    assert metadata["selected_symbols"] == ["AAPL", "MSFT"]
    for symbols in [[], ["AAPL", "AAPL"], ["../AAPL"], ["aapl"]]:
        with pytest.raises(ValueError, match="symbols"):
            import_frd_samples(tmp_path, symbols)


def test_input_size_limits_apply_before_parse(tmp_path, monkeypatch):
    from tradecopilot.forecast import historical

    sample(tmp_path, [bar("2026-09-16 15:59:00")])
    monkeypatch.setattr(historical, "MAX_MEMBER_BYTES", 10)
    with pytest.raises(ValueError, match="limit"):
        historical.import_frd_samples(tmp_path, ["AAPL"])


def test_feature_inputs_cannot_use_a_bar_before_its_end_or_same_day_final_close(tmp_path):
    from tradecopilot.forecast.contracts import ForecastConfig
    from tradecopilot.forecast.features import feature_result
    from tradecopilot.forecast.historical import import_frd_samples

    rows = [bar("2026-09-16 15:59:00", "90")]
    rows += [bar(f"2026-09-17 09:{minute}:00", str(100 + minute)) for minute in range(30, 37)]
    sample(tmp_path, rows)
    observations, _ = import_frd_samples(tmp_path, ["AAPL"])
    as_of = datetime(2026, 9, 17, 13, 36, tzinfo=UTC)
    config = ForecastConfig(symbols=("AAPL",))
    example, _ = feature_result(observations, as_of, "AAPL", config)
    assert example is not None and example.anchor_price == Decimal("135")
    ids = {row.observation_id: row for row in observations}
    assert all(ids[key].receipt_timestamp <= as_of for key in example.observation_ids)
    sample(tmp_path, [*rows, bar("2026-09-17 15:59:00", "999")])
    with_future, _ = import_frd_samples(tmp_path, ["AAPL"])
    assert feature_result(with_future, as_of, "AAPL", config)[0] == example


def test_missing_minutes_are_not_filled_and_close_proxy_uses_last_observed_minute(tmp_path):
    from tradecopilot.forecast.historical import import_frd_samples

    sample(tmp_path, [
        bar("2026-09-16 15:57:00", "90"),
        bar("2026-09-17 09:30:00"),
        bar("2026-09-17 09:32:00"),
    ])
    observations, _ = import_frd_samples(tmp_path, ["AAPL"])
    assert len(observations) == 2
    assert (observations[1].provider_timestamp - observations[0].provider_timestamp).total_seconds() == 120
    assert all(row.previous_close == Decimal("90") for row in observations)
