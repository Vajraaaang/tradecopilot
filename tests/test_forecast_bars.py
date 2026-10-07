"""OHLCV preservation checks use only small fabricated provider-format ZIPs."""

import hashlib
import json
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest


def sample(directory: Path, rows: list[str], symbol: str = "AAPL") -> None:
    archive_path = directory / f"{symbol}_sample.zip"
    with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
        archive.writestr(
            f"{symbol}_1min_sample.csv",
            "timestamp,open,high,low,close,volume\n" + "\n".join(rows) + "\n",
        )
    manifest = directory / "downloads.json"
    records = json.loads(manifest.read_text()) if manifest.exists() else []
    records = [record for record in records if record["symbol"] != symbol]
    records.append({
        "symbol": symbol,
        "url": f"https://frd001.s3.us-east-2.amazonaws.com/frd_sample_stock_{symbol}.zip",
        "retrieved_at": "2026-10-02T02:35:09.119873+00:00",
        "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "bytes": archive_path.stat().st_size,
    })
    manifest.write_text(json.dumps(records))


def row(at: str) -> str:
    return f"{at},100.0010,101.1234,99.9876,100.7654,123456789"


def make_bar(**changes):
    from tradecopilot.forecast.bars import HistoricalBar

    start = datetime(2026, 9, 17, 13, 30, tzinfo=UTC)
    return HistoricalBar(**{
        "symbol": "AAPL", "start_time": start,
        "end_time": start + timedelta(minutes=1), "available_at": start + timedelta(minutes=1),
        "opening": Decimal("100.0010"), "high": Decimal("101.1234"),
        "low": Decimal("99.9876"), "close": Decimal("100.7654"), "volume": Decimal("123456789"),
        **changes,
    })


def test_import_preserves_exact_ohlcv_and_first_session(tmp_path):
    from tradecopilot.forecast.bars import import_frd_bars

    sample(tmp_path, [row("2026-09-17 09:30:00"), row("2026-09-16 15:59:00")])
    bars, metadata = import_frd_bars(tmp_path, ["AAPL"])
    assert len(bars) == metadata["bar_count"] == 2
    assert bars[0].start_time == datetime(2026, 9, 16, 19, 59, tzinfo=UTC)
    assert bars[0].end_time == bars[0].available_at == datetime(2026, 9, 16, 20, tzinfo=UTC)
    assert (bars[1].opening, bars[1].high, bars[1].low, bars[1].close, bars[1].volume) == (
        Decimal("100.0010"), Decimal("101.1234"), Decimal("99.9876"),
        Decimal("100.7654"), Decimal("123456789"),
    )
    assert str(bars[1].opening) == "100.0010"
    assert bars[0].provenance == "historical" and bars[0].source == "firstratedata_1min_bar"
    assert metadata["schema_version"] == "historical-ohlcv-bars-v2"
    assert metadata["adjustment_policy"] == "unspecified_in_free_sample"
    assert metadata["downloads"] == json.loads((tmp_path / "downloads.json").read_text())
    assert metadata["retrieved_at"] == "2026-10-02T02:35:09.119873+00:00"
    assert "bar end" in metadata["availability_assumption"]
    assert metadata["csv_files"][0]["observed_sessions"] == ["2026-09-16", "2026-09-17"]
    assert metadata["first_bar_start_utc"] == bars[0].start_time.isoformat()
    assert metadata["last_bar_end_utc"] == bars[-1].end_time.isoformat()
    assert not list(tmp_path.glob("*.csv"))


def test_import_is_deterministic_and_keeps_missing_minutes_missing(tmp_path):
    from tradecopilot.forecast.bars import import_frd_bars

    sample(tmp_path, [row("2026-09-17 09:32:00"), row("2026-09-17 09:30:00")])
    sample(tmp_path, [row("2026-09-17 09:30:00")], "MSFT")
    bars, metadata = import_frd_bars(tmp_path, ["MSFT", "AAPL"])
    assert [bar.symbol for bar in bars] == ["AAPL", "MSFT", "AAPL"]
    assert bars[-1].start_time - bars[0].start_time == timedelta(minutes=2)
    assert import_frd_bars(tmp_path, ["AAPL", "MSFT"]) == (bars, metadata)


def test_import_obeys_dst_calendar_and_early_close(tmp_path):
    from tradecopilot.forecast.bars import import_frd_bars

    sample(tmp_path, [row(at) for at in [
        "2026-03-06 09:30:00", "2026-03-09 09:30:00", "2026-11-02 09:30:00",
        "2026-11-26 09:30:00", "2026-11-27 09:29:00", "2026-11-27 09:30:00",
        "2026-11-27 12:59:00", "2026-11-27 13:00:00", "2026-11-28 09:30:00",
    ]])
    bars, metadata = import_frd_bars(tmp_path, ["AAPL"])
    assert [bar.available_at for bar in bars] == [
        datetime(2026, 3, 6, 14, 31, tzinfo=UTC), datetime(2026, 3, 9, 13, 31, tzinfo=UTC),
        datetime(2026, 11, 2, 14, 31, tzinfo=UTC), datetime(2026, 11, 27, 14, 31, tzinfo=UTC),
        datetime(2026, 11, 27, 18, 0, tzinfo=UTC),
    ]
    assert metadata["exclusions"] == {"outside_regular_session": 4}


@pytest.mark.parametrize("changes", [
    {"opening": "0"}, {"high": "NaN"}, {"low": "Infinity"}, {"close": "-1"},
    {"high": "99"}, {"low": "101"}, {"volume": "-1"}, {"volume": "NaN"},
    {"start_time": datetime(2026, 9, 17, 13, 30)},
    {"end_time": datetime(2026, 9, 17, 13, 32, tzinfo=UTC)},
    {"available_at": datetime(2026, 9, 17, 13, 30, tzinfo=UTC)},
    {"symbol": "../AAPL"}, {"provenance": "market"}, {"source": "live_quote"},
])
def test_bar_rejects_invalid_prices_timing_or_origin(changes):
    with pytest.raises(ValueError):
        make_bar(**changes)


def test_bar_identity_binds_values_and_normalizes_timezone():
    bar = make_bar()
    offset = timezone(timedelta(hours=-4))
    same = make_bar(start_time=bar.start_time.astimezone(offset), end_time=bar.end_time.astimezone(offset))
    assert same.start_time.tzinfo == UTC and same.bar_id == bar.bar_id
    assert make_bar(volume=Decimal("1.5")).bar_id != bar.bar_id
    assert make_bar(close=Decimal("100.7")).bar_id != bar.bar_id
    with pytest.raises(ValueError):
        bar.close = Decimal("100")


@pytest.mark.parametrize("tamper", ["archive", "hash", "bytes", "retrieved_at", "url"])
def test_source_manifest_and_archive_integrity_are_required(tmp_path, tamper):
    from tradecopilot.forecast.bars import import_frd_bars

    sample(tmp_path, [row("2026-09-16 15:59:00")])
    if tamper == "archive":
        with (tmp_path / "AAPL_sample.zip").open("ab") as stream:
            stream.write(b"tampered")
    else:
        manifest = tmp_path / "downloads.json"
        records = json.loads(manifest.read_text())
        field = "sha256" if tamper == "hash" else tamper
        records[0][field] = {
            "hash": "0" * 64, "bytes": 1, "retrieved_at": "2026-10-01", "url": "https://example.com",
        }[tamper]
        manifest.write_text(json.dumps(records))
    with pytest.raises(ValueError):
        import_frd_bars(tmp_path, ["AAPL"])


def test_import_reuses_bounded_archive_validation(tmp_path, monkeypatch):
    from tradecopilot.forecast import historical
    from tradecopilot.forecast.bars import import_frd_bars

    sample(tmp_path, [row("2026-09-16 15:59:00")])
    monkeypatch.setattr(historical, "MAX_MEMBER_BYTES", 10)
    with pytest.raises(ValueError, match="limit"):
        import_frd_bars(tmp_path, ["AAPL"])


def test_dataset_exact_roundtrip_and_idempotent_immutable_write(tmp_path):
    from tradecopilot.forecast.bars import load_bar_dataset, write_bar_dataset

    bars = [make_bar(), make_bar(symbol="MSFT")]
    metadata = {"downloads": [{"url": "https://example.com", "sha256": "a" * 64}], "adjustment": "unspecified"}
    manifest_path = write_bar_dataset(tmp_path, bars[::-1], metadata)
    assert manifest_path == tmp_path / "manifest.json"
    loaded, manifest = load_bar_dataset(tmp_path)
    assert loaded == bars
    assert str(loaded[0].opening) == "100.0010"
    assert manifest["schema_version"] == "historical-ohlcv-bars-v2"
    assert manifest["bar_count"] == 2 and manifest["source_metadata"] == metadata
    assert manifest["content_hash"] == hashlib.sha256((tmp_path / "bars.jsonl").read_bytes()).hexdigest()
    original = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert write_bar_dataset(tmp_path, bars, metadata) == manifest_path
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == original
    with pytest.raises(ValueError, match="immutable"):
        write_bar_dataset(tmp_path, [make_bar(close=Decimal("100.7"))], metadata)
    with pytest.raises(ValueError, match="immutable"):
        write_bar_dataset(tmp_path, bars, {"different": True})
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == original


@pytest.mark.parametrize("tamper", ["data", "data_id", "content_hash", "bar_count", "schema", "metadata"])
def test_dataset_detects_data_and_manifest_tampering(tmp_path, tamper):
    from tradecopilot.forecast.bars import load_bar_dataset, write_bar_dataset

    write_bar_dataset(tmp_path, [make_bar()], {"source": "synthetic fixture"})
    if tamper == "data":
        with (tmp_path / "bars.jsonl").open("ab") as stream:
            stream.write(b"{}\n")
    else:
        manifest_path = tmp_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        field = {"schema": "schema_version", "metadata": "source_metadata"}.get(tamper, tamper)
        manifest[field] = {
            "data_id": "0" * 64, "content_hash": "0" * 64, "bar_count": 99,
            "schema": "historical-frd-sample-v1", "metadata": {"source": "changed"},
        }[tamper]
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_bar_dataset(tmp_path)


def test_writer_revalidates_model_copy_and_rejects_duplicate_minutes(tmp_path):
    from tradecopilot.forecast.bars import write_bar_dataset

    with pytest.raises(ValueError):
        write_bar_dataset(tmp_path, [make_bar().model_copy(update={"opening": Decimal("0")})], {})
    with pytest.raises(ValueError, match="duplicate"):
        write_bar_dataset(tmp_path, [make_bar(), make_bar()], {})
    assert not (tmp_path / "manifest.json").exists()


def rehash_dataset(directory: Path) -> None:
    from tradecopilot.forecast.contracts import content_hash

    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    payload = (directory / "bars.jsonl").read_bytes()
    manifest["bytes"] = len(payload)
    manifest["content_hash"] = hashlib.sha256(payload).hexdigest()
    manifest["data_id"] = content_hash({key: value for key, value in manifest.items() if key != "data_id"})
    manifest_path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("tamper", ["price", "timing", "count", "duplicate", "order", "extra_field"])
def test_loader_checks_rows_and_count_even_with_recomputed_hashes(tmp_path, tamper):
    from tradecopilot.forecast.bars import load_bar_dataset, write_bar_dataset

    write_bar_dataset(tmp_path, [make_bar(), make_bar(symbol="MSFT")], {})
    bars_path = tmp_path / "bars.jsonl"
    rows = [json.loads(line) for line in bars_path.read_text().splitlines()]
    if tamper == "price":
        rows[0]["opening"] = "0"
    elif tamper == "timing":
        rows[0]["available_at"] = "2026-09-17T13:30:00Z"
    elif tamper == "extra_field":
        rows[0]["bid"] = "100"
    elif tamper == "duplicate":
        rows[1] = rows[0]
    elif tamper == "order":
        rows.reverse()
    else:
        rows.pop()
    bars_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    rehash_dataset(tmp_path)
    with pytest.raises(ValueError):
        load_bar_dataset(tmp_path)


def test_partial_or_tampered_dataset_is_never_overwritten(tmp_path):
    from tradecopilot.forecast.bars import write_bar_dataset

    bars_path = tmp_path / "bars.jsonl"
    bars_path.write_text("partial\n")
    with pytest.raises(ValueError, match="invalid bar dataset"):
        write_bar_dataset(tmp_path, [make_bar()], {})
    assert bars_path.read_text() == "partial\n" and not (tmp_path / "manifest.json").exists()


def test_import_counts_identical_duplicates_and_rejects_conflicts(tmp_path):
    from tradecopilot.forecast.bars import import_frd_bars

    first = row("2026-09-17 09:30:00")
    sample(tmp_path, [first, first])
    bars, metadata = import_frd_bars(tmp_path, ["AAPL"])
    assert len(bars) == 1 and metadata["exclusions"] == {"duplicate_identical": 1}
    sample(tmp_path, [first, first.replace("123456789", "1")])
    with pytest.raises(ValueError, match="conflict"):
        import_frd_bars(tmp_path, ["AAPL"])


@pytest.mark.parametrize("symbols", [[], ["AAPL", "AAPL"], ["../AAPL"], ["aapl"]])
def test_import_rejects_invalid_symbol_selection(tmp_path, symbols):
    from tradecopilot.forecast.bars import import_frd_bars

    with pytest.raises(ValueError, match="symbols"):
        import_frd_bars(tmp_path, symbols)


def test_alpaca_source_is_preserved_without_changing_existing_default():
    from datetime import UTC, datetime, timedelta

    from tradecopilot.forecast.bars import HistoricalBar

    at = datetime(2026, 8, 3, 14, tzinfo=UTC)
    fields = dict(symbol="AAPL", start_time=at, end_time=at + timedelta(minutes=1),
                  available_at=at + timedelta(minutes=1), opening="100", high="101", low="99", close="100", volume="10")
    assert HistoricalBar(**fields).source == "firstratedata_1min_bar"
    assert HistoricalBar(**fields, source="alpaca_sip_1min_bar").source == "alpaca_sip_1min_bar"
