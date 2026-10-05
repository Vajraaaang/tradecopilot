from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from tradecopilot.forecast.bars import HistoricalBar, write_bar_dataset
from tradecopilot.forecast.contracts import LABELS, ForecastConfig, content_hash
from tradecopilot.forecast.ohlcv_features import OHLCV_FEATURE_NAMES
from tradecopilot.forecast.sessions import CALENDAR_VERSION, session_bounds

SYMBOLS = ("AAPL", "AMZN", "MSFT", "NVDA", "TSLA")
ROLE_DATES = {
    "TRAIN": date(2025, 1, 3),
    "TUNE": date(2025, 4, 2),
    "CAL": date(2025, 4, 16),
    "GATE": date(2025, 5, 1),
    "TEST": date(2025, 5, 15),
}
SOURCE_DAYS = tuple(
    sorted(
        {
            *ROLE_DATES.values(),
            date(2025, 1, 2),
            date(2025, 4, 1),
            date(2025, 4, 15),
            date(2025, 4, 30),
            date(2025, 5, 14),
        }
    )
)


def test_supervised_data_contract_is_available():
    assert importlib.util.find_spec("tradecopilot.forecast.supervised_data") is not None


def registration(root: Path) -> Path:
    config = ForecastConfig(symbols=SYMBOLS, max_source_age_seconds=0, max_outcome_delay_seconds=0)
    value = {
        "schema_version": "supervised-forecast-registration-v1",
        "symbols": list(SYMBOLS),
        "source_range": {
            "start": "2025-01-02",
            "end_exclusive": "2025-06-13",
            "provider": "Alpaca",
            "feed": "sip",
            "adjustment": "raw",
        },
        "source_parts": [["2025-01-02", "2025-04-02"], ["2025-04-02", "2025-06-13"]],
        "retired_ranges": [["2025-06-30", "2025-10-23"], ["2025-11-03", "2026-03-31"], ["2026-04-20", "2026-10-01"]],
        "forecast_config": config.model_dump(mode="json"),
        "splits": {
            "train": ["2025-01-03", "2025-04-01"],
            "tune": ["2025-04-02", "2025-04-15"],
            "calibration": ["2025-04-16", "2025-04-30"],
            "gate": ["2025-05-01", "2025-05-14"],
            "test": ["2025-05-15", "2025-06-12"],
        },
        "representation": {
            "version": "causal-supervised-sequence-v1",
            "sequence_length": 60,
            "warmup_minutes": 61,
            "anchor_stride_minutes": 5,
            "sequence_channels": [
                "open_bps_from_prewindow_close",
                "high_bps_from_prewindow_close",
                "low_bps_from_prewindow_close",
                "close_bps_from_prewindow_close",
                "log1p_volume",
                "one_minute_close_return_bps",
            ],
            "static_raw_features": 55,
            "static_neural_dimensions": 115,
            "static_cpu_dimensions": 60,
        },
    }
    value["registration_id"] = content_hash(value)
    path = root / "registration.json"
    path.write_text(json.dumps(value))
    return path


def fixture_bars() -> list[HistoricalBar]:
    rows = []
    for day in SOURCE_DAYS:
        bounds = session_bounds(day)
        assert bounds is not None
        for symbol in SYMBOLS:
            for minute in range(1, 91):
                end = bounds[0] + timedelta(minutes=minute)
                rows.append(
                    HistoricalBar(
                        symbol=symbol,
                        start_time=end - timedelta(minutes=1),
                        end_time=end,
                        available_at=end,
                        opening=Decimal("100"),
                        high=Decimal("101"),
                        low=Decimal("99"),
                        close=Decimal("100"),
                        volume=Decimal(100 + minute),
                        source="alpaca_sip_1min_bar",
                    )
                )
    return rows


def source(
    root: Path, bars: list[HistoricalBar], *, start: str = "2025-01-02", end: str = "2025-06-13", name: str = "source"
) -> Path:
    directory = root / name
    (directory / "raw").mkdir(parents=True)
    payload = json.dumps(
        {
            "bars": {
                symbol: [
                    {
                        "t": bar.start_time.isoformat(),
                        "o": str(bar.opening),
                        "h": str(bar.high),
                        "l": str(bar.low),
                        "c": str(bar.close),
                        "v": str(bar.volume),
                    }
                    for bar in bars
                    if bar.symbol == symbol
                ]
                for symbol in SYMBOLS
            },
            "next_page_token": None,
        }
    ).encode()
    (directory / "raw" / "page-0001.json").write_bytes(payload)
    metadata = {
        "schema_version": "historical-ohlcv-bars-v2",
        "provider": "Alpaca",
        "feed": "sip",
        "adjustment_policy": "raw",
        "provenance": "historical",
        "selected_symbols": list(SYMBOLS),
        "start_date": start,
        "end_date_exclusive": end,
        "downloads": [
            {
                "path": "raw/page-0001.json",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "retrieved_at": "2026-10-04T23:00:00+00:00",
            }
        ],
        "downloaded_bar_count": len(bars),
        "bar_count": len(bars),
        "exclusions": {},
        "calendar": "XNYS",
        "calendar_version": CALENDAR_VERSION,
        "interval_seconds": 60,
    }
    write_bar_dataset(directory / "bars", bars, metadata)
    return directory / "bars"


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    from tradecopilot.forecast.supervised_data import prepare_supervised

    root = tmp_path_factory.mktemp("supervised")
    reg = registration(root)
    bar_dir = source(root, fixture_bars())
    output = root / "prepared"
    path = prepare_supervised(bar_dir, reg, output)
    return root, reg, bar_dir, output, path


def test_frozen_roles_shapes_shared_identity_and_complete_catalog(prepared):
    from tradecopilot.forecast.supervised_data import load_stage

    _, _, _, directory, path = prepared
    manifest = json.loads(path.read_text())
    assert manifest["schema_version"] == "supervised-prepared-v1"
    assert manifest["planned_session_counts"] == {"TRAIN": 60, "TUNE": 10, "CAL": 10, "GATE": 10, "TEST": 20}
    assert manifest["prepared_data_id"] == content_hash({k: v for k, v in manifest.items() if k != "prepared_data_id"})
    stage, loaded = load_stage(directory, "TRAIN")
    assert loaded == manifest and stage.role == "TRAIN"
    assert stage.sequence.shape == (15, 60, 6) and stage.sequence.dtype == np.float32
    assert stage.static.shape == (15, 115) and stage.static.dtype == np.float32
    assert stage.targets.shape == (15,) and stage.targets.dtype == np.int64
    assert stage.case_ids == tuple(row.example_id for row in stage.examples)
    assert stage.case_ids == tuple(row.base_example_id for row in stage.features)
    assert len(set(stage.case_ids)) == 15 and set(stage.targets.tolist()) == {LABELS.index("FLAT")}
    for example in stage.examples:
        bounds = session_bounds(example.session_date)
        assert bounds is not None
        assert (example.as_of - bounds[0]).total_seconds() / 60 in {61, 66, 71}
        assert example.target_time == example.as_of + timedelta(minutes=15)
    catalog = [json.loads(line) for line in (directory / manifest["catalog"]["path"]).read_text().splitlines()]
    assert len(catalog) == sum(manifest["planned_anchor_counts"].values())
    assert {row["role"] for row in catalog} == {"TRAIN", "TUNE", "CAL", "GATE", "TEST"}
    assert sum(row["eligible"] for row in catalog) == sum(manifest["role_counts"].values())
    assert all("as_of" in row and "symbol" in row and "session_date" in row for row in catalog)
    cases = [
        json.loads(line) for line in (directory / manifest["roles"]["TRAIN"]["cases"]["path"]).read_text().splitlines()
    ]
    assert all(len(row["sequence_bar_ids"]) == 61 and len(row["sequence_input_hash"]) == 64 for row in cases)
    assert manifest["normalizer_id"] == content_hash(manifest["normalizers"])


def selection_seal(directory: Path, manifest: dict) -> Path:
    seal = {
        "stage": "selection_frozen",
        "registration_id": manifest["registration_id"],
        "prepared_data_id": manifest["prepared_data_id"],
        "config_id": manifest["config_id"],
        "source_data_id": manifest["source_data_id"],
        "class_order": list(LABELS),
        "normalizer_id": manifest["normalizer_id"],
        "selected_candidate_id": "prior",
        "selected_artifact_ids": ["1" * 64],
        "selected_checkpoint_ids": [],
        "weights_id": "4" * 64,
        "calibration_id": "3" * 64,
        "temperature": 1.0,
        "gate": {"gate_id": "2" * 64, "threshold": None},
        "cpu_reference_candidate_id": "prior",
        "cpu_reference_artifact_id": "1" * 64,
    }
    seal["selection_id"] = content_hash(seal)
    path = directory.parent / "selection.json"
    path.write_text(json.dumps(seal))
    return path


def test_test_requires_full_matching_selection_seal_before_label_decoding(prepared, monkeypatch):
    import tradecopilot.forecast.supervised_data as data
    from tradecopilot.forecast.supervised_data import load_stage

    _, _, _, directory, path = prepared
    reads = []
    original = data._verified_artifact

    def traced_artifact(directory, descriptor, expected_path):
        reads.append(expected_path)
        return original(directory, descriptor, expected_path)

    monkeypatch.setattr(data, "_verified_artifact", traced_artifact)
    with pytest.raises(ValueError, match="selection"):
        load_stage(directory, "TEST")
    assert not any(name.startswith("TEST/") for name in reads)
    manifest = json.loads(path.read_text())
    seal_path = selection_seal(directory, manifest)
    stage, _ = load_stage(directory, "TEST", selection_path=seal_path)
    assert len(stage.case_ids) == 15
    seal = json.loads(seal_path.read_text())
    seal.pop("gate")
    seal["selection_id"] = content_hash({k: v for k, v in seal.items() if k != "selection_id"})
    seal_path.write_text(json.dumps(seal))
    reads.clear()
    with pytest.raises(ValueError, match="selection"):
        load_stage(directory, "TEST", selection_path=seal_path)
    assert not any(name.startswith("TEST/") for name in reads)
    with pytest.raises(ValueError, match="role"):
        load_stage(directory, "train")


@pytest.mark.parametrize(
    "change,reason",
    [
        ("missing_past", "sequence_missing_minute"),
        ("missing_asof", "sequence_missing_minute"),
        ("missing_target", "target_missing_minute"),
        ("late_target", "target_not_available"),
        ("late_past", "sequence_not_available"),
    ],
)
def test_timestamp_exclusions_keep_full_planned_catalog(tmp_path, change, reason):
    from tradecopilot.forecast.supervised_data import load_stage, prepare_supervised

    reg = registration(tmp_path)
    bars = fixture_bars()
    bounds = session_bounds(ROLE_DATES["TRAIN"])
    assert bounds is not None
    minute = {"missing_past": 30, "missing_asof": 61, "missing_target": 76, "late_target": 76, "late_past": 30}[change]
    end = bounds[0] + timedelta(minutes=minute)
    selected = next(row for row in bars if row.symbol == "AAPL" and row.end_time == end)
    if change.startswith("missing"):
        bars.remove(selected)
    else:
        available_at = bounds[0] + timedelta(minutes=62) if change == "late_past" else end + timedelta(minutes=1)
        bars[bars.index(selected)] = selected.model_copy(update={"available_at": available_at})
    source_dir = source(tmp_path, bars)
    output = tmp_path / "prepared"
    manifest = json.loads(prepare_supervised(source_dir, reg, output).read_text())
    stage, _ = load_stage(output, "TRAIN")
    assert not any(row.symbol == "AAPL" and row.as_of == bounds[0] + timedelta(minutes=61) for row in stage.examples)
    catalog = [json.loads(line) for line in (output / manifest["catalog"]["path"]).read_text().splitlines()]
    excluded = [
        row
        for row in catalog
        if row["symbol"] == "AAPL" and row["as_of"] == (bounds[0] + timedelta(minutes=61)).isoformat()
    ]
    assert len(excluded) == 1 and excluded[0]["exclusion_reason"] == reason


@pytest.mark.parametrize(
    "price,label", [("100.1", "FLAT"), ("99.9", "FLAT"), ("100.10001", "UP"), ("99.89999", "DOWN")]
)
def test_exact_inclusive_label_boundaries(tmp_path, price, label):
    from tradecopilot.forecast.supervised_data import load_stage, prepare_supervised

    reg = registration(tmp_path)
    bars = fixture_bars()
    bounds = session_bounds(ROLE_DATES["TRAIN"])
    assert bounds is not None
    bars = [
        row.model_copy(update={"close": Decimal(price)})
        if row.symbol == "AAPL" and row.end_time == bounds[0] + timedelta(minutes=76)
        else row
        for row in bars
    ]
    output = tmp_path / "prepared"
    prepare_supervised(source(tmp_path, bars), reg, output)
    stage, _ = load_stage(output, "TRAIN")
    assert (
        next(
            row for row in stage.examples if row.symbol == "AAPL" and row.as_of == bounds[0] + timedelta(minutes=61)
        ).label
        == label
    )


def test_target_values_do_not_change_case_ids_or_train_normalization(prepared, tmp_path):
    from tradecopilot.forecast.supervised_data import load_stage, prepare_supervised

    _, _, _, original_dir, _ = prepared
    original, original_manifest = load_stage(original_dir, "TRAIN")
    reg = registration(tmp_path)
    bars = fixture_bars()
    for day in (ROLE_DATES["TRAIN"], ROLE_DATES["TEST"]):
        bounds = session_bounds(day)
        assert bounds is not None
        bars = [
            row.model_copy(update={"close": Decimal("100.5")})
            if row.symbol == "AAPL" and row.end_time == bounds[0] + timedelta(minutes=76)
            else row
            for row in bars
        ]
    directory = tmp_path / "prepared"
    prepare_supervised(source(tmp_path, bars), reg, directory)
    revised, manifest = load_stage(directory, "TRAIN")
    assert original.case_ids == revised.case_ids
    assert np.array_equal(original.sequence, revised.sequence) and np.array_equal(original.static, revised.static)
    assert original_manifest["normalizers"] == manifest["normalizers"]
    assert original_manifest["prepared_data_id"] != manifest["prepared_data_id"]
    assert original.targets.tolist() != revised.targets.tolist()


def test_earlier_sequence_input_mutation_changes_case_id(prepared, tmp_path):
    from tradecopilot.forecast.supervised_data import load_stage, prepare_supervised

    _, _, _, original_dir, _ = prepared
    original, _ = load_stage(original_dir, "TRAIN")
    reg = registration(tmp_path)
    bars = fixture_bars()
    bounds = session_bounds(ROLE_DATES["TRAIN"])
    assert bounds is not None
    bars = [
        row.model_copy(update={"close": Decimal("100.5")})
        if row.symbol == "AAPL" and row.end_time == bounds[0] + timedelta(minutes=1)
        else row
        for row in bars
    ]
    directory = tmp_path / "prepared"
    prepare_supervised(source(tmp_path, bars), reg, directory)
    revised, _ = load_stage(directory, "TRAIN")
    old = next(row for row in original.examples if row.symbol == "AAPL")
    new = next(row for row in revised.examples if row.symbol == "AAPL")
    assert old.features == new.features and old.example_id != new.example_id


def test_future_tune_mutation_does_not_change_train_inputs_or_scaler(prepared, tmp_path):
    from tradecopilot.forecast.supervised_data import load_stage, prepare_supervised

    _, _, _, original_dir, _ = prepared
    original, original_manifest = load_stage(original_dir, "TRAIN")
    reg = registration(tmp_path)
    bars = [
        row.model_copy(update={"volume": row.volume * 1000}) if row.start_time.date() >= ROLE_DATES["TUNE"] else row
        for row in fixture_bars()
    ]
    directory = tmp_path / "prepared"
    prepare_supervised(source(tmp_path, bars), reg, directory)
    revised, manifest = load_stage(directory, "TRAIN")
    assert original.case_ids == revised.case_ids
    assert original_manifest["normalizers"] == manifest["normalizers"]
    assert np.array_equal(original.sequence, revised.sequence) and np.array_equal(original.static, revised.static)
    binary = [index for index, name in enumerate(OHLCV_FEATURE_NAMES) if name.startswith("missing_window_")]
    assert np.isin(revised.static[:, binary], [0, 1]).all()
    assert np.isin(revised.static[:, 55:], [0, 1]).all()


def test_dst_session_phase_uses_exchange_open_and_rejects_retired_primer(tmp_path):
    from tradecopilot.forecast.supervised_data import prepare_supervised

    winter = session_bounds(date(2025, 3, 7))
    summer = session_bounds(date(2025, 3, 10))
    assert winter is not None and summer is not None
    assert winter[0].hour == 14 and summer[0].hour == 13
    reg = registration(tmp_path)
    retired_start = datetime(2025, 6, 30, 13, 30, tzinfo=UTC)
    retired = HistoricalBar(
        symbol="AAPL",
        start_time=retired_start,
        end_time=retired_start + timedelta(minutes=1),
        available_at=retired_start + timedelta(minutes=1),
        opening=Decimal(100),
        high=Decimal(100),
        low=Decimal(100),
        close=Decimal(100),
        volume=Decimal(1),
        source="alpaca_sip_1min_bar",
    )
    with pytest.raises(ValueError, match=r"retired|range"):
        prepare_supervised(source(tmp_path, [*fixture_bars(), retired]), reg, tmp_path / "prepared")


def test_immutable_preparation_and_artifact_tamper_detection(prepared, tmp_path):
    from tradecopilot.forecast.supervised_data import load_stage, prepare_supervised

    _, reg, bars, directory, path = prepared
    before = path.read_bytes()
    assert prepare_supervised(bars, reg, directory) == path and path.read_bytes() == before
    local_reg = registration(tmp_path)
    local_dir = tmp_path / "prepared"
    local_bars = source(tmp_path, fixture_bars())
    prepare_supervised(local_bars, local_reg, local_dir)
    manifest = json.loads((local_dir / "manifest.json").read_text())
    array = local_dir / manifest["roles"]["TRAIN"]["sequence"]["path"]
    with array.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match=r"hash|bytes"):
        load_stage(local_dir, "TRAIN")
    with pytest.raises(ValueError, match=r"immutable|hash|bytes"):
        prepare_supervised(local_bars, local_reg, local_dir)


def test_merge_preserves_verified_parent_manifests_raw_pages_and_counts(tmp_path):
    from tradecopilot.forecast.bars import load_bar_dataset
    from tradecopilot.forecast.supervised_data import merge_source_parts

    reg = registration(tmp_path)
    bars = fixture_bars()
    first = source(
        tmp_path, [row for row in bars if row.start_time.date() < date(2025, 4, 2)], end="2025-04-02", name="part-01"
    )
    second = source(
        tmp_path, [row for row in bars if row.start_time.date() >= date(2025, 4, 2)], start="2025-04-02", name="part-02"
    )
    output = tmp_path / "merged-bars"
    assert merge_source_parts([first, second], reg, output) == output / "manifest.json"
    merged, manifest = load_bar_dataset(output)
    assert len(merged) == len(bars)
    metadata = manifest["source_metadata"]
    assert metadata["source_merge_version"] == "supervised-source-merge-v1"
    assert len(metadata["parents"]) == 2
    assert all(parent["manifest"]["data_id"] == parent["data_id"] for parent in metadata["parents"])
    assert metadata["downloaded_bar_count"] == len(bars)
    assert len(metadata["downloads"]) == 2
    raw = tmp_path / metadata["downloads"][0]["path"]
    raw.write_bytes(b"tampered")
    with pytest.raises(ValueError, match=r"hash|bytes"):
        merge_source_parts([first, second], reg, tmp_path / "invalid")


def test_merge_rejects_gap_overlap_feed_mismatch_and_identical_duplicates(tmp_path):
    from tradecopilot.forecast.supervised_data import merge_source_parts

    reg = registration(tmp_path)
    bars = fixture_bars()
    first = source(
        tmp_path, [row for row in bars if row.start_time.date() < date(2025, 4, 2)], end="2025-04-02", name="part-01"
    )
    second = source(
        tmp_path, [row for row in bars if row.start_time.date() >= date(2025, 4, 2)], start="2025-04-03", name="part-02"
    )
    with pytest.raises(ValueError, match=r"range|contiguous|part"):
        merge_source_parts([first, second], reg, tmp_path / "gap")
    with pytest.raises(ValueError, match=r"duplicate|part|range"):
        merge_source_parts([first, first], reg, tmp_path / "duplicate")


def test_optional_method_registry_is_bound_before_preparation_and_rejects_retired_primer(tmp_path):
    from tradecopilot.forecast.supervised_data import load_stage, prepare_supervised

    reg = registration(tmp_path)
    registered = json.loads(reg.read_text())
    registry = {
        "schema_version": "supervised-method-registry-v1",
        "registration_id": registered["registration_id"],
        "before_learning_and_quality_scoring": True,
        "conservative_retired_ranges": [["2025-06-30", "2025-10-23"], ["2025-11-03", "2026-10-01"]],
    }
    registry["method_registry_id"] = content_hash(registry)
    registry_path = tmp_path / "method-registry.json"
    registry_path.write_text(json.dumps(registry))
    source_dir = source(tmp_path, fixture_bars())
    output = tmp_path / "prepared"
    manifest = json.loads(prepare_supervised(source_dir, reg, output).read_text())
    assert manifest["method_registry_id"] == registry["method_registry_id"]
    assert any(row["path"] == "method-registry.json" for row in manifest["source_inventory"])
    registry["conservative_retired_ranges"] = [["2025-01-02", "2025-01-03"]]
    registry["method_registry_id"] = content_hash({k: v for k, v in registry.items() if k != "method_registry_id"})
    registry_path.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="retired"):
        prepare_supervised(source_dir, reg, tmp_path / "invalid-prepared")
    with pytest.raises(ValueError):
        load_stage(output, "TRAIN")


def test_sequence_uses_fixed_predecessor_close_and_train_pooled_time_positions():
    from tradecopilot.forecast.supervised_data import _raw_sequence

    bounds = session_bounds(ROLE_DATES["TRAIN"])
    assert bounds is not None
    rows = []
    for index in range(61):
        close = Decimal(100 + index)
        end = bounds[0] + timedelta(minutes=index + 1)
        rows.append(
            HistoricalBar(
                symbol="AAPL",
                start_time=end - timedelta(minutes=1),
                end_time=end,
                available_at=end,
                opening=close - Decimal("0.5"),
                high=close + 1,
                low=close - 1,
                close=close,
                volume=Decimal(index),
                source="alpaca_sip_1min_bar",
            )
        )
    result = _raw_sequence(rows)
    assert result.shape == (60, 6)
    assert result[0].tolist() == pytest.approx([50, 200, 0, 100, np.log1p(1), 100])
    assert result[-1].tolist() == pytest.approx([5950, 6100, 5900, 6000, np.log1p(60), 10_000 * (160 / 159 - 1)])


def test_optional_static_missing_values_are_retained_and_masked(tmp_path):
    from tradecopilot.forecast.supervised_data import load_stage, prepare_supervised

    reg = registration(tmp_path)
    bars = [
        row.model_copy(update={"volume": Decimal(0)})
        if row.symbol == "AAPL" and row.start_time.date() == ROLE_DATES["TRAIN"]
        else row
        for row in fixture_bars()
    ]
    directory = tmp_path / "prepared"
    prepare_supervised(source(tmp_path, bars), reg, directory)
    stage, _ = load_stage(directory, "TRAIN")
    index = next(i for i, row in enumerate(stage.features) if row.symbol == "AAPL")
    feature_index = OHLCV_FEATURE_NAMES.index("relative_volume_prior_20m")
    assert stage.features[index].values["relative_volume_prior_20m"] is None
    assert stage.static[index, feature_index] == 0
    assert stage.static[index, 55 + feature_index] == 1
    assert np.isfinite(stage.static).all() and len(stage.case_ids) == 15


@pytest.mark.parametrize(
    "field,value",
    [("feed", "iex"), ("adjustment_policy", "all"), ("selected_symbols", ["AAPL", "AMZN", "MSFT", "NFLX", "TSLA"])],
)
def test_source_merge_rejects_parent_contract_changes(tmp_path, field, value):
    from tradecopilot.forecast.supervised_data import merge_source_parts

    reg = registration(tmp_path)
    rows = fixture_bars()
    first = source(
        tmp_path, [row for row in rows if row.start_time.date() < date(2025, 4, 2)], end="2025-04-02", name="part-01"
    )
    second = source(
        tmp_path, [row for row in rows if row.start_time.date() >= date(2025, 4, 2)], start="2025-04-02", name="part-02"
    )
    manifest_path = second / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_metadata"][field] = value
    manifest["data_id"] = content_hash({k: v for k, v in manifest.items() if k != "data_id"})
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r"provider|feed|adjustment|symbol"):
        merge_source_parts([first, second], reg, tmp_path / "invalid")


def test_catalog_must_match_stage_cohort_even_when_hashes_are_recomputed(prepared):
    from tradecopilot.forecast.supervised_data import load_stage

    _, _, _, directory, manifest_path = prepared
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    catalog_path = directory / manifest["catalog"]["path"]
    catalog_bytes = catalog_path.read_bytes()
    rows = [json.loads(line) for line in catalog_bytes.splitlines()]
    next(row for row in rows if row["eligible"])["case_id"] = "0" * 64
    changed = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    manifest["catalog"].update({"sha256": hashlib.sha256(changed).hexdigest(), "bytes": len(changed)})
    manifest["prepared_data_id"] = content_hash({k: v for k, v in manifest.items() if k != "prepared_data_id"})
    try:
        catalog_path.write_bytes(changed)
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="catalog"):
            load_stage(directory, "TRAIN")
    finally:
        catalog_path.write_bytes(catalog_bytes)
        manifest_path.write_bytes(manifest_bytes)


def test_raw_page_duplicate_is_rejected_even_without_a_duplicate_exclusion(tmp_path):
    from tradecopilot.forecast.supervised_data import merge_source_parts

    reg = registration(tmp_path)
    rows = fixture_bars()
    first = source(
        tmp_path, [row for row in rows if row.start_time.date() < date(2025, 4, 2)], end="2025-04-02", name="part-01"
    )
    second = source(
        tmp_path, [row for row in rows if row.start_time.date() >= date(2025, 4, 2)], start="2025-04-02", name="part-02"
    )
    page_path = first.parent / "raw/page-0001.json"
    page = json.loads(page_path.read_text())
    page["bars"]["AAPL"].append(page["bars"]["AAPL"][0])
    payload = json.dumps(page).encode()
    page_path.write_bytes(payload)
    manifest_path = first / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_metadata"]["downloads"][0].update(
        {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    )
    manifest["source_metadata"]["downloaded_bar_count"] += 1
    manifest["data_id"] = content_hash({k: v for k, v in manifest.items() if k != "data_id"})
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="duplicate"):
        merge_source_parts([first, second], reg, tmp_path / "invalid")


def test_catalog_phase_tracks_dst_with_fixed_utc_expected_times(prepared):
    _, _, _, directory, path = prepared
    manifest = json.loads(path.read_text())
    rows = [json.loads(line) for line in (directory / manifest["catalog"]["path"]).read_text().splitlines()]
    winter = [row for row in rows if row["symbol"] == "AAPL" and row["session_date"] == "2025-03-07"]
    summer = [row for row in rows if row["symbol"] == "AAPL" and row["session_date"] == "2025-03-10"]
    assert winter[0]["as_of"] == "2025-03-07T15:31:00+00:00"
    assert summer[0]["as_of"] == "2025-03-10T14:31:00+00:00"
    assert winter[-1]["target_time"] == "2025-03-07T20:56:00+00:00"
    assert summer[-1]["target_time"] == "2025-03-10T19:56:00+00:00"


@pytest.mark.parametrize("field", ["weights_id", "calibration_id"])
def test_test_requires_weight_and_calibration_component_seals(prepared, field):
    from tradecopilot.forecast.supervised_data import load_stage

    _, _, _, directory, path = prepared
    manifest = json.loads(path.read_text())
    seal_path = selection_seal(directory, manifest)
    seal = json.loads(seal_path.read_text())
    seal.pop(field)
    seal["selection_id"] = content_hash({k: v for k, v in seal.items() if k != "selection_id"})
    seal_path.write_text(json.dumps(seal))
    with pytest.raises(ValueError, match="selection"):
        load_stage(directory, "TEST", selection_path=seal_path)
