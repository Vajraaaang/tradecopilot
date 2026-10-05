"""Frozen, causal supervised cases with TRAIN-only preprocessing and sealed TEST access."""

from __future__ import annotations

import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from numpy.typing import NDArray

from tradecopilot.forecast.alpaca_history import MAX_BAR_ROWS, MAX_PAGES, MAX_RESPONSE_BYTES, _parse_bar, _parse_page
from tradecopilot.forecast.bars import BAR_DATA_VERSION, HistoricalBar, load_bar_dataset, write_bar_dataset
from tradecopilot.forecast.contracts import LABELS, ForecastConfig, ForecastExample, content_hash
from tradecopilot.forecast.features import feature_result, label_example
from tradecopilot.forecast.ohlcv_features import (
    OHLCV_FEATURE_NAMES,
    OHLCV_FEATURE_VERSION,
    OhlcvFeatureBuilder,
    OhlcvFeatureRecord,
)
from tradecopilot.forecast.selective_study import replay_observations
from tradecopilot.forecast.sessions import CALENDAR_VERSION, session_bounds

PREPARED_VERSION = "supervised-prepared-v1"
REPRESENTATION_VERSION = "causal-supervised-sequence-v1"
ROLES = ("TRAIN", "TUNE", "CAL", "GATE", "TEST")
SEQUENCE_CHANNELS = (
    "open_bps_from_prewindow_close",
    "high_bps_from_prewindow_close",
    "low_bps_from_prewindow_close",
    "close_bps_from_prewindow_close",
    "log1p_volume",
    "one_minute_close_return_bps",
)
_ROLE_SPLITS = dict(zip(ROLES, ("train", "tune", "calibration", "gate", "test"), strict=True))
_SESSION_COUNTS = dict(zip(ROLES, (60, 10, 10, 10, 20), strict=True))
_BINARY_INDICES = tuple(index for index, name in enumerate(OHLCV_FEATURE_NAMES) if name.startswith("missing_window_"))
_MINUTE = timedelta(minutes=1)
_NEW_YORK = ZoneInfo("America/New_York")
_MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class SupervisedStage:
    role: str
    sequence: NDArray[np.float32]
    static: NDArray[np.float32]
    examples: tuple[ForecastExample, ...]
    features: tuple[OhlcvFeatureRecord, ...]
    targets: NDArray[np.int64]
    case_ids: tuple[str, ...]


def _encode(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _read(path: Path, maximum: int = _MAX_ARTIFACT_BYTES) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("invalid or oversized private artifact")
    with path.open("rb") as stream:
        result = stream.read(maximum + 1)
    if len(result) > maximum:
        raise ValueError("private artifact exceeds size limit")
    return result


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(_read(path, _MAX_MANIFEST_BYTES))
    if not isinstance(value, dict):
        raise ValueError("manifest must be a JSON object")
    return value


def _contained(root: Path, path: Path) -> Path:
    root = root.resolve()
    try:
        relative = path.absolute().relative_to(root)
    except ValueError:
        raise ValueError("private source path must be below the registered experiment directory") from None
    current = root
    for component in relative.parts:
        if component in {"..", "."}:
            raise ValueError("invalid private artifact path")
        current /= component
        if current.is_symlink():
            raise ValueError("private artifact symlinks are forbidden")
    if not path.resolve().is_relative_to(root):
        raise ValueError("private artifact path escapes the experiment directory")
    return path


def _sessions(start: date, end: date) -> tuple[date, ...]:
    return tuple(
        day
        for index in range((end - start).days + 1)
        if session_bounds(day := start + timedelta(days=index)) is not None
    )


def _registration(path: Path) -> tuple[dict[str, Any], ForecastConfig, dict[str, tuple[date, ...]]]:
    return _validate_registration(_json(path))


def _validate_registration(value: dict[str, Any]) -> tuple[dict[str, Any], ForecastConfig, dict[str, tuple[date, ...]]]:
    if value.get("schema_version") != "supervised-forecast-registration-v1" or value.get(
        "registration_id"
    ) != content_hash({key: item for key, item in value.items() if key != "registration_id"}):
        raise ValueError("registration identity/schema mismatch")
    config = ForecastConfig.model_validate(value["forecast_config"])
    if (
        list(config.symbols) != value["symbols"]
        or len(config.symbols) != 5
        or config.horizon_minutes != 15
        or config.lookback_minutes != 5
        or config.anchor_seconds != 60
        or config.flat_threshold_bps != 10
        or config.max_source_age_seconds != 0
        or config.max_outcome_delay_seconds != 0
        or config.min_history_points != 6
        or config.max_history_gap_seconds != 60
        or config.seed != 42
    ):
        raise ValueError("registration must preserve the frozen exact-minute forecast configuration")
    representation = value["representation"]
    if any(
        representation.get(key) != expected
        for key, expected in {
            "version": REPRESENTATION_VERSION,
            "sequence_length": 60,
            "warmup_minutes": 61,
            "anchor_stride_minutes": 5,
            "sequence_channels": list(SEQUENCE_CHANNELS),
            "static_raw_features": 55,
            "static_neural_dimensions": 115,
            "static_cpu_dimensions": 60,
        }.items()
    ):
        raise ValueError("registration representation mismatch")
    source = value["source_range"]
    if (source.get("provider"), source.get("feed"), source.get("adjustment")) != ("Alpaca", "sip", "raw"):
        raise ValueError("registration source must be Alpaca SIP/raw")
    start, end = date.fromisoformat(source["start"]), date.fromisoformat(source["end_exclusive"])
    if not start < end or (end - start).days > 366:
        raise ValueError("invalid registered source range")
    parts = [(date.fromisoformat(a), date.fromisoformat(b)) for a, b in value["source_parts"]]
    if (
        len(parts) != 2
        or parts[0][0] != start
        or parts[-1][1] != end
        or parts[0][1] != parts[1][0]
        or any(a >= b for a, b in parts)
    ):
        raise ValueError("registered source parts must be contiguous and cover the source range")
    retired = [(date.fromisoformat(a), date.fromisoformat(b)) for a, b in value["retired_ranges"]]
    if any(a >= b or max(a, start) < min(b, end) for a, b in retired):
        raise ValueError("registered source range includes retired dates, including its primer")
    roles: dict[str, tuple[date, ...]] = {}
    all_dates: list[date] = []
    for role, key in _ROLE_SPLITS.items():
        first, last = (date.fromisoformat(item) for item in value["splits"][key])
        dates = _sessions(first, last)
        if len(dates) != _SESSION_COUNTS[role] or not start < first <= last < end:
            raise ValueError("registration must use 60/10/10/10/20 chronological sessions")
        roles[role] = dates
        all_dates.extend(dates)
    if tuple(all_dates) != _sessions(start, end - timedelta(days=1))[1:] or len(set(all_dates)) != 110:
        raise ValueError("split dates must cover all 110 label sessions following the primer")
    return value, config, roles


def _inventory(root: Path, path: Path, kind: str) -> dict[str, Any]:
    payload = _read(_contained(root, path))
    return {
        "path": path.absolute().relative_to(root.resolve()).as_posix(),
        "kind": kind,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _method_registry(root: Path, registration: Mapping[str, Any]) -> dict[str, Any] | None:
    path = root / "method-registry.json"
    if not path.exists():
        return None
    value = _json(_contained(root, path))
    if (
        value.get("schema_version") != "supervised-method-registry-v1"
        or value.get("registration_id") != registration["registration_id"]
        or value.get("before_learning_and_quality_scoring") is not True
        or value.get("method_registry_id")
        != content_hash({key: item for key, item in value.items() if key != "method_registry_id"})
    ):
        raise ValueError("method registry identity/registration mismatch")
    retired = [(date.fromisoformat(a), date.fromisoformat(b)) for a, b in value["conservative_retired_ranges"]]
    start = date.fromisoformat(registration["source_range"]["start"])
    end = date.fromisoformat(registration["source_range"]["end_exclusive"])
    if not retired or any(a >= b or max(a, start) < min(b, end) for a, b in retired):
        raise ValueError("method registry retired dates overlap source bounds, including primer")
    for a, b in registration["retired_ranges"]:
        if not any(first <= date.fromisoformat(a) and date.fromisoformat(b) <= last for first, last in retired):
            raise ValueError("method registry must retain all previously retired date ranges")
    return value


def _validate_source_rows(
    bars: Sequence[HistoricalBar], metadata: Mapping[str, Any], registration: Mapping[str, Any], bounds: tuple[str, str]
) -> None:
    if (
        (metadata.get("provider"), metadata.get("feed"), metadata.get("adjustment_policy")) != ("Alpaca", "sip", "raw")
        or metadata.get("selected_symbols") != sorted(registration["symbols"])
        or (metadata.get("start_date"), metadata.get("end_date_exclusive")) != bounds
        or metadata.get("bar_count") != len(bars)
        or metadata.get("calendar") != "XNYS"
        or metadata.get("calendar_version") != CALENDAR_VERSION
        or metadata.get("interval_seconds") != 60
    ):
        raise ValueError("source provider/feed/adjustment/symbol/range/count/calendar mismatch")
    retired = [(date.fromisoformat(a), date.fromisoformat(b)) for a, b in registration["retired_ranges"]]
    for bar in bars:
        day = bar.start_time.astimezone(_NEW_YORK).date()
        if any(a <= day < b for a, b in retired):
            raise ValueError("source contains a retired date, including primer data")
        session = session_bounds(day)
        if (
            not bounds[0] <= day.isoformat() < bounds[1]
            or bar.symbol not in registration["symbols"]
            or bar.source != "alpaca_sip_1min_bar"
            or bar.start_time.second
            or bar.start_time.microsecond
            or session is None
            or bar.start_time < session[0]
            or bar.end_time > session[1]
        ):
            raise ValueError("source bar violates registered range/symbol/minute/session bounds")


def _verify_raw_pages(
    directory: Path, bars: Sequence[HistoricalBar], manifest: dict[str, Any], root: Path, bounds: tuple[str, str]
) -> tuple[list[dict[str, Any]], set[tuple[str, datetime]]]:
    metadata = manifest["source_metadata"]
    downloads = metadata.get("downloads")
    if not isinstance(downloads, list) or not 1 <= len(downloads) <= MAX_PAGES:
        raise ValueError("source raw-page manifest is missing or exceeds the parent page cap")
    if any("duplicate" in key and count for key, count in metadata.get("exclusions", {}).items()):
        raise ValueError("duplicate raw source records are forbidden")
    start, end = (datetime.combine(date.fromisoformat(item), time.min, tzinfo=_NEW_YORK) for item in bounds)
    seen: set[tuple[str, datetime]] = set()
    regular: dict[tuple[str, datetime], dict[str, Any]] = {}
    inventory = [
        _inventory(root, directory / "manifest.json", "parent_bar_manifest"),
        _inventory(root, directory / "bars.jsonl", "parent_bars"),
    ]
    symbols = tuple(metadata["selected_symbols"])
    for record in downloads:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid parent raw-page path")
        path = _contained(root, directory.parent / relative)
        payload = _read(path, MAX_RESPONSE_BYTES)
        if len(payload) != record["bytes"] or hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError("source raw page hash/bytes mismatch")
        inventory.append(_inventory(root, path, "raw_page"))
        rows, _ = _parse_page(payload, symbols)
        for symbol, values in rows.items():
            for row in values:
                bar = _parse_bar(symbol, row, start, end)
                key = (symbol, bar.start_time)
                if key in seen:
                    raise ValueError("duplicate raw source record, even if identical")
                seen.add(key)
                session = session_bounds(bar.start_time.astimezone(_NEW_YORK).date())
                if session is not None and session[0] <= bar.start_time and bar.end_time <= session[1]:
                    regular[key] = bar.model_dump(mode="json", exclude={"available_at"})
    if (
        len(seen) > MAX_BAR_ROWS
        or len(seen) != metadata.get("downloaded_bar_count")
        or len(regular) != len(bars)
        or metadata.get("exclusions", {}).get("outside_regular_session", 0) != len(seen) - len(regular)
    ):
        raise ValueError("parent source row count/exclusion mismatch or raw-row cap exceeded")
    if any(
        regular.get((bar.symbol, bar.start_time)) != bar.model_dump(mode="json", exclude={"available_at"})
        for bar in bars
    ):
        raise ValueError("source bars differ from verified parent raw-page data")
    return inventory, seen


def merge_source_parts(part_bars_dirs: list[Path], registration: Path, output_dir: Path) -> Path:
    """Verify bounded parents and their private raw pages before an immutable atomic merge."""
    registered, _, _ = _registration(registration)
    root = registration.parent.resolve()
    _contained(root, output_dir)
    expected = [tuple(part) for part in registered["source_parts"]]
    if len(part_bars_dirs) != len(expected) or len({path.resolve() for path in part_bars_dirs}) != len(expected):
        raise ValueError("source part count or duplicate part mismatch")
    merged: list[HistoricalBar] = []
    raw_keys: set[tuple[str, datetime]] = set()
    parents: list[dict[str, Any]] = []
    downloads: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    for directory, bounds in zip(part_bars_dirs, expected, strict=True):
        _contained(root, directory)
        bars, manifest = load_bar_dataset(directory)
        metadata = manifest["source_metadata"]
        _validate_source_rows(bars, metadata, registered, bounds)
        _, keys = _verify_raw_pages(directory, bars, manifest, root, bounds)
        if raw_keys & keys:
            raise ValueError("duplicate source records across parents")
        raw_keys.update(keys)
        parents.append(
            {
                "data_id": manifest["data_id"],
                "manifest": manifest,
                "manifest_path": (directory / "manifest.json").absolute().relative_to(root).as_posix(),
                "bars_path": (directory / "bars.jsonl").absolute().relative_to(root).as_posix(),
                "bounds": list(bounds),
                "bar_count": len(bars),
                "downloaded_bar_count": metadata["downloaded_bar_count"],
                "exclusions": metadata.get("exclusions", {}),
            }
        )
        for record in metadata["downloads"]:
            downloads.append(
                {
                    **record,
                    "path": (directory.parent / record["path"]).absolute().relative_to(root).as_posix(),
                    "parent_data_id": manifest["data_id"],
                }
            )
        exclusions.update(metadata.get("exclusions", {}))
        merged.extend(bars)
    first_metadata = parents[0]["manifest"]["source_metadata"]
    metadata = {
        **first_metadata,
        "source_merge_version": "supervised-source-merge-v1",
        "registration_id": registered["registration_id"],
        "parents": parents,
        "downloads": downloads,
        "start_date": registered["source_range"]["start"],
        "end_date_exclusive": registered["source_range"]["end_exclusive"],
        "downloaded_bar_count": len(raw_keys),
        "bar_count": len(merged),
        "exclusions": dict(sorted(exclusions.items())),
        "source_params": None,
        "observed_sessions": sorted({bar.start_time.astimezone(_NEW_YORK).date().isoformat() for bar in merged}),
        "first_bar_start_utc": min(bar.start_time for bar in merged).isoformat() if merged else None,
        "last_bar_end_utc": max(bar.end_time for bar in merged).isoformat() if merged else None,
        "retrieved_at": max(record["retrieved_at"] for record in downloads),
    }
    if output_dir.exists():
        return write_bar_dataset(output_dir, merged, metadata)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".supervised-source-", dir=output_dir.parent) as temporary:
        stage = Path(temporary) / "bars"
        write_bar_dataset(stage, merged, metadata)
        stage.chmod(0o700)
        for path in stage.iterdir():
            path.chmod(0o600)
        if output_dir.exists():
            raise ValueError("source merge output is immutable")
        stage.rename(output_dir)
    return output_dir / "manifest.json"


def _validated_source(
    directory: Path, registration: dict[str, Any], root: Path
) -> tuple[list[HistoricalBar], dict[str, Any], list[dict[str, Any]]]:
    _contained(root, directory)
    bars, manifest = load_bar_dataset(directory)
    metadata = manifest["source_metadata"]
    bounds = (registration["source_range"]["start"], registration["source_range"]["end_exclusive"])
    _validate_source_rows(bars, metadata, registration, bounds)
    inventory: list[dict[str, Any]] = []
    if metadata.get("source_merge_version") is None:
        inventory, _ = _verify_raw_pages(directory, bars, manifest, root, bounds)
    else:
        if (
            metadata["source_merge_version"] != "supervised-source-merge-v1"
            or metadata.get("registration_id") != registration["registration_id"]
            or len(metadata.get("parents", [])) != len(registration["source_parts"])
        ):
            raise ValueError("merged source registration/parents mismatch")
        expected_rows: list[HistoricalBar] = []
        all_keys: set[tuple[str, datetime]] = set()
        for parent, part in zip(metadata["parents"], registration["source_parts"], strict=True):
            parent_directory = _contained(root, root / parent["manifest_path"]).parent
            parent_rows, parent_manifest = load_bar_dataset(parent_directory)
            if parent_manifest != parent["manifest"] or parent_manifest["data_id"] != parent["data_id"]:
                raise ValueError("merged parent manifest identity mismatch")
            _validate_source_rows(parent_rows, parent_manifest["source_metadata"], registration, tuple(part))
            parent_inventory, keys = _verify_raw_pages(
                parent_directory, parent_rows, parent_manifest, root, tuple(part)
            )
            if keys & all_keys:
                raise ValueError("duplicate records across merged parents")
            all_keys.update(keys)
            inventory.extend(parent_inventory)
            expected_rows.extend(parent_rows)
        expected_rows.sort(key=lambda bar: (bar.available_at, bar.symbol, bar.start_time))
        if expected_rows != bars or metadata.get("downloaded_bar_count") != len(all_keys):
            raise ValueError("merged source does not match its frozen parent rows")
        inventory.extend(
            [
                _inventory(root, directory / "manifest.json", "merged_bar_manifest"),
                _inventory(root, directory / "bars.jsonl", "merged_bars"),
            ]
        )
    return bars, manifest, sorted(inventory, key=lambda row: str(row["path"]))


def _sequence_hash(ids: Sequence[str]) -> str:
    return content_hash({"representation_version": REPRESENTATION_VERSION, "sequence_bar_ids": list(ids)})


def _static_hash(record: OhlcvFeatureRecord) -> str:
    return content_hash(
        {"feature_version": record.feature_version, "values": record.values, "input_bars_hash": record.input_bars_hash}
    )


def _raw_sequence(bars: Sequence[HistoricalBar]) -> NDArray[np.float64]:
    reference = float(bars[0].close)
    result = np.asarray(
        [
            [10_000 * (float(value) / reference - 1) for value in (bar.opening, bar.high, bar.low, bar.close)]
            + [math.log1p(float(bar.volume)), 10_000 * (float(bar.close) / float(previous.close) - 1)]
            for previous, bar in pairwise(bars)
        ],
        dtype=np.float64,
    )
    if result.shape != (60, 6) or not np.isfinite(result).all():
        raise ValueError("nonfinite sequence inputs cannot be silently excluded")
    return result


def _raw_static(records: Sequence[OhlcvFeatureRecord]) -> NDArray[np.float64]:
    return np.asarray(
        [[np.nan if row.values[name] is None else row.values[name] for name in OHLCV_FEATURE_NAMES] for row in records],
        dtype=np.float64,
    ).reshape(len(records), 55)


def _normalizers(
    raw_sequence: NDArray[np.float64], records: Sequence[OhlcvFeatureRecord], cases: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    if not len(cases):
        raise ValueError("TRAIN has no eligible cases for preprocessing")
    sequence_mean = raw_sequence.mean(axis=(0, 1))
    sequence_scale = raw_sequence.std(axis=(0, 1))
    sequence_scale[sequence_scale == 0] = 1
    matrix = _raw_static(records)
    available = np.isfinite(matrix)
    counts = available.sum(axis=0)
    static_mean = np.where(available, matrix, 0).sum(axis=0) / np.maximum(counts, 1)
    imputed = np.where(available, matrix, static_mean)
    static_scale = imputed.std(axis=0)
    static_scale[static_scale == 0] = 1
    static_mean[list(_BINARY_INDICES)] = 0
    static_scale[list(_BINARY_INDICES)] = 1
    return {
        "fit_role": "TRAIN",
        "fit_case_count": len(cases),
        "fit_time_positions_count": len(cases) * 60,
        "sequence": {"mean": sequence_mean.tolist(), "scale": sequence_scale.tolist()},
        "static": {
            "mean": static_mean.tolist(),
            "scale": static_scale.tolist(),
            "binary_indices": list(_BINARY_INDICES),
        },
        "train_case_ids_hash": content_hash([row["case_id"] for row in cases]),
        "train_raw_inputs_hash": content_hash(
            [
                {key: row[key] for key in ("case_id", "sequence_input_hash", "static_input_hash", "feature_id")}
                for row in cases
            ]
        ),
    }


def _transform(
    raw_sequence: NDArray[np.float64],
    records: Sequence[OhlcvFeatureRecord],
    normalizers: Mapping[str, Any],
    symbols: tuple[str, ...],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    seq = (raw_sequence - np.asarray(normalizers["sequence"]["mean"])) / np.asarray(normalizers["sequence"]["scale"])
    raw_static = _raw_static(records)
    missing = np.isnan(raw_static)
    mean, scale = np.asarray(normalizers["static"]["mean"]), np.asarray(normalizers["static"]["scale"])
    static = (np.where(missing, mean, raw_static) - mean) / scale
    onehot = np.asarray(
        [[float(record.symbol == symbol) for symbol in symbols] for record in records], dtype=np.float64
    ).reshape(len(records), 5)
    matrix = np.concatenate([static, missing.astype(np.float64), onehot], axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        sequence32, static32 = seq.astype(np.float32), matrix.astype(np.float32)
    if not np.isfinite(sequence32).all() or not np.isfinite(static32).all():
        raise ValueError("nonfinite normalized inputs cannot be silently excluded")
    return sequence32, static32


def _npy(array: NDArray[Any]) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _descriptor(path: str, payload: bytes) -> dict[str, Any]:
    if len(payload) > _MAX_ARTIFACT_BYTES:
        raise ValueError("prepared artifact exceeds size cap")
    return {"path": path, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def prepare_supervised(bars_dir: Path, registration_path: Path, output_dir: Path) -> Path:
    """Construct the entire registered timestamp catalog; no models or quality scores are fitted."""
    registration, config, dates = _registration(registration_path)
    root = registration_path.parent.resolve()
    _contained(root, output_dir)
    method_registry = _method_registry(root, registration)
    bars, source_manifest, inventory = _validated_source(bars_dir, registration, root)
    if method_registry is not None:
        inventory.append(_inventory(root, root / "method-registry.json", "method_registry"))
    by_minute = {(bar.symbol, bar.end_time): bar for bar in bars}
    observations = defaultdict(list)
    for row in replay_observations(bars):
        observations[(row.symbol, row.provider_timestamp.date())].append(row)
    builder = OhlcvFeatureBuilder(bars)
    examples: dict[str, list[ForecastExample]] = {role: [] for role in ROLES}
    features: dict[str, list[OhlcvFeatureRecord]] = {role: [] for role in ROLES}
    sequences: dict[str, list[NDArray[np.float64]]] = {role: [] for role in ROLES}
    cases: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLES}
    catalog: list[dict[str, Any]] = []
    planned: Counter[str] = Counter()
    exclusion_counts: dict[str, Counter[str]] = {role: Counter() for role in ROLES}
    for role in ROLES:
        for day in dates[role]:
            bounds = session_bounds(day)
            assert bounds is not None
            as_of = bounds[0] + timedelta(minutes=61)
            while as_of + timedelta(minutes=15) <= bounds[1]:
                target_time = as_of + timedelta(minutes=15)
                for symbol in config.symbols:
                    planned[role] += 1
                    entry: dict[str, Any] = {
                        "role": role,
                        "symbol": symbol,
                        "session_date": day.isoformat(),
                        "as_of": as_of.isoformat(),
                        "target_time": target_time.isoformat(),
                        "eligible": False,
                        "exclusion_reason": None,
                    }
                    sequence_times = [as_of - timedelta(minutes=60 - index) for index in range(61)]
                    window = [by_minute.get((symbol, timestamp)) for timestamp in sequence_times]
                    missing = [
                        timestamp.isoformat()
                        for timestamp, bar in zip(sequence_times, window, strict=True)
                        if bar is None
                    ]
                    target = by_minute.get((symbol, target_time))
                    reason = None
                    if missing:
                        reason = "sequence_missing_minute"
                        entry["missing_sequence_end_times"] = missing
                    elif any(bar is not None and bar.available_at > as_of for bar in window):
                        reason = "sequence_not_available"
                        entry["late_sequence_end_times"] = [
                            bar.end_time.isoformat() for bar in window if bar is not None and bar.available_at > as_of
                        ]
                    elif target is None:
                        reason = "target_missing_minute"
                    elif target.available_at > target_time:
                        reason = "target_not_available"
                    else:
                        visible = observations[(symbol, day)]
                        input_example, reason = feature_result(visible, as_of, symbol, config)
                        if input_example is not None:
                            # Keep every outcome field absent at the feature-builder boundary.
                            input_example = input_example.model_copy(
                                update={
                                    "label": None,
                                    "target_price": None,
                                    "label_observed_at": None,
                                    "target_return_bps": None,
                                    "exclusion_reason": None,
                                }
                            )
                            record = builder.build(input_example)
                            available_window = [bar for bar in window if bar is not None]
                            bar_ids = [bar.bar_id for bar in available_window]
                            sequence_hash, static_hash = _sequence_hash(bar_ids), _static_hash(record)
                            price_example_id = input_example.example_id
                            input_example = input_example.model_copy(
                                update={
                                    "observation_ids": (
                                        *input_example.observation_ids,
                                        "supervised-sequence:" + sequence_hash,
                                        "supervised-static:" + static_hash,
                                    )
                                }
                            )
                            record = record.model_copy(update={"base_example_id": input_example.example_id})
                            outcome = [row for row in visible if row.provider_timestamp == target_time]
                            example = label_example(outcome, input_example, config)
                            if example.label is None:
                                raise ValueError("timestamp-eligible exact target unexpectedly has no label")
                            examples[role].append(example)
                            features[role].append(record)
                            sequences[role].append(_raw_sequence(available_window))
                            cases[role].append(
                                {
                                    "case_id": example.example_id,
                                    "price_example_id": price_example_id,
                                    "sequence_bar_ids": bar_ids,
                                    "sequence_input_hash": sequence_hash,
                                    "static_input_hash": static_hash,
                                    "feature_id": record.feature_id,
                                    "source_data_id": source_manifest["data_id"],
                                }
                            )
                            entry.update({"eligible": True, "case_id": example.example_id})
                    if not entry["eligible"]:
                        entry["exclusion_reason"] = reason or "previous_session_unavailable"
                        exclusion_counts[role][entry["exclusion_reason"]] += 1
                    catalog.append(entry)
                as_of += timedelta(minutes=5)
    raw_sequences = {
        role: np.asarray(sequences[role], dtype=np.float64).reshape(len(sequences[role]), 60, 6) for role in ROLES
    }
    normalizers = _normalizers(raw_sequences["TRAIN"], features["TRAIN"], cases["TRAIN"])
    artifacts: dict[str, bytes] = {}
    role_artifacts: dict[str, dict[str, Any]] = {}
    for role in ROLES:
        sequence, static = _transform(raw_sequences[role], features[role], normalizers, config.symbols)
        labels = np.asarray([LABELS.index(str(row.label)) for row in examples[role]], dtype=np.int64)
        role_artifacts[role] = {
            "count": len(labels),
            "case_ids_hash": content_hash([row["case_id"] for row in cases[role]]),
        }
        for key, payload in {
            "sequence": _npy(sequence),
            "static": _npy(static),
            "targets": _npy(labels),
            "examples": b"".join(_encode(row.model_dump(mode="json")) + b"\n" for row in examples[role]),
            "features": b"".join(_encode(row.model_dump(mode="json")) + b"\n" for row in features[role]),
            "cases": b"".join(_encode(row) + b"\n" for row in cases[role]),
        }.items():
            filename = f"{role}/{key}." + ("npy" if key in {"sequence", "static", "targets"} else "jsonl")
            artifacts[filename] = payload
            role_artifacts[role][key] = _descriptor(filename, payload)
    catalog_payload = b"".join(_encode(entry) + b"\n" for entry in catalog)
    artifacts["catalog.jsonl"] = catalog_payload
    catalog_counts: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLES}
    grouped_catalog: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in catalog:
        grouped_catalog[(entry["role"], entry["session_date"], entry["symbol"])].append(entry)
    for (role, day_string, symbol), entries in sorted(grouped_catalog.items()):
        eligible = sum(entry["eligible"] for entry in entries)
        reasons = Counter(entry["exclusion_reason"] for entry in entries if not entry["eligible"])
        catalog_counts[role].append(
            {
                "session_date": day_string,
                "symbol": symbol,
                "planned": len(entries),
                "eligible": eligible,
                "excluded": len(entries) - eligible,
                "exclusion_reasons": dict(sorted(reasons.items())),
            }
        )
    manifest = {
        "schema_version": PREPARED_VERSION,
        "representation_version": REPRESENTATION_VERSION,
        "registration": registration,
        "registration_id": registration["registration_id"],
        "method_registry": method_registry,
        "method_registry_id": method_registry["method_registry_id"] if method_registry is not None else None,
        "source_data_id": source_manifest["data_id"],
        "source_manifest": source_manifest,
        "source_inventory": inventory,
        "experiment_root": root.as_posix(),
        "forecast_config": config.model_dump(mode="json"),
        "config_id": config.config_id,
        "calendar": "XNYS",
        "calendar_version": CALENDAR_VERSION,
        "class_order": list(LABELS),
        "sequence_channels": list(SEQUENCE_CHANNELS),
        "sequence_length": 60,
        "static_feature_names": list(OHLCV_FEATURE_NAMES),
        "feature_version": OHLCV_FEATURE_VERSION,
        "static_neural_feature_names": [
            *OHLCV_FEATURE_NAMES,
            *("missing_" + name for name in OHLCV_FEATURE_NAMES),
            *("symbol_" + symbol for symbol in config.symbols),
        ],
        "static_cpu_feature_names": [*OHLCV_FEATURE_NAMES, *("symbol_" + symbol for symbol in config.symbols)],
        "normalizers": normalizers,
        "normalizer_id": content_hash(normalizers),
        "planned_session_counts": _SESSION_COUNTS,
        "planned_anchor_counts": dict(planned),
        "role_counts": {role: len(cases[role]) for role in ROLES},
        "roles": role_artifacts,
        "exclusions": {role: dict(sorted(exclusion_counts[role].items())) for role in ROLES},
        "catalog_counts": catalog_counts,
        "catalog": _descriptor("catalog.jsonl", catalog_payload),
    }
    manifest["prepared_data_id"] = content_hash(manifest)
    artifacts["manifest.json"] = _encode(manifest) + b"\n"
    if output_dir.exists():
        for filename, payload in artifacts.items():
            if _read(output_dir / filename) != payload:
                raise ValueError("prepared dataset is immutable; choose a different output directory")
        return output_dir / "manifest.json"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".supervised-prepare-", dir=output_dir.parent) as temporary:
        stage = Path(temporary) / "prepared"
        stage.mkdir(mode=0o700)
        for filename, payload in artifacts.items():
            path = stage / filename
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(payload)
            path.chmod(0o600)
        if output_dir.exists():
            raise ValueError("prepared dataset output is immutable")
        stage.rename(output_dir)
    return output_dir / "manifest.json"


def _verified_artifact(directory: Path, descriptor: Mapping[str, Any], expected_path: str) -> bytes:
    if set(descriptor) != {"path", "sha256", "bytes"} or descriptor.get("path") != expected_path:
        raise ValueError("prepared artifact path/schema mismatch")
    path = _contained(directory.resolve(), directory / expected_path)
    payload = _read(path)
    if len(payload) != descriptor["bytes"] or hashlib.sha256(payload).hexdigest() != descriptor["sha256"]:
        raise ValueError("prepared artifact hash/bytes mismatch")
    return payload


def _validate_catalog(
    payload: bytes, manifest: Mapping[str, Any], config: ForecastConfig, dates: Mapping[str, tuple[date, ...]]
) -> None:
    expected: set[tuple[str, str, str]] = set()
    for role, days in dates.items():
        for day in days:
            bounds = session_bounds(day)
            assert bounds is not None
            as_of = bounds[0] + timedelta(minutes=61)
            while as_of + timedelta(minutes=15) <= bounds[1]:
                expected.update((role, symbol, as_of.isoformat()) for symbol in config.symbols)
                as_of += timedelta(minutes=5)
    planned: Counter[str] = Counter()
    eligible_ids: dict[str, list[str]] = {role: [] for role in ROLES}
    exclusions: dict[str, Counter[str]] = {role: Counter() for role in ROLES}
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    reasons = {
        "sequence_missing_minute",
        "sequence_not_available",
        "target_missing_minute",
        "target_not_available",
        "insufficient_history",
        "stale_source",
        "history_gap",
        "missing_return_endpoint",
        "previous_session_unavailable",
    }
    for line in payload.splitlines():
        row = json.loads(line)
        key = (row["role"], row["symbol"], row["as_of"])
        if key not in expected or type(row.get("eligible")) is not bool:
            raise ValueError("catalog missing/duplicate/unregistered anchor")
        expected.remove(key)
        role, symbol, as_of_string = key
        as_of = datetime.fromisoformat(as_of_string)
        if (
            row["session_date"] != as_of.date().isoformat()
            or row["target_time"] != (as_of + timedelta(minutes=15)).isoformat()
        ):
            raise ValueError("catalog as-of/target/session mismatch")
        planned[role] += 1
        group_key = (role, row["session_date"], symbol)
        group = groups.setdefault(
            group_key,
            {
                "session_date": row["session_date"],
                "symbol": symbol,
                "planned": 0,
                "eligible": 0,
                "excluded": 0,
                "exclusion_reasons": {},
            },
        )
        group["planned"] += 1
        if row["eligible"]:
            if row["exclusion_reason"] is not None or not isinstance(row.get("case_id"), str):
                raise ValueError("catalog eligibility/identity mismatch")
            eligible_ids[role].append(row["case_id"])
            group["eligible"] += 1
        else:
            reason = row["exclusion_reason"]
            if reason not in reasons or "case_id" in row:
                raise ValueError("catalog requires an explicit timestamp exclusion")
            exclusions[role][reason] += 1
            group["excluded"] += 1
            group["exclusion_reasons"][reason] = group["exclusion_reasons"].get(reason, 0) + 1
    counts: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLES}
    for (role, _, _), group in sorted(groups.items()):
        counts[role].append(group)
    if (
        expected
        or dict(planned) != manifest["planned_anchor_counts"]
        or counts != manifest["catalog_counts"]
        or {role: dict(counter) for role, counter in exclusions.items()} != manifest["exclusions"]
        or any(
            len(ids) != manifest["role_counts"][role]
            or len(set(ids)) != len(ids)
            or content_hash(ids) != manifest["roles"][role]["case_ids_hash"]
            for role, ids in eligible_ids.items()
        )
    ):
        raise ValueError("catalog counts/exclusions/stage cohort identity mismatch")


def _prepared_manifest(directory: Path) -> tuple[dict[str, Any], ForecastConfig, dict[str, tuple[date, ...]]]:
    manifest = _json(directory / "manifest.json")
    if manifest.get("schema_version") != PREPARED_VERSION or manifest.get("prepared_data_id") != content_hash(
        {key: value for key, value in manifest.items() if key != "prepared_data_id"}
    ):
        raise ValueError("prepared manifest identity/schema mismatch")
    registration, config, dates = _validate_registration(manifest["registration"])
    if (
        manifest.get("registration_id") != registration["registration_id"]
        or manifest.get("forecast_config") != config.model_dump(mode="json")
        or manifest.get("config_id") != config.config_id
        or manifest.get("class_order") != list(LABELS)
        or manifest.get("calendar") != "XNYS"
        or manifest.get("calendar_version") != CALENDAR_VERSION
        or manifest.get("representation_version") != REPRESENTATION_VERSION
        or manifest.get("sequence_channels") != list(SEQUENCE_CHANNELS)
        or manifest.get("sequence_length") != 60
        or manifest.get("static_feature_names") != list(OHLCV_FEATURE_NAMES)
        or manifest.get("feature_version") != OHLCV_FEATURE_VERSION
        or manifest.get("static_neural_feature_names")
        != [
            *OHLCV_FEATURE_NAMES,
            *("missing_" + name for name in OHLCV_FEATURE_NAMES),
            *("symbol_" + symbol for symbol in config.symbols),
        ]
        or manifest.get("static_cpu_feature_names")
        != [*OHLCV_FEATURE_NAMES, *("symbol_" + symbol for symbol in config.symbols)]
        or manifest.get("planned_session_counts") != _SESSION_COUNTS
        or set(manifest.get("roles", {})) != set(ROLES)
        or set(manifest.get("role_counts", {})) != set(ROLES)
    ):
        raise ValueError("prepared feature/config/role schema mismatch")
    source = manifest["source_manifest"]
    if (
        source.get("schema_version") != BAR_DATA_VERSION
        or source.get("data_id") != manifest["source_data_id"]
        or source.get("data_id") != content_hash({key: value for key, value in source.items() if key != "data_id"})
    ):
        raise ValueError("prepared source manifest identity mismatch")
    norm = manifest["normalizers"]
    if (
        manifest.get("normalizer_id") != content_hash(norm)
        or norm.get("fit_role") != "TRAIN"
        or norm.get("fit_case_count") != manifest["role_counts"]["TRAIN"]
        or norm.get("fit_time_positions_count") != manifest["role_counts"]["TRAIN"] * 60
        or norm["static"].get("binary_indices") != list(_BINARY_INDICES)
        or norm.get("train_case_ids_hash") != manifest["roles"]["TRAIN"].get("case_ids_hash")
    ):
        raise ValueError("normalizer TRAIN provenance mismatch")
    for group, dimensions in (("sequence", 6), ("static", 55)):
        mean, scale = np.asarray(norm[group]["mean"], dtype=float), np.asarray(norm[group]["scale"], dtype=float)
        if (
            mean.shape != (dimensions,)
            or scale.shape != (dimensions,)
            or not np.isfinite(mean).all()
            or (not np.isfinite(scale).all() or (scale <= 0).any())
        ):
            raise ValueError("invalid TRAIN normalizer shape/value")
    if any(norm["static"]["mean"][index] != 0 or norm["static"]["scale"][index] != 1 for index in _BINARY_INDICES):
        raise ValueError("binary missing-window flags must be preserved")
    root = Path(manifest["experiment_root"])
    if root != root.resolve() or not directory.resolve().is_relative_to(root):
        raise ValueError("prepared directory must remain below its registered experiment root")
    registry = _method_registry(root, registration)
    if registry != manifest.get("method_registry") or manifest.get("method_registry_id") != (
        registry["method_registry_id"] if registry is not None else None
    ):
        raise ValueError("prepared method registry identity mismatch")
    paths = set()
    for row in manifest["source_inventory"]:
        relative = Path(row["path"])
        if relative.is_absolute() or relative in paths:
            raise ValueError("source inventory path mismatch")
        paths.add(relative)
        payload = _read(_contained(root, root / relative))
        if len(payload) != row["bytes"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise ValueError("private source inventory hash/bytes mismatch")
    _validate_catalog(_verified_artifact(directory, manifest["catalog"], "catalog.jsonl"), manifest, config, dates)
    return manifest, config, dates


def _selection(path: Path | None, manifest: Mapping[str, Any]) -> None:
    if path is None:
        raise ValueError("TEST labels require a frozen selection seal")
    seal = _json(path)
    required = {
        "stage",
        "registration_id",
        "prepared_data_id",
        "config_id",
        "source_data_id",
        "class_order",
        "normalizer_id",
        "selected_candidate_id",
        "selected_artifact_ids",
        "selected_checkpoint_ids",
        "weights_id",
        "calibration_id",
        "temperature",
        "gate",
        "cpu_reference_candidate_id",
        "cpu_reference_artifact_id",
        "selection_id",
    }
    if (
        not required <= set(seal)
        or seal.get("stage") != "selection_frozen"
        or seal.get("selection_id")
        != content_hash({key: value for key, value in seal.items() if key != "selection_id"})
        or any(
            seal.get(key) != manifest[key]
            for key in (
                "registration_id",
                "prepared_data_id",
                "config_id",
                "source_data_id",
                "class_order",
                "normalizer_id",
            )
        )
    ):
        raise ValueError("TEST selection seal is incomplete or does not match prepared data")
    artifacts, checkpoints = seal["selected_artifact_ids"], seal["selected_checkpoint_ids"]
    candidate = seal["selected_candidate_id"]
    neural = candidate in {"tcn-32", "lstm-64"}
    if (
        not isinstance(candidate, str)
        or not candidate
        or not isinstance(artifacts, list)
        or not artifacts
        or any(not isinstance(item, str) or not item for item in artifacts)
        or not isinstance(checkpoints, list)
        or any(not isinstance(item, str) or not item for item in checkpoints)
        or (neural and (len(artifacts) != 3 or len(checkpoints) != 3))
        or any(
            not isinstance(seal[key], str)
            or len(seal[key]) != 64
            or any(character not in "0123456789abcdef" for character in seal[key])
            for key in ("weights_id", "calibration_id")
        )
        or not isinstance(seal["temperature"], (float, int))
        or isinstance(seal["temperature"], bool)
        or not math.isfinite(seal["temperature"])
        or not 0.25 <= seal["temperature"] <= 4
        or not isinstance(seal["gate"], dict)
        or not seal["gate"]
        or any(
            not isinstance(seal[key], str) or not seal[key]
            for key in ("cpu_reference_candidate_id", "cpu_reference_artifact_id")
        )
    ):
        raise ValueError("TEST selection seal must freeze weights/checkpoints/calibration/gate/reference")


def _array(payload: bytes, dtype: np.dtype[Any], shape: tuple[int, ...]) -> NDArray[Any]:
    value = np.load(io.BytesIO(payload), allow_pickle=False)
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != dtype
        or value.shape != shape
        or not np.isfinite(value).all()
    ):
        raise ValueError("prepared array shape/dtype/finite schema mismatch")
    value.setflags(write=False)
    return value


def load_stage(
    directory: Path, role: str, *, selection_path: Path | None = None
) -> tuple[SupervisedStage, dict[str, Any]]:
    """Validate named stage artifacts; decode TEST outcomes only after all selection components are sealed."""
    if role not in ROLES:
        raise ValueError("role must be one of TRAIN/TUNE/CAL/GATE/TEST")
    try:
        manifest, config, dates = _prepared_manifest(directory)
        if role == "TEST":
            _selection(selection_path, manifest)
        artifacts = manifest["roles"][role]
        count = artifacts["count"]
        if type(count) is not int or count < 0 or count != manifest["role_counts"][role]:
            raise ValueError("prepared role count mismatch")
        payloads = {
            key: _verified_artifact(
                directory,
                artifacts[key],
                f"{role}/{key}." + ("npy" if key in {"sequence", "static", "targets"} else "jsonl"),
            )
            for key in ("sequence", "static", "targets", "examples", "features", "cases")
        }
        sequence = _array(payloads["sequence"], np.dtype(np.float32), (count, 60, 6))
        static = _array(payloads["static"], np.dtype(np.float32), (count, 115))
        targets = _array(payloads["targets"], np.dtype(np.int64), (count,))
        examples = tuple(ForecastExample.model_validate_json(line) for line in payloads["examples"].splitlines())
        features = tuple(OhlcvFeatureRecord.model_validate_json(line) for line in payloads["features"].splitlines())
        cases = [json.loads(line) for line in payloads["cases"].splitlines()]
        if (
            len(examples) != count
            or len(features) != count
            or len(cases) != count
            or not np.isin(targets, [0, 1, 2]).all()
            or not np.isin(static[:, 55:], [0, 1]).all()
            or not np.isin(static[:, list(_BINARY_INDICES)], [0, 1]).all()
        ):
            raise ValueError("prepared case/class/mask schema mismatch")
        for index, (example, record, case) in enumerate(zip(examples, features, cases, strict=True)):
            bounds = session_bounds(example.session_date)
            if (
                bounds is None
                or example.session_date not in dates[role]
                or example.config_id != config.config_id
                or example.provenance != "historical"
                or example.symbol not in config.symbols
                or record.config_id != config.config_id
                or record.symbol != example.symbol
                or record.as_of != example.as_of
                or example.target_time != example.as_of + timedelta(minutes=15)
                or example.target_time > bounds[1]
                or example.as_of < bounds[0] + timedelta(minutes=61)
                or (example.as_of - bounds[0] - timedelta(minutes=61)).total_seconds() % 300
                or example.label is None
                or example.label_observed_at != example.target_time
                or example.exclusion_reason is not None
                or LABELS[targets[index]] != example.label
            ):
                raise ValueError("prepared example config/class/as-of/target/role mismatch")
            ids = case["sequence_bar_ids"]
            if (
                not isinstance(ids, list)
                or len(ids) != 61
                or len(set(ids)) != 61
                or any(
                    not isinstance(item, str) or len(item) != 64 or any(c not in "0123456789abcdef" for c in item)
                    for item in ids
                )
                or case["sequence_input_hash"] != _sequence_hash(ids)
                or case["static_input_hash"] != _static_hash(record)
                or case["feature_id"] != record.feature_id
                or case["source_data_id"] != manifest["source_data_id"]
                or case["case_id"] != example.example_id
                or record.base_example_id != example.example_id
                or example.observation_ids[-2:]
                != (
                    "supervised-sequence:" + case["sequence_input_hash"],
                    "supervised-static:" + case["static_input_hash"],
                )
                or case["price_example_id"]
                != example.model_copy(update={"observation_ids": example.observation_ids[:-2]}).example_id
            ):
                raise ValueError("prepared case input/bar/feature/source identity mismatch")
            assert example.target_price is not None
            change = (example.target_price / example.anchor_price - 1) * 10_000
            expected_label = "UP" if change > 10 else "DOWN" if change < -10 else "FLAT"
            if expected_label != example.label or float(change) != example.target_return_bps:
                raise ValueError("prepared target value/label mismatch")
        case_ids = tuple(row.example_id for row in examples)
        if len(set(case_ids)) != count or content_hash(case_ids) != artifacts["case_ids_hash"]:
            raise ValueError("prepared case IDs/count/hash mismatch")
        _, expected_static = _transform(np.zeros((count, 60, 6)), features, manifest["normalizers"], config.symbols)
        if not np.array_equal(static, expected_static):
            raise ValueError("prepared static inputs differ from raw feature records and TRAIN normalizers")
        if role == "TRAIN":
            norm = _normalizers(np.zeros((count, 60, 6)), features, cases)
            if any(
                norm[key] != manifest["normalizers"][key]
                for key in ("train_case_ids_hash", "train_raw_inputs_hash", "static")
            ):
                raise ValueError("TRAIN raw-input/normalizer fit provenance mismatch")
        return SupervisedStage(role, sequence, static, examples, features, targets, case_ids), manifest
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
        raise ValueError(f"invalid supervised stage: {error}") from error
