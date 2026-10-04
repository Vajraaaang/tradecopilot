"""Immutable causal observations with training-only normalization and sealed test access."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from numpy.typing import NDArray

from tradecopilot.forecast.bars import HistoricalBar, load_bar_dataset
from tradecopilot.forecast.contracts import content_hash
from tradecopilot.forecast.ohlcv_features import OHLCV_FEATURE_NAMES, OHLCV_FEATURE_VERSION, OhlcvFeatureBuilder
from tradecopilot.forecast.sessions import session_bounds
from tradecopilot.rl.contracts import EpisodeData

SCHEMA = "rl-prepared-market-v1"
_TIME_ENCODING = {"starts_ends": "exact-minute epoch seconds",
                  "available_at": "ceil UTC epoch seconds; never release before source availability"}
_ARRAYS = ("starts", "ends", "available_at", "opens", "closes", "volumes", "features")
_NY = ZoneInfo("America/New_York")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def _verify_id(value: dict[str, Any], key: str) -> None:
    if value.get(key) != content_hash({k: v for k, v in value.items() if k != key}):
        raise ValueError(f"invalid {key}")


def _dates(reg: dict[str, Any]) -> dict[str, set[date]]:
    result = {}
    for role in ("train", "tune", "test"):
        first, last = map(date.fromisoformat, reg["splits"][role])
        if first > last:
            raise ValueError("reversed split")
        days = set()
        while first <= last:
            if session_bounds(first) is not None:
                days.add(first)
            first += timedelta(days=1)
        if not days:
            raise ValueError("empty split")
        result[role] = days
    if any(result[a] & result[b] for a, b in (("train", "tune"), ("train", "test"), ("tune", "test"))):
        raise ValueError("overlapping splits")
    if not (max(result["train"]) < min(result["tune"]) <= max(result["tune"]) < min(result["test"])):
        raise ValueError("splits must be chronological")
    return result


def _source_check(bars: list[HistoricalBar], source: dict[str, Any], reg: dict[str, Any]) -> None:
    meta = source["source_metadata"]
    bounds = reg["source_range"]
    expected = {"provider": bounds["provider"], "feed": bounds["feed"],
                "adjustment_policy": bounds["adjustment"], "start_date": bounds["start"],
                "end_date_exclusive": bounds["end_exclusive"], "selected_symbols": reg["symbols"]}
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError("source metadata differs from registration")
    if bool(meta.get("fixture", False)) != bool(reg.get("fixture", False)):
        raise ValueError("synthetic source must have matching fixture registration")
    if not bars or set(bar.symbol for bar in bars) != set(reg["symbols"]):
        raise ValueError("source symbols differ from registration")
    if len(reg["symbols"]) != 5 or len(set(reg["symbols"])) != 5:
        raise ValueError("five unique registered symbols required")
    first, last = date.fromisoformat(bounds["start"]), date.fromisoformat(bounds["end_exclusive"])
    if any(not first <= bar.start_time.astimezone(_NY).date() < last for bar in bars):
        raise ValueError("source bar outside registered bounds")
    if any(bar.start_time.second or bar.start_time.microsecond for bar in bars):
        raise ValueError("source bars must have exact minute boundaries")
    if min(bar.start_time.astimezone(_NY).date() for bar in bars) != first:
        raise ValueError("source primer date absent")
    if reg.get("forecast_profile") != "MARKET_ONLY":
        raise ValueError("only registered MARKET_ONLY representation is supported")


def prepare_data(bars_directory: Path, registration_path: Path, output_dir: Path) -> Path:
    """Build label-free features; source context may precede the three episode splits."""
    if output_dir.exists():
        raise ValueError("prepared output is immutable; choose a new directory")
    reg = _read(registration_path)
    _verify_id(reg, "registration_id")
    dates = _dates(reg)
    bars, source = load_bar_dataset(bars_directory)
    _source_check(bars, source, reg)
    groups: dict[tuple[str, date], list[HistoricalBar]] = defaultdict(list)
    for bar in bars:
        groups[(bar.symbol, bar.start_time.astimezone(_NY).date())].append(bar)
    for split_role, days in dates.items():
        if any((symbol, day) not in groups for symbol in reg["symbols"] for day in days):
            raise ValueError(f"missing source episode in {split_role}")
    config_id = content_hash({"registration_id": reg["registration_id"], "feature_version": OHLCV_FEATURE_VERSION})
    builder = OhlcvFeatureBuilder(bars)
    raw_episodes = []
    fit = []
    fit_ids: list[str] = []
    warmup = int(reg["environment"]["warmup_minutes"])
    for symbol, day in sorted(groups, key=lambda key: (key[1], key[0])):
        role = next((role for role, days in dates.items() if day in days), None)
        if role is None:
            continue
        rows = sorted(groups[(symbol, day)], key=lambda bar: bar.start_time)
        bounds = session_bounds(day)
        assert bounds is not None
        values = []
        for bar in rows:
            try:
                record = builder.build_as_of(symbol, bar.end_time, config_id)
                values.append([np.nan if record.values[name] is None else record.values[name]
                               for name in OHLCV_FEATURE_NAMES])
            except ValueError as error:
                if "anchor" not in str(error) and "reference unavailable" not in str(error):
                    raise
                values.append([np.nan] * len(OHLCV_FEATURE_NAMES))
        raw = np.asarray(values, dtype=np.float64)
        eligible = np.asarray([(bar.end_time - bounds[0]).total_seconds() >= warmup * 60
                               and bar.available_at <= bar.end_time for bar in rows])
        if role == "train":
            fit.append(raw[eligible])
            fit_ids.extend(bar.bar_id for bar, ok in zip(rows, eligible, strict=True) if ok)
        raw_episodes.append((symbol, day, role, rows, bounds, raw))
    train = np.concatenate(fit)
    if not len(train):
        raise ValueError("no eligible training rows for normalization")
    count = np.isfinite(train).sum(axis=0)
    mean = np.divide(np.nansum(train, axis=0), count, out=np.zeros(len(OHLCV_FEATURE_NAMES)), where=count > 0)
    scale = np.sqrt(np.divide(np.nansum((train - mean) ** 2, axis=0), count,
                              out=np.zeros_like(mean), where=count > 0))
    scale[scale == 0] = 1
    flags = [index for index, name in enumerate(OHLCV_FEATURE_NAMES) if name.startswith("missing_window_")]
    mean[flags], scale[flags] = 0, 1
    normalizer = {"mean": mean.tolist(), "scale": scale.tolist(), "feature_names": list(OHLCV_FEATURE_NAMES),
                  "fit_dates": sorted(day.isoformat() for day in dates["train"]),
                  "fit_row_hash": content_hash(fit_ids), "fit_rows": len(train), "warmup_minutes": warmup}
    feature_names = (*OHLCV_FEATURE_NAMES, *(f"missing_{name}" for name in OHLCV_FEATURE_NAMES),
                     *(f"symbol_{symbol}" for symbol in reg["symbols"]))
    arrays: dict[str, NDArray[Any]] = {}
    table = []
    for ordinal, (symbol, day, role, rows, bounds, raw) in enumerate(raw_episodes):
        missing = ~np.isfinite(raw)
        normalized = (np.where(missing, mean, raw) - mean) / scale
        onehot = np.zeros((len(rows), 5))
        onehot[:, reg["symbols"].index(symbol)] = 1
        features = np.concatenate((normalized, missing.astype(float), onehot), axis=1).astype(np.float32)
        key = f"episode_{ordinal:06d}"
        row_arrays: dict[str, NDArray[Any]] = {
            "starts": np.asarray([int(bar.start_time.timestamp()) for bar in rows], dtype=np.int64),
            "ends": np.asarray([int(bar.end_time.timestamp()) for bar in rows], dtype=np.int64),
            "available_at": np.asarray([math.ceil(bar.available_at.timestamp()) for bar in rows], dtype=np.int64),
            "opens": np.asarray([float(bar.opening) for bar in rows], dtype=np.float64),
            "closes": np.asarray([float(bar.close) for bar in rows], dtype=np.float64),
            "volumes": np.asarray([float(bar.volume) for bar in rows], dtype=np.float64), "features": features,
        }
        arrays.update({f"{key}_{name}": value for name, value in row_arrays.items()})
        table.append({"key": key, "symbol": symbol, "session_date": day.isoformat(), "role": role,
                      "session_open": int(bounds[0].timestamp()), "session_close": int(bounds[1].timestamp()),
                      "rows": len(rows)})
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        np.savez_compressed(stage / "episodes.npz", allow_pickle=False, **arrays)
        blob = (stage / "episodes.npz").read_bytes()
        manifest = {"schema_version": SCHEMA, "registration": reg, "registration_id": reg["registration_id"],
                    "time_encoding": _TIME_ENCODING,
                    "source_manifest": source, "source_data_id": source["data_id"],
                    "source_metadata": source["source_metadata"], "config_id": config_id,
                    "normalizer": normalizer, "feature_names": feature_names, "episodes": table,
                    "role_counts": dict(Counter(row["role"] for row in table)),
                    "blob": {"path": "episodes.npz", "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}}
        manifest["data_id"] = content_hash(manifest)
        (stage / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, allow_nan=False))
        if output_dir.exists():
            raise ValueError("prepared output is immutable")
        stage.rename(output_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return output_dir / "manifest.json"


def load_prepared(directory: Path, role: str, *, selection_path: Path | None = None
                  ) -> tuple[list[EpisodeData], dict[str, Any]]:
    """Verify immutable cache identity; release test only against a matching hashed selection."""
    if role not in ("train", "tune", "test"):
        raise ValueError("unknown episode role")
    try:
        meta = _read(directory / "manifest.json")
        _verify_id(meta, "data_id")
        if meta["schema_version"] != SCHEMA:
            raise ValueError("unsupported prepared schema")
        if "time_encoding" in meta and meta["time_encoding"] != _TIME_ENCODING:
            raise ValueError("unsupported availability time encoding")
        reg = meta["registration"]
        _verify_id(reg, "registration_id")
        source = meta["source_manifest"]
        _verify_id(source, "data_id")
        if (meta["registration_id"] != reg["registration_id"] or meta["source_data_id"] != source["data_id"]
                or meta["source_metadata"] != source["source_metadata"]):
            raise ValueError("source or registration identity mismatch")
        expected_config = content_hash({"registration_id": reg["registration_id"],
                                        "feature_version": OHLCV_FEATURE_VERSION})
        if meta["config_id"] != expected_config:
            raise ValueError("feature configuration identity mismatch")
        source_range = reg["source_range"]
        expected_source = {"provider": source_range["provider"], "feed": source_range["feed"],
                           "adjustment_policy": source_range["adjustment"],
                           "start_date": source_range["start"],
                           "end_date_exclusive": source_range["end_exclusive"],
                           "selected_symbols": reg["symbols"]}
        if any(source["source_metadata"].get(k) != v for k, v in expected_source.items()):
            raise ValueError("source registration bounds or symbols mismatch")
        if role == "test":
            if selection_path is None:
                raise ValueError("test locked until selection frozen")
            selection = _read(selection_path)
            _verify_id(selection, "selection_id")
            if (selection.get("stage") != "selection_frozen"
                    or selection.get("registration_id") != meta["registration_id"]
                    or selection.get("prepared_data_id") != meta["data_id"]):
                raise ValueError("test selection seal mismatch")
        dates = _dates(reg)
        names = tuple(meta["feature_names"])
        expected = (*OHLCV_FEATURE_NAMES, *(f"missing_{name}" for name in OHLCV_FEATURE_NAMES),
                    *(f"symbol_{symbol}" for symbol in reg["symbols"]))
        if names != expected or len(names) != 115:
            raise ValueError("invalid observation feature order")
        normalizer = meta["normalizer"]
        mean, scale = np.asarray(normalizer["mean"]), np.asarray(normalizer["scale"])
        if (mean.shape != (55,) or scale.shape != (55,) or not np.isfinite(mean).all()
                or not np.isfinite(scale).all() or (scale <= 0).any()
                or normalizer["feature_names"] != list(OHLCV_FEATURE_NAMES)
                or normalizer["fit_dates"] != sorted(day.isoformat() for day in dates["train"])):
            raise ValueError("invalid training normalizer")
        if meta["blob"]["path"] != "episodes.npz":
            raise ValueError("invalid blob path")
        blob = (directory / "episodes.npz").read_bytes()
        if len(blob) != meta["blob"]["bytes"] or hashlib.sha256(blob).hexdigest() != meta["blob"]["sha256"]:
            raise ValueError("prepared blob hash mismatch")
        result = []
        seen = set()
        with np.load(directory / "episodes.npz", allow_pickle=False) as archive:
            required = {f"{row['key']}_{name}" for row in meta["episodes"] for name in _ARRAYS}
            if set(archive.files) != required:
                raise ValueError("unexpected observation arrays")
            for ordinal, row in enumerate(meta["episodes"]):
                day = date.fromisoformat(row["session_date"])
                pair = row["symbol"], day
                bounds = session_bounds(day)
                if (row["key"] != f"episode_{ordinal:06d}" or pair in seen or row["symbol"] not in reg["symbols"]
                        or row["role"] not in dates or day not in dates[row["role"]] or bounds is None
                        or row["session_open"] != int(bounds[0].timestamp())
                        or row["session_close"] != int(bounds[1].timestamp())):
                    raise ValueError("invalid episode split or session")
                seen.add(pair)
                values = {name: archive[f"{row['key']}_{name}"] for name in _ARRAYS}
                for name, arr in values.items():
                    dtype = np.int64 if name in _ARRAYS[:3] else np.float32 if name == "features" else np.float64
                    shape = (row["rows"], 115) if name == "features" else (row["rows"],)
                    if arr.dtype != dtype or arr.shape != shape or not np.isfinite(arr).all():
                        raise ValueError("invalid cache array shape or dtype")
                features = values["features"]
                if (values["starts"] % 60 != 0).any() or (values["ends"] % 60 != 0).any():
                    raise ValueError("cache bars must have exact minute boundaries")
                expected_symbol = np.zeros(5, dtype=np.float32)
                expected_symbol[reg["symbols"].index(row["symbol"])] = 1
                if (not np.isin(features[:, 55:110], (0, 1)).all()
                        or not (features[:, 110:] == expected_symbol).all()
                        or (features[:, :55][features[:, 55:110].astype(bool)] != 0).any()):
                    raise ValueError("invalid missing flags or symbol encoding")
                flag_indices = [i for i, name in enumerate(OHLCV_FEATURE_NAMES)
                                if name.startswith("missing_window_")]
                if not np.isin(features[:, flag_indices], (0, 1)).all():
                    raise ValueError("window flags must retain binary encoding")
                episode = EpisodeData(symbol=row["symbol"], session_date=day, session_open=row["session_open"],
                                      session_close=row["session_close"], feature_names=names,
                                      data_id=meta["data_id"], **values)
                if row["role"] == role:
                    result.append(episode)
        expected_pairs = {(symbol, day) for days in dates.values() for day in days for symbol in reg["symbols"]}
        if seen != expected_pairs or meta["role_counts"] != dict(Counter(row["role"] for row in meta["episodes"])):
            raise ValueError("incomplete episode coverage")
        return result, meta
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid prepared data: {error}") from error
