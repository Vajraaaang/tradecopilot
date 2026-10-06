"""Pinned, local-only CPU Kronos inference with individual uncalibrated paths."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from zoneinfo import ZoneInfo

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from tradecopilot.forecast.bars import HistoricalBar

CHECKPOINT_REVISIONS = {
    "Kronos-mini": "f4e68697d9d5aed55cef5c96aabc3376bcad9f81",
    "Kronos-small": "901c26c1332695a2a8f243eb2f37243a37bea320",
    "Kronos-Tokenizer-2k": "26966d0035065a0cae0ebad7af8ece35bc1fb51c",
    "Kronos-Tokenizer-base": "0e0117387f39004a9016484a186a908917e22426",
}
CHECKPOINT_FILES = {
    "Kronos-mini": {
        "config.json": (225, "70daca2cb11e3a979dd6b8ac12ee08e2aace877acf28f5b8dfb4fe5609736201"),
        "model.safetensors": (16440776, "a7d5f37e2e9fbd9891f7d7d4f72574512dd1f704fee14223e0a8cd0fbf54197c"),
    },
    "Kronos-small": {
        "config.json": (228, "5e0f6a605d5f81b5c9b559fe5cf716a1acb041c744e6f41bd05b097b7a685396"),
        "model.safetensors": (98980656, "b082dfcbd8e8c142a725c8bbb99781802f38fec81210e13479effb32b3c3e020"),
    },
    "Kronos-Tokenizer-2k": {
        "config.json": (301, "0b30a443affb03e05a876a083857de9164f899feb7b4d261da02c485c9a3e3b6"),
        "model.safetensors": (15842376, "b97ec46b3b72160509e289183eaf7bdf5f0dac5bb9b49522f6d46638a99a8717"),
    },
    "Kronos-Tokenizer-base": {
        "config.json": (301, "2366e7ccfec76cbc19cf3c4c1b9c5d901be336ca1e83f2d2292c9bff381b77a2"),
        "model.safetensors": (15842368, "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee"),
    },
}
_VARIANTS = {
    "mini": ("Kronos-mini", "Kronos-Tokenizer-2k", 2048),
    "small": ("Kronos-small", "Kronos-Tokenizer-base", 512),
}
_SOURCE_COMMIT = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
_SOURCE_REPOSITORY = "https://github.com/shiyu-coder/Kronos"
_SOURCE_FILES = {
    "kronos.py": "2960874d0487078a0900b588b340f9679ab3e804f5750c28ab9922eff5723fc1",
    "module.py": "a07edbadc0e96804c8158c021bbc6063bb7cc43b34d7fc470d5c8ff2005a409f",
    "LICENSE": "acb2d194d378204e5f2be4dcd24d39ecac437903620c790c3315a96dab388fdc",
}
_VENDOR_DIR = Path(__file__).resolve().parents[1] / "_vendor" / "kronos"
_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]
_MINUTE = timedelta(minutes=1)
_NEW_YORK = ZoneInfo("America/New_York")


def _json_file(path: Path) -> Any:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("missing, symlinked, or oversized Kronos manifest/config")
    return json.loads(path.read_bytes())


def _verify_file(path: Path, expected_bytes: int, expected_hash: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size != expected_bytes:
        raise ValueError(f"Kronos file size/path mismatch: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != expected_hash:
        raise ValueError(f"Kronos file hash mismatch: {path.name}")
    return {"name": path.name, "bytes": expected_bytes, "sha256": expected_hash}


def _verify_source() -> dict[str, Any]:
    manifest = _json_file(_VENDOR_DIR / "UPSTREAM.json")
    if (
        not isinstance(manifest, dict) or manifest.get("schema_version") != "kronos-upstream-source-v1"
        or manifest.get("repository") != _SOURCE_REPOSITORY or manifest.get("commit") != _SOURCE_COMMIT
        or manifest.get("license") != "MIT" or not isinstance(manifest.get("files"), dict)
        or set(manifest["files"]) != set(_SOURCE_FILES)
    ):
        raise ValueError("Kronos upstream source manifest mismatch")
    files = {}
    for name, digest in _SOURCE_FILES.items():
        record = manifest["files"][name]
        if not isinstance(record, dict) or record.get("packaged_sha256") != digest:
            raise ValueError("Kronos source registry hash mismatch")
        path = _VENDOR_DIR / name
        files[name] = _verify_file(path, path.stat().st_size, digest)
    return {"repository": _SOURCE_REPOSITORY, "commit": _SOURCE_COMMIT, "license": "MIT", "files": files}


def _verify_cache(cache: Path, required: tuple[str, str]) -> dict[str, dict[str, Any]]:
    manifest = _json_file(cache / "manifest.json")
    if (
        not isinstance(manifest, dict) or set(manifest) != {"schema_version", "downloaded_at", "checkpoints"}
        or manifest["schema_version"] != "kronos-local-checkpoints-v1"
        or not isinstance(manifest["checkpoints"], list) or not 2 <= len(manifest["checkpoints"]) <= 4
    ):
        raise ValueError("invalid Kronos checkpoint manifest schema")
    downloaded = datetime.fromisoformat(manifest["downloaded_at"])
    if downloaded.tzinfo is None or downloaded.utcoffset() != timedelta(0):
        raise ValueError("Kronos download timestamp must be UTC")
    records = {}
    for record in manifest["checkpoints"]:
        if not isinstance(record, dict) or set(record) != {"name", "repository", "revision", "files"}:
            raise ValueError("invalid Kronos checkpoint record")
        name = record["name"]
        if (
            not isinstance(name, str) or name not in CHECKPOINT_REVISIONS or name in records
            or record["repository"] != f"NeoQuasar/{name}" or record["revision"] != CHECKPOINT_REVISIONS[name]
            or not isinstance(record["files"], list) or len(record["files"]) != 2
        ):
            raise ValueError("unknown, duplicate, or unpinned Kronos checkpoint")
        directory = cache / name
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("missing or symlinked Kronos checkpoint directory")
        if any(path.name not in {"config.json", "model.safetensors", ".cache"} for path in directory.iterdir()):
            raise ValueError("Kronos checkpoint directories accept config and safetensors only")
        files = {}
        for file in record["files"]:
            if not isinstance(file, dict) or set(file) != {"name", "bytes", "sha256"}:
                raise ValueError("invalid Kronos checkpoint file record")
            filename = file["name"]
            if not isinstance(filename, str) or filename not in CHECKPOINT_FILES[name] or filename in files:
                raise ValueError("Kronos checkpoint file must be config.json or model.safetensors")
            size, digest = CHECKPOINT_FILES[name][filename]
            if type(file["bytes"]) is not int or file["bytes"] != size or file["sha256"] != digest:
                raise ValueError("Kronos checkpoint manifest does not match pinned file identity")
            files[filename] = _verify_file(directory / filename, size, digest)
        records[name] = {"name": name, "repository": record["repository"], "revision": record["revision"],
                         "files": [files[filename] for filename in ("config.json", "model.safetensors")]}
    if any(name not in records for name in required):
        raise ValueError("matching local Kronos model and tokenizer are required")
    return records


def _validate_times(history: Sequence[HistoricalBar], future: Sequence[datetime]) -> list[HistoricalBar]:
    from tradecopilot.forecast.bars import HistoricalBar
    from tradecopilot.forecast.sessions import session_bounds

    if len(history) != 60 or not all(isinstance(bar, HistoricalBar) for bar in history):
        raise ValueError("Kronos requires exactly 60 HistoricalBar inputs")
    bars = [HistoricalBar.model_validate(bar.model_dump()) for bar in history]
    as_of = bars[-1].end_time
    day = bars[0].start_time.astimezone(_NEW_YORK).date()
    bounds = session_bounds(day)
    if bounds is None or len({bar.symbol for bar in bars}) != 1:
        raise ValueError("Kronos history must contain a single symbol in one regular XNYS session")
    for index, bar in enumerate(bars):
        if (
            bar.start_time.second or bar.start_time.microsecond or bar.available_at > as_of
            or not bounds[0] <= bar.start_time < bar.end_time <= bounds[1]
            or (index > 0 and bar.start_time != bars[index - 1].end_time)
        ):
            raise ValueError("Kronos input minutes must be consecutive, regular, and available at the anchor")
    if not 1 <= len(future) <= 60:
        raise ValueError("Kronos requires 1-60 future minute ends")
    for index, timestamp in enumerate(future):
        if (
            not isinstance(timestamp, datetime) or timestamp.tzinfo is None
            or timestamp.utcoffset() != timedelta(0) or timestamp <= as_of
            or timestamp.second or timestamp.microsecond
            or (index > 0 and timestamp - future[index - 1] != _MINUTE)
        ):
            raise ValueError("Kronos future timestamps must be increasing consecutive UTC minute ends after history")
        future_bounds = session_bounds(timestamp.astimezone(_NEW_YORK).date())
        if future_bounds is None or not future_bounds[0] < timestamp <= future_bounds[1]:
            raise ValueError("Kronos future timestamp is not a regular XNYS minute end")
    return bars


class KronosEngine:
    """Use verified local mini/small weights; inference never downloads or accepts outcomes.

    ``amount`` is volume times mean OHLC, a derived proxy rather than provider
    dollar volume. Forecast paths and sample intervals are not calibrated.
    """

    def __init__(self, variant: Literal["mini", "small"], cache_dir: Path, *, threads: int = 1) -> None:
        if variant not in _VARIANTS or type(threads) is not int or not 1 <= threads <= 8:
            raise ValueError("Kronos supports mini/small and 1-8 CPU threads")
        if not cache_dir.is_absolute() or cache_dir.is_symlink() or not cache_dir.is_dir():
            raise ValueError("Kronos cache must be an existing absolute local directory")
        model_name, tokenizer_name, context = _VARIANTS[variant]
        try:
            source = _verify_source()
            checkpoints = _verify_cache(cache_dir, (model_name, tokenizer_name))
        except (OSError, TypeError, KeyError, ValueError) as error:
            raise ValueError(f"invalid local Kronos cache/source: {error}") from error
        try:
            self._torch = importlib.import_module("torch")
            self._pandas = importlib.import_module("pandas")
            importlib.import_module("huggingface_hub")
            importlib.import_module("safetensors.torch")
            importlib.import_module("einops")
            upstream = importlib.import_module("tradecopilot._vendor.kronos.kronos")
        except ImportError as error:
            raise RuntimeError("Kronos requires the optional kronos extra: uv sync --extra kronos") from error
        self._torch.set_num_threads(threads)
        local_options = {"local_files_only": True, "token": False, "map_location": "cpu", "strict": True}
        with self._torch.random.fork_rng(devices=[]):
            model = upstream.Kronos.from_pretrained(str(cache_dir / model_name), **local_options)
            tokenizer = upstream.KronosTokenizer.from_pretrained(str(cache_dir / tokenizer_name), **local_options)
        model = model.to(device="cpu", dtype=self._torch.float32).eval()
        tokenizer = tokenizer.to(device="cpu", dtype=self._torch.float32).eval()
        self._predictor = upstream.KronosPredictor(model, tokenizer, device="cpu", max_context=context)
        self.metadata: dict[str, Any] = {
            "schema_version": "kronos-local-engine-v1", "variant": variant, "source": source,
            "model": checkpoints[model_name], "tokenizer": checkpoints[tokenizer_name],
            "device": "cpu", "dtype": "float32", "threads": threads, "max_context": context,
            "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "tokenizer_parameter_count": sum(parameter.numel() for parameter in tokenizer.parameters()),
            "dependencies": {
                name: version(name) for name in ("torch", "pandas", "huggingface-hub", "safetensors", "einops")
            },
            "columns": list(_COLUMNS), "input_bars": 60, "calendar": "XNYS", "artifact_timezone": "UTC",
            "model_timezone": "America/New_York", "timestamp_semantics": "completed minute bar ends",
            "amount_policy": "derived_proxy: volume * mean(open,high,low,close); not provider dollar volume",
            "sampler": {"temperature": 1.0, "top_p": 0.9, "top_k": 0, "sample_count_per_series": 1,
                        "paths": "individual duplicated-input batch series", "seed_policy": "one CPU seed per call"},
            "intervals_calibrated": False, "fine_tuned": False,
        }
        self.last_prediction: dict[str, Any] = {}

    def predict(
        self, history: Sequence[HistoricalBar], future_times: Sequence[datetime], *, seed: int = 42, samples: int = 20,
    ) -> NDArray[np.float64]:
        """Return all individual paths as [samples, future minutes, OHLC/volume/amount]."""
        if type(seed) is not int or not 0 <= seed < 2**32 or type(samples) is not int or not 1 <= samples <= 32:
            raise ValueError("Kronos seed must be a uint32 integer; samples must be 1-32")
        bars = _validate_times(history, future_times)
        values = np.asarray([[float(bar.opening), float(bar.high), float(bar.low), float(bar.close), float(bar.volume)]
                             for bar in bars], dtype=np.float64)
        values = np.column_stack((values, values[:, 4] * values[:, :4].mean(axis=1)))
        if not np.isfinite(values).all() or not np.isfinite(values.astype(np.float32)).all():
            raise ValueError("Kronos input OHLCV/proxy must remain finite in float32")
        frame = self._pandas.DataFrame(values, columns=_COLUMNS)
        x_times = self._pandas.Series([bar.end_time.astimezone(_NEW_YORK).replace(tzinfo=None) for bar in bars])
        y_times = self._pandas.Series([time.astimezone(_NEW_YORK).replace(tzinfo=None) for time in future_times])
        with self._torch.random.fork_rng(devices=[]), self._torch.inference_mode():
            # Seed the CPU generator only; torch.manual_seed also mutates CUDA generators.
            self._torch.random.default_generator.manual_seed(seed)
            frames = self._predictor.predict_batch(
                [frame.copy() for _ in range(samples)], [x_times.copy() for _ in range(samples)],
                [y_times.copy() for _ in range(samples)], pred_len=len(future_times),
                T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=False,
            )
        if not isinstance(frames, (list, tuple)) or len(frames) != samples:
            raise ValueError("Kronos returned an invalid sample count")
        for result in frames:
            if (
                not isinstance(result, self._pandas.DataFrame) or list(result.columns) != _COLUMNS
                or result.shape != (len(future_times), 6) or result.index.to_list() != y_times.to_list()
            ):
                raise ValueError("Kronos output columns, shape, or timestamps are invalid")
        paths = np.stack([result.to_numpy(dtype=np.float64, copy=True) for result in frames])
        diagnostics = self.diagnostics(paths)
        self.last_prediction = {
            "seed": seed, "samples": samples, "future_minutes": len(future_times),
            "as_of_utc": bars[-1].end_time.astimezone(UTC).isoformat(),
            "future_end_times_utc": [time.isoformat() for time in future_times], "diagnostics": diagnostics,
        }
        return paths

    @staticmethod
    def diagnostics(paths: NDArray[np.float64]) -> dict[str, int]:
        """Reject unusable paths and count retained, unrepaired physical inconsistencies."""
        if paths.ndim != 3 or paths.shape[2] != 6 or not np.isfinite(paths).all() or np.any(paths[:, :, 3] <= 0):
            raise ValueError("Kronos paths must have six finite columns and positive closes")
        prices = paths[:, :, :4]
        invalid_ohlc = (
            (prices[:, :, 1] < np.maximum(prices[:, :, 0], prices[:, :, 3]))
            | (prices[:, :, 2] > np.minimum(prices[:, :, 0], prices[:, :, 3]))
            | (prices[:, :, 1] < prices[:, :, 2]) | np.any(prices <= 0, axis=2)
        )
        return {"paths": int(paths.shape[0]), "rows": int(paths.shape[0] * paths.shape[1]),
                "nonphysical_ohlc_rows": int(np.count_nonzero(invalid_ohlc)),
                "negative_volume_rows": int(np.count_nonzero(paths[:, :, 4] < 0)),
                "negative_amount_rows": int(np.count_nonzero(paths[:, :, 5] < 0))}
