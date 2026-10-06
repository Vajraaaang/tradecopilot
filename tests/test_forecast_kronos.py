"""Synthetic adapter contracts; the tiny backend is not model-quality evidence."""

from __future__ import annotations

import hashlib
import importlib
import json
import runpy
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tradecopilot.forecast.bars import HistoricalBar

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["open", "high", "low", "close", "volume", "amount"]
REVISIONS = {
    "Kronos-mini": "f4e68697d9d5aed55cef5c96aabc3376bcad9f81",
    "Kronos-small": "901c26c1332695a2a8f243eb2f37243a37bea320",
    "Kronos-Tokenizer-2k": "26966d0035065a0cae0ebad7af8ece35bc1fb51c",
    "Kronos-Tokenizer-base": "0e0117387f39004a9016484a186a908917e22426",
}


def adapter():
    assert importlib.util.find_spec("tradecopilot.forecast.kronos") is not None, "local Kronos adapter is missing"
    return importlib.import_module("tradecopilot.forecast.kronos")


def history(start=None):
    start = start or datetime(2026, 10, 1, 13, 30, tzinfo=UTC)
    return [HistoricalBar(
        symbol="AAPL", start_time=start + timedelta(minutes=i),
        end_time=start + timedelta(minutes=i + 1), available_at=start + timedelta(minutes=i + 1),
        opening=Decimal("100") + Decimal(i) / 100, high=Decimal("102"), low=Decimal("99"),
        close=Decimal("101"), volume=Decimal(i + 100), source="alpaca_iex_1min_bar",
    ) for i in range(60)]


def future(bars, count=15, start=None):
    start = start or bars[-1].end_time + timedelta(minutes=1)
    return [start + timedelta(minutes=i) for i in range(count)]


@pytest.fixture
def local_cache(tmp_path, monkeypatch):
    records = []
    for name, revision in REVISIONS.items():
        folder = tmp_path / name
        folder.mkdir()
        config = {"s1_bits": 10, "s2_bits": 10}
        if "Tokenizer" in name:
            config["d_in"] = 6
        files = []
        for filename, raw in (
            ("config.json", json.dumps(config).encode()), ("model.safetensors", b"synthetic-placeholder"),
        ):
            (folder / filename).write_bytes(raw)
            files.append({"name": filename, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        records.append({"name": name, "repository": f"NeoQuasar/{name}", "revision": revision, "files": files})
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema_version": "kronos-local-checkpoints-v1", "downloaded_at": "2026-10-06T00:00:00+00:00",
        "checkpoints": records,
    }))
    module = adapter()
    monkeypatch.setattr(module, "CHECKPOINT_FILES", {
        record["name"]: {file["name"]: (file["bytes"], file["sha256"]) for file in record["files"]}
        for record in records
    })
    return tmp_path


@pytest.fixture
def backend(monkeypatch):
    module = adapter()
    torch = pytest.importorskip("torch")
    pandas = pytest.importorskip("pandas")
    calls = {"loads": [], "predictions": []}

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["loads"].append((Path(path), kwargs))
            return cls()

    class TinyPredictor:
        def __init__(self, model, tokenizer, **kwargs):
            calls["predictor"] = (model, tokenizer, kwargs)

        def predict_batch(self, dfs, xs, ys, **kwargs):
            calls["predictions"].append((dfs, xs, ys, kwargs))
            paths = []
            for y in ys:
                values = torch.rand((len(y), 6)).numpy().astype(np.float64) + 100
                values[:, 1] = values[:, :4].max(axis=1)
                values[:, 2] = values[:, :4].min(axis=1)
                paths.append(pandas.DataFrame(values, columns=COLUMNS, index=y))
            return paths

    fake = SimpleNamespace(Kronos=TinyModel, KronosTokenizer=TinyModel, KronosPredictor=TinyPredictor)
    original_import = module.importlib.import_module

    def load(name):
        if name == "tradecopilot._vendor.kronos.kronos":
            return fake
        return original_import(name)

    monkeypatch.setattr(module.importlib, "import_module", load)
    return module, torch, pandas, calls, fake


def test_adapter_module_import_does_not_import_optional_backends():
    adapter()
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    assert name.split('.')[0] not in {'torch', 'pandas', 'huggingface_hub', 'safetensors', 'einops'}
    assert not name.startswith('tradecopilot._vendor')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from tradecopilot.forecast.kronos import KronosEngine
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("variant,model,tokenizer,context", [
    ("mini", "Kronos-mini", "Kronos-Tokenizer-2k", 2048),
    ("small", "Kronos-small", "Kronos-Tokenizer-base", 512),
])
def test_known_pair_local_safetensors_cpu_eval_and_metadata(backend, local_cache, variant, model, tokenizer, context):
    module, torch, _, calls, _ = backend
    engine = module.KronosEngine(variant, local_cache)
    assert [path.name for path, _ in calls["loads"]] == [model, tokenizer]
    assert all(kwargs == {"local_files_only": True, "token": False, "map_location": "cpu", "strict": True}
               for _, kwargs in calls["loads"])
    loaded_model, loaded_tokenizer, kwargs = calls["predictor"]
    assert kwargs == {"device": "cpu", "max_context": context}
    assert not loaded_model.training and not loaded_tokenizer.training
    assert next(loaded_model.parameters()).dtype == torch.float32
    metadata = engine.metadata
    assert metadata["model"]["revision"] == REVISIONS[model]
    assert metadata["tokenizer"]["revision"] == REVISIONS[tokenizer]
    assert metadata["device"] == "cpu" and metadata["dtype"] == "float32"
    assert metadata["model_parameter_count"] == 2
    assert metadata["intervals_calibrated"] is False
    assert "derived_proxy" in metadata["amount_policy"]
    assert set(metadata["source"]["files"]) == {"kronos.py", "module.py", "LICENSE"}


def test_individual_paths_seed_once_restore_rng_and_new_york_wall_time(backend, local_cache):
    module, torch, _, calls, _ = backend
    engine = module.KronosEngine("mini", local_cache, threads=1)
    bars = history()
    times = future(bars)
    torch.manual_seed(987)
    before = torch.get_rng_state().clone()
    paths = engine.predict(bars, times, seed=42, samples=5)
    assert torch.equal(before, torch.get_rng_state())
    assert paths.shape == (5, 15, 6) and paths.dtype == np.float64
    assert not np.array_equal(paths[0], paths[1])
    assert np.array_equal(paths, engine.predict(bars, times, seed=42, samples=5))
    assert not np.array_equal(paths, engine.predict(bars, times, seed=43, samples=5))
    dfs, xs, ys, kwargs = calls["predictions"][0]
    assert len(dfs) == len(xs) == len(ys) == 5
    assert kwargs == {"pred_len": 15, "T": 1.0, "top_k": 0, "top_p": 0.9,
                      "sample_count": 1, "verbose": False}
    assert list(dfs[0].columns) == COLUMNS
    assert dfs[0].iloc[0]["amount"] == pytest.approx(100 * (100 + 102 + 99 + 101) / 4)
    assert xs[0].dt.tz is None and xs[0].iloc[0].hour == 9 and xs[0].iloc[0].minute == 31
    assert ys[0].dt.tz is None and ys[0].iloc[0].hour == 10 and ys[0].iloc[0].minute == 31
    assert engine.last_prediction["as_of_utc"] == bars[-1].end_time.isoformat()


@pytest.mark.parametrize("problem", ["short", "symbol", "gap", "late", "outside", "malformed"])
def test_rejects_incomplete_noncausal_or_nonregular_inputs_before_predict(backend, local_cache, problem):
    module, _, _, calls, _ = backend
    engine = module.KronosEngine("mini", local_cache)
    bars = history()
    if problem == "short":
        bars = bars[:-1]
    elif problem == "symbol":
        bars[2] = bars[2].model_copy(update={"symbol": "MSFT"})
    elif problem == "gap":
        bars[2] = bars[3]
    elif problem == "late":
        bars[2] = bars[2].model_copy(update={"available_at": bars[-1].end_time + timedelta(seconds=1)})
    elif problem == "outside":
        bars = history(datetime(2026, 10, 1, 12, 0, tzinfo=UTC))
    elif problem == "malformed":
        bars[2] = bars[2].model_copy(update={"close": Decimal("NaN")})
    with pytest.raises(ValueError):
        engine.predict(bars, future(bars))
    assert not calls["predictions"]


@pytest.mark.parametrize("problem", ["naive", "nonutc", "unordered", "gap", "past", "empty", "long", "weekend",
                                      "opening", "postclose", "fraction"])
def test_future_times_are_only_real_consecutive_utc_regular_minute_ends(backend, local_cache, problem):
    module, _, _, calls, _ = backend
    engine = module.KronosEngine("small", local_cache)
    bars = history()
    times = future(bars)
    if problem == "naive":
        times[0] = times[0].replace(tzinfo=None)
    elif problem == "nonutc":
        from zoneinfo import ZoneInfo
        times[0] = times[0].astimezone(ZoneInfo("America/New_York"))
    elif problem == "unordered":
        times[0], times[1] = times[1], times[0]
    elif problem == "gap":
        times = times[::2]
    elif problem == "past":
        times = future(bars, start=bars[-1].end_time)
    elif problem == "empty":
        times = []
    elif problem == "long":
        times = future(bars, count=61)
    elif problem == "weekend":
        times = future(bars, start=datetime(2026, 10, 3, 14, 0, tzinfo=UTC))
    elif problem == "opening":
        times = future(bars, start=datetime(2026, 10, 2, 13, 30, tzinfo=UTC))
    elif problem == "postclose":
        times = future(bars, start=datetime(2026, 10, 1, 20, 0, tzinfo=UTC))
    elif problem == "fraction":
        times = [time + timedelta(seconds=1) for time in times]
    with pytest.raises(ValueError):
        engine.predict(bars, times)
    assert not calls["predictions"]


def test_next_session_and_early_close_future_are_allowed_with_dst_features(backend, local_cache):
    module, _, _, calls, _ = backend
    engine = module.KronosEngine("mini", local_cache)
    bars = history()
    times = future(bars, start=datetime(2026, 11, 27, 14, 31, tzinfo=UTC))
    assert engine.predict(bars, times, samples=2).shape == (2, 15, 6)
    assert calls["predictions"][-1][2][0].iloc[0].hour == 9
    with pytest.raises(ValueError):
        engine.predict(bars, future(bars, start=datetime(2026, 11, 27, 18, 0, tzinfo=UTC)))


@pytest.mark.parametrize("problem", ["hash", "bytes", "revision", "repository", "filename", "schema", "duplicate",
                                      "missing", "symlink", "pickle", "tokenizer"])
def test_untrusted_cache_is_rejected_before_optional_imports(local_cache, monkeypatch, problem):
    module = adapter()
    path = local_cache / "manifest.json"
    manifest = json.loads(path.read_text())
    record = manifest["checkpoints"][0]
    if problem == "hash":
        (local_cache / record["name"] / "model.safetensors").write_bytes(b"corrupted")
    elif problem == "bytes":
        record["files"][0]["bytes"] += 1
    elif problem == "revision":
        record["revision"] = "main"
    elif problem == "repository":
        record["repository"] = "arbitrary/remote-code"
    elif problem == "filename":
        record["files"][0]["name"] = "../../config.json"
    elif problem == "schema":
        manifest["schema_version"] = "unknown"
    elif problem == "duplicate":
        manifest["checkpoints"].append(record)
    elif problem == "missing":
        (local_cache / "Kronos-mini" / "model.safetensors").unlink()
    elif problem == "symlink":
        target = local_cache / "Kronos-mini" / "model.safetensors"
        real = local_cache / "other.safetensors"
        target.rename(real)
        target.symlink_to(real)
    elif problem == "pickle":
        (local_cache / "Kronos-mini" / "pytorch_model.bin").write_bytes(b"untrusted pickle")
    elif problem == "tokenizer":
        manifest["checkpoints"] = manifest["checkpoints"][:2]
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(module.importlib, "import_module", lambda name: pytest.fail(f"premature import: {name}"))
    with pytest.raises(ValueError):
        module.KronosEngine("mini", local_cache)


def test_missing_cache_arbitrary_variant_relative_path_and_thread_limits(tmp_path):
    module = adapter()
    for variant, path, threads in (("arbitrary", tmp_path, 1), ("mini", Path("relative"), 1),
                                   ("mini", tmp_path, 0), ("mini", tmp_path, 9), ("mini", tmp_path, True),
                                   ("mini", tmp_path / "missing", 1)):
        with pytest.raises(ValueError):
            module.KronosEngine(variant, path, threads=threads)


def test_missing_optional_dependency_has_actionable_error(local_cache, monkeypatch):
    module = adapter()
    def missing(name):
        raise ModuleNotFoundError(name)
    monkeypatch.setattr(module.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match=r"kronos.*extra|extra.*kronos"):
        module.KronosEngine("mini", local_cache)


def test_checkpoint_pins_cannot_be_replaced_by_a_self_consistent_local_manifest(local_cache, monkeypatch):
    module = adapter()
    monkeypatch.setitem(module.CHECKPOINT_FILES["Kronos-mini"], "model.safetensors", (99, "0" * 64))
    with pytest.raises(ValueError):
        module.KronosEngine("mini", local_cache)


def test_vendor_code_hash_is_checked_before_import(local_cache, monkeypatch, tmp_path):
    module = adapter()
    source = tmp_path / "vendor"
    source.mkdir()
    actual = ROOT / "src/tradecopilot/_vendor/kronos"
    for filename in ("kronos.py", "module.py", "LICENSE", "UPSTREAM.json"):
        (source / filename).write_bytes((actual / filename).read_bytes())
    (source / "kronos.py").write_text("raise RuntimeError('untrusted code')")
    monkeypatch.setattr(module, "_VENDOR_DIR", source)
    monkeypatch.setattr(module.importlib, "import_module", lambda name: pytest.fail(f"premature import: {name}"))
    with pytest.raises(ValueError):
        module.KronosEngine("mini", local_cache)


@pytest.mark.parametrize("bad", ["nan", "close", "shape", "columns"])
def test_bad_outputs_fail_whole_prediction_without_repair(backend, local_cache, bad):
    module, _, _, _, fake = backend
    engine = module.KronosEngine("mini", local_cache)
    original = fake.KronosPredictor.predict_batch
    def invalid(self, *args, **kwargs):
        frames = original(self, *args, **kwargs)
        if bad == "nan":
            frames[1].iloc[0, 0] = np.nan
        elif bad == "close":
            frames[1].iloc[0, 3] = 0
        elif bad == "shape":
            frames.pop()
        elif bad == "columns":
            frames[1] = frames[1].drop(columns="amount")
        return frames
    fake.KronosPredictor.predict_batch = invalid
    with pytest.raises(ValueError):
        engine.predict(history(), future(history()), samples=3)


def test_nonphysical_rows_remain_in_paths_and_are_counted(backend, local_cache):
    module, _, _, _, fake = backend
    engine = module.KronosEngine("mini", local_cache)
    original = fake.KronosPredictor.predict_batch
    def inconsistent(self, *args, **kwargs):
        frames = original(self, *args, **kwargs)
        frames[0].iloc[0, 1] = 1
        frames[1].iloc[1, 4] = -2
        return frames
    fake.KronosPredictor.predict_batch = inconsistent
    paths = engine.predict(history(), future(history()), samples=3)
    assert paths.shape == (3, 15, 6)
    assert paths[0, 0, 1] == 1 and paths[1, 1, 4] == -2
    assert engine.diagnostics(paths)["nonphysical_ohlc_rows"] == 1
    assert engine.diagnostics(paths)["negative_volume_rows"] == 1


def test_no_labels_or_outcomes_and_invalid_sampler_inputs(backend, local_cache):
    module, _, _, calls, _ = backend
    engine = module.KronosEngine("mini", local_cache)
    bars = history()
    with pytest.raises(TypeError):
        engine.predict(bars, future(bars), labels=[100])
    with pytest.raises(TypeError):
        engine.predict(bars, future(bars), outcomes=[100])
    for seed, samples in ((True, 2), (-1, 2), (2**32, 2), (42, 0), (42, True), (42, 33)):
        with pytest.raises(ValueError):
            engine.predict(bars, future(bars), seed=seed, samples=samples)
    assert not calls["predictions"]


def test_provenance_is_stable_while_prediction_diagnostics_change(backend, local_cache):
    module, _, _, _, _ = backend
    engine = module.KronosEngine("mini", local_cache)
    provenance = json.dumps(engine.metadata, sort_keys=True)
    engine.predict(history(), future(history()), samples=1, seed=2**32 - 1)
    assert json.dumps(engine.metadata, sort_keys=True) == provenance
    assert engine.last_prediction["samples"] == 1
    engine.predict(history(), future(history(), count=2), samples=32, seed=0)
    assert json.dumps(engine.metadata, sort_keys=True) == provenance
    assert engine.last_prediction["samples"] == 32 and engine.last_prediction["future_minutes"] == 2


def test_native_predictor_normalization_and_one_path_per_series(backend, local_cache, monkeypatch):
    module, torch, _, calls, fake = backend
    native = __import__("tradecopilot._vendor.kronos.kronos", fromlist=["KronosPredictor"])
    generated = []

    def generate(self, x, x_stamp, y_stamp, pred_len, temperature, top_k, top_p, sample_count, verbose):
        generated.append((x.copy(), x_stamp.copy(), y_stamp.copy(), sample_count, self.device))
        return torch.rand((x.shape[0], pred_len, 6)).numpy()

    monkeypatch.setattr(native.KronosPredictor, "generate", generate)
    fake.KronosPredictor = native.KronosPredictor
    engine = module.KronosEngine("small", local_cache)
    paths = engine.predict(history(), future(history()), samples=4)
    assert len(calls["loads"]) == 2 and paths.shape == (4, 15, 6)
    x, x_stamp, y_stamp, sample_count, device = generated[0]
    assert x.shape == (4, 60, 6) and x.dtype == np.float32
    assert x_stamp[0, 0].tolist() == [31, 9, 3, 1, 10]
    assert y_stamp[0, 0].tolist() == [31, 10, 3, 1, 10]
    assert sample_count == 1 and device == "cpu"
    assert not np.array_equal(paths[0], paths[1])


def test_predictor_failure_restores_cpu_rng(backend, local_cache):
    module, torch, _, _, fake = backend
    engine = module.KronosEngine("mini", local_cache)
    def fail(self, *args, **kwargs):
        torch.rand(3)
        raise RuntimeError("synthetic model failure")
    fake.KronosPredictor.predict_batch = fail
    before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="synthetic model failure"):
        engine.predict(history(), future(history()))
    assert torch.equal(before, torch.get_rng_state())


def setup_script():
    path = ROOT / "scripts/setup_kronos.py"
    assert path.is_file(), "explicit pinned Kronos setup script is missing"
    return runpy.run_path(str(path))["download_checkpoints"]


def test_setup_downloads_only_pinned_safe_files_and_reuses_immutable_manifest(local_cache, tmp_path, monkeypatch):
    download = setup_script()
    calls = []
    def fetch(**kwargs):
        calls.append(kwargs)
        name = kwargs["repo_id"].split("/")[1]
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((local_cache / name / kwargs["filename"]).read_bytes())
        return str(path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=fetch))
    output = tmp_path / "new-checkpoints"
    manifest_path = download(output, variants=("mini",))
    before = manifest_path.read_bytes()
    assert len(calls) == 4
    assert all(call["token"] is False and call["local_files_only"] is False for call in calls)
    assert all(call["filename"] in ("config.json", "model.safetensors") for call in calls)
    assert all(call["revision"] == REVISIONS[call["repo_id"].split("/")[1]] for call in calls)
    records = json.loads(before)["checkpoints"]
    assert [record["name"] for record in records] == ["Kronos-mini", "Kronos-Tokenizer-2k"]
    assert download(output, variants=("mini",)) == manifest_path
    assert manifest_path.read_bytes() == before and len(calls) == 4
    (output / "Kronos-mini" / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        download(output, variants=("mini",))
    assert len(calls) == 4


def test_setup_rejects_relative_destination_or_unknown_variant_before_backend_import(tmp_path, monkeypatch):
    download = setup_script()
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    for path, variants in ((Path("relative"), ("mini",)), (tmp_path, ("arbitrary",))):
        with pytest.raises(ValueError):
            download(path, variants=variants)
