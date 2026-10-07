"""Execute a fixed four-session integration pilot and preserve pending paper forecasts."""

from __future__ import annotations

import hashlib
import io
import json
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tradecopilot.forecast.bars import load_bar_dataset
from tradecopilot.forecast.contracts import content_hash
from tradecopilot.forecast.experiment import source_provenance
from tradecopilot.forecast.kronos_report import validate_prospective_grid, write_report
from tradecopilot.forecast.kronos_study import baseline_paths, build_cases, future_grid, price_metrics

DATES = tuple(date(2026, 10, d) for d in (1, 2, 5, 6))


def _encode(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def _validate_paths(value: Any, samples: int) -> NDArray[np.float64]:
    paths = np.asarray(value, dtype=np.float64)
    if paths.shape != (samples, 15, 6) or not np.isfinite(paths).all() or (paths[:, :, 3] <= 0).any():
        raise ValueError("invalid forecast paths")
    return paths


def _progress(path: Path, stage: str, **details: Any) -> None:
    path.write_bytes(_encode({"stage": stage, **details}))
    path.chmod(0o600)


def _forecast(paths: NDArray[np.float64] | None, elapsed: float, error: str | None = None) -> dict[str, Any]:
    if paths is None:
        return {"status": "error", "error": error, "elapsed_seconds": elapsed}
    from tradecopilot.forecast.kronos import KronosEngine

    close = paths[:, :, 3]
    return {
        "status": "ok",
        "mean_close": close.mean(axis=0).tolist(),
        "lower_close": np.quantile(close, 0.1, axis=0).tolist(),
        "upper_close": np.quantile(close, 0.9, axis=0).tolist(),
        "samples": len(paths),
        "intervals_calibrated": False,
        "elapsed_seconds": elapsed,
        "diagnostics": KronosEngine.diagnostics(paths),
    }


def run_pilot(
    bars_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    *,
    samples: int = 20,
    engine_factory: Callable[..., Any] | None = None,
    prospective: bool = True,
) -> Path:
    run_started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    if output_dir.exists():
        raise FileExistsError("choose a new immutable Kronos pilot directory")
    if (
        not all(p.is_absolute() for p in (bars_dir, cache_dir, output_dir))
        or type(samples) is not int
        or not 1 <= samples <= 32
    ):
        raise ValueError("absolute paths and 1-32 samples are required")
    from tradecopilot.forecast.kronos import KronosEngine

    native_engine = engine_factory is None or engine_factory is KronosEngine
    if engine_factory is None:
        engine_factory = KronosEngine
    bars, source = load_bar_dataset(bars_dir)
    metadata = source["source_metadata"]
    if metadata.get("paper_endpoint") != "https://paper-api.alpaca.markets" or metadata.get("feed") not in (
        "iex",
        "sip",
    ):
        raise ValueError("verified paper-account candle source is required")
    symbols = tuple(metadata["selected_symbols"])
    cases, catalog = build_cases(bars, symbols, DATES)
    if not cases:
        raise ValueError("no complete cases in the fixed pilot")
    implementation = source_provenance()
    lock_path = Path(__file__).resolve().parents[3] / "uv.lock"
    protocol = {
        "schema_version": "kronos-pilot-protocol-v1",
        "registered_at": datetime.now(UTC).isoformat(),
        "dates": [d.isoformat() for d in DATES],
        "symbols": symbols,
        "lookback": 60,
        "horizon": 15,
        "offsets": [61, 121, 181, 241, 301],
        "samples": samples,
        "seed": 42,
        "flat_bps": 10,
        "max_seconds_per_model": 900,
        "source_data_id": source["data_id"],
        "case_ids_hash": content_hash([c.case_id for c in cases]),
        "fine_tuning": False,
        "evidence": "retrospective_development_pilot",
        "broker_orders": 0,
        "jev_calls": 0,
        "prospectively_requested": prospective,
        "implementation": implementation,
        "dependency_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest() if lock_path.is_file() else None,
    }
    protocol["protocol_id"] = content_hash(protocol)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir.parent / (output_dir.name + "-registration.json")
    with protocol_path.open("xb") as stream:
        stream.write(_encode(protocol))
    protocol_path.chmod(0o600)
    rows: list[dict[str, Any]] = [
        {
            "case_id": c.case_id,
            "group": "historical",
            "symbol": c.symbol,
            "as_of": c.as_of.isoformat(),
            "future_times": [t.isoformat() for t in c.future_times],
            "forecast_kind": "intraday",
            "history_close": [float(b.close) for b in c.history],
            "actual_close": [float(b.close) for b in c.actual],
            "forecasts": {},
        }
        for c in cases
    ]
    models: dict[str, dict[str, Any]] = {}
    raw_paths: dict[str, Any] = {}
    predictions: dict[str, dict[str, NDArray[np.float64] | None]] = {}
    engines: dict[str, Any] = {}
    progress_path = output_dir.parent / (output_dir.name + "-progress.json")
    preparation_seconds = time.monotonic() - run_started
    for variant in ("mini", "small"):
        _progress(progress_path, "initialization", model=variant)
        initialization_started = time.monotonic()
        initialization_error = None
        engine = None
        try:
            assert engine_factory is not None
            engine = engine_factory(variant, cache_dir, threads=1)
            stable_meta = json.loads(_encode(engine.metadata))
            if time.monotonic() - initialization_started >= 900:
                raise TimeoutError("registered model compute budget exhausted")
            engines[variant] = engine
        except Exception as exc:
            # Native cache/source validation remains a fatal integrity failure.
            if native_engine and isinstance(exc, ValueError):
                _progress(progress_path, "failed", phase="initialization", model=variant,
                          error_type=type(exc).__name__)
                raise
            initialization_error = type(exc).__name__
            engine = None
            stable_meta = {"variant": variant, "initialization_status": "error"}
        initialization_seconds = time.monotonic() - initialization_started
        predictions[variant] = {}
        started = time.monotonic()
        failures = []
        for index, c in enumerate(cases):
            _progress(progress_path, "inference", model=variant, case=index + 1, cases=len(cases))
            paths, reason = None, initialization_error
            phase = "initialization" if initialization_error else "historical"
            t0 = time.monotonic()
            if reason is None:
                try:
                    assert engine is not None
                    if initialization_seconds + t0 - started >= 900:
                        raise TimeoutError("registered model compute budget exhausted")
                    paths = _validate_paths(engine.predict(c.history, c.future_times, seed=42 + index, samples=samples),
                                            samples)
                    if initialization_seconds + time.monotonic() - started >= 900:
                        raise TimeoutError("registered model compute budget exhausted")
                except Exception as exc:
                    reason = type(exc).__name__
                    paths = None
            if reason is not None:
                failures.append({"case_id": c.case_id, "error_type": reason, "phase": phase})
            predictions[variant][c.case_id] = paths
            rows[index]["forecasts"][variant] = _forecast(paths, time.monotonic() - t0, reason)
            if reason is not None:
                rows[index]["forecasts"][variant]["failure_phase"] = phase
            if paths is not None:
                raw_paths[f"{variant}_{index}"] = paths
        if engine is not None and json.loads(_encode(engine.metadata)) != stable_meta:
            _progress(progress_path, "failed", phase="historical_provenance", model=variant, error_type="ValueError")
            raise ValueError("model provenance changed during inference")
        metrics = price_metrics(cases, predictions[variant], sample_intervals=True)
        diagnostics = {
            key: sum(row["forecasts"][variant].get("diagnostics", {}).get(key, 0) for row in rows)
            for key in ("paths", "rows", "nonphysical_ohlc_rows", "negative_volume_rows", "negative_amount_rows")
        }
        historical_seconds = time.monotonic() - started
        models[variant] = {
            "metadata": stable_meta,
            "initialization": {"status": "error" if initialization_error else "ok",
                               "error_type": initialization_error, "phase": "initialization"},
            "timing": {"initialization_seconds": initialization_seconds, "historical_seconds": historical_seconds,
                       "prospective_seconds": 0.0},
            "elapsed_seconds": initialization_seconds + historical_seconds,
            "failures": failures,
            "metrics": metrics,
            "diagnostics": diagnostics,
        }
    controls_started = time.monotonic()
    for name in ("persistence", "momentum"):
        predictions[name] = {}
        failures = []
        for index, c in enumerate(cases):
            reason = None
            try:
                p = baseline_paths(c.history, 15, control=name)[name]
            except ValueError as exc:
                p = None
                reason = type(exc).__name__
                failures.append({"case_id": c.case_id, "error_type": reason})
            predictions[name][c.case_id] = p
            rows[index]["forecasts"][name] = _forecast(p, 0, reason)
        models[name] = {
            "metadata": {"kind": "fixed_past_only_control"},
            "metrics": price_metrics(cases, predictions[name], sample_intervals=False),
            "failures": failures,
        }
    controls_seconds = time.monotonic() - controls_started
    prospective_unavailable = []
    if prospective:
        recorded_at = datetime.now(UTC)
        for symbol in symbols:
            history = tuple(sorted((b for b in bars if b.symbol == symbol), key=lambda b: b.end_time)[-60:])
            try:
                future, kind = future_grid(history, metadata["clock"], recorded_at)
                validate_prospective_grid([t.isoformat() for t in future])
            except ValueError as exc:
                prospective_unavailable.append({"symbol": symbol, "reason": str(exc)})
                continue
            row = {
                "case_id": content_hash({"history": [b.bar_id for b in history], "future": future}),
                "group": "prospective",
                "symbol": symbol,
                "as_of": history[-1].end_time.isoformat(),
                "future_times": [t.isoformat() for t in future],
                "forecast_kind": kind,
                "history_close": [float(b.close) for b in history],
                "actual_close": None,
                "outcome_status": "pending",
                "forecasts": {},
            }
            for variant in ("mini", "small"):
                model = models[variant]
                engine = engines.get(variant)
                t0 = time.monotonic()
                reason = model["initialization"]["error_type"]
                phase = "initialization" if reason else "prospective"
                if reason is None:
                    try:
                        assert engine is not None
                        _progress(progress_path, "prospective", model=variant, symbol=symbol)
                        if model["elapsed_seconds"] >= 900:
                            raise TimeoutError("registered model compute budget exhausted")
                        p = _validate_paths(engine.predict(history, future, seed=42, samples=samples), samples)
                        if model["elapsed_seconds"] + time.monotonic() - t0 >= 900:
                            raise TimeoutError("registered model compute budget exhausted")
                        generated_at = datetime.now(UTC)
                        if generated_at >= future[0]:
                            raise ValueError("forecast first target expired before publication")
                        row["forecasts"][variant] = _forecast(p, time.monotonic() - t0) | {
                            "generated_at": generated_at.isoformat(),
                            "outcome_status": "pending",
                        }
                        raw_paths[f"pending_{variant}_{symbol}"] = p
                    except Exception as exc:
                        reason = type(exc).__name__
                elapsed = time.monotonic() - t0
                model["timing"]["prospective_seconds"] += elapsed
                model["elapsed_seconds"] += elapsed
                if reason is not None:
                    row["forecasts"][variant] = _forecast(None, elapsed, reason) | {"failure_phase": phase}
                    model["failures"].append({"case_id": row["case_id"], "error_type": reason, "phase": phase})
                if engine is not None and json.loads(_encode(engine.metadata)) != model["metadata"]:
                    _progress(progress_path, "failed", phase="prospective_provenance", model=variant,
                              error_type="ValueError")
                    raise ValueError("model provenance changed during prospective inference")
            rows.append(row)
    if source_provenance() != implementation:
        _progress(progress_path, "failed", phase="source_provenance", error_type="ValueError")
        raise ValueError("implementation changed during inference")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **raw_paths)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "evidence_mode": "synthetic_contract_fixture" if metadata.get("fixture") else "retrospective_development_pilot",
        "connection": {
            "mode": "paper",
            "account": metadata["paper_account"],
            "clock": metadata["clock"],
            "feed": metadata["feed"],
        },
        "source_data_id": source["data_id"],
        "protocol": protocol,
        "catalog_counts": {"planned": len(catalog), "eligible": len(cases), "excluded": len(catalog) - len(cases)},
        "models": models,
        "cases": rows,
        "prospective_unavailable": prospective_unavailable,
        "broker_orders": 0,
        "jev_calls": 0,
        "limitations": [
            "Four-session development pilot is not confirmation.",
            "Raw sample bands are uncalibrated.",
            "Historical availability is a bar-end assumption.",
            "Kronos pretraining overlap is not ruled out.",
            "No order policy or profitability claim.",
        ],
        "timing": {
            "started_at": started_at,
            "preparation_seconds": preparation_seconds,
            "controls_seconds": controls_seconds,
            "elapsed_seconds_before_publication": time.monotonic() - run_started,
            "budget_enforcement": "admission_and_post_return; blocking_calls_not_cancelled",
            "measurement_scope": "prepublication; complete timing in external progress",
        },
    }
    _progress(progress_path, "publication", cases=len(cases))
    publication_started = time.monotonic()
    try:
        path = write_report(
            output_dir, report,
            {"protocol.json": _encode(protocol), "catalog.json": _encode(catalog),
             "source.json": _encode(source), "paths.npz": buffer.getvalue()},
        )
    except Exception as exc:
        _progress(progress_path, "failed", phase="publication", error_type=type(exc).__name__,
                  started_at=started_at, finished_at=datetime.now(UTC).isoformat(),
                  elapsed_seconds=time.monotonic() - run_started)
        raise
    _progress(progress_path, "complete", cases=len(cases), started_at=started_at,
              finished_at=datetime.now(UTC).isoformat(), elapsed_seconds=time.monotonic() - run_started,
              timing=report["timing"] | {"publication_seconds": time.monotonic() - publication_started})
    return path
