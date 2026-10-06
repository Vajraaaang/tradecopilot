"""Execute a fixed four-session integration pilot and preserve pending paper forecasts."""

from __future__ import annotations

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
from tradecopilot.forecast.kronos_report import write_report
from tradecopilot.forecast.kronos_study import baseline_paths, build_cases, future_grid, price_metrics

DATES = tuple(date(2026, 10, d) for d in (1, 2, 5, 6))


def _encode(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def _forecast(paths: NDArray[np.float64] | None, elapsed: float, error: str | None = None) -> dict[str, Any]:
    if paths is None:
        return {"status": "error", "error": error, "elapsed_seconds": elapsed}
    close = paths[:, :, 3]
    return {
        "status": "ok",
        "mean_close": close.mean(axis=0).tolist(),
        "lower_close": np.quantile(close, 0.1, axis=0).tolist(),
        "upper_close": np.quantile(close, 0.9, axis=0).tolist(),
        "samples": len(paths),
        "intervals_calibrated": False,
        "elapsed_seconds": elapsed,
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
    if output_dir.exists():
        raise FileExistsError("choose a new immutable Kronos pilot directory")
    if (
        not all(p.is_absolute() for p in (bars_dir, cache_dir, output_dir))
        or type(samples) is not int
        or not 1 <= samples <= 32
    ):
        raise ValueError("absolute paths and 1-32 samples are required")
    if engine_factory is None:
        from tradecopilot.forecast.kronos import KronosEngine

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
    }
    protocol["protocol_id"] = content_hash(protocol)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir.parent / (output_dir.name + "-registration.json")
    with protocol_path.open("xb") as stream:
        stream.write(_encode(protocol))
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
    for variant in ("mini", "small"):
        engine = engine_factory(variant, cache_dir, threads=1)
        engines[variant] = engine
        stable_meta = json.loads(_encode(engine.metadata))
        predictions[variant] = {}
        started = time.monotonic()
        failures = []
        for index, c in enumerate(cases):
            progress_path.write_bytes(
                _encode({"stage": "inference", "model": variant, "case": index + 1, "cases": len(cases)})
            )
            paths, reason = None, None
            t0 = time.monotonic()
            try:
                if t0 - started >= 900:
                    raise TimeoutError("registered model compute budget exhausted")
                paths = engine.predict(c.history, c.future_times, seed=42 + index, samples=samples)
                # Independent shape/value scoring validator runs without outcome-dependent selection.
                if paths.shape != (samples, 15, 6) or not np.isfinite(paths).all() or (paths[:, :, 3] <= 0).any():
                    raise ValueError("invalid paths")
            except Exception as exc:
                reason = type(exc).__name__
                failures.append({"case_id": c.case_id, "error_type": reason})
                paths = None
            predictions[variant][c.case_id] = paths
            rows[index]["forecasts"][variant] = _forecast(paths, time.monotonic() - t0, reason)
            if paths is not None:
                raw_paths[f"{variant}_{index}"] = paths
        if json.loads(_encode(engine.metadata)) != stable_meta:
            raise ValueError("model provenance changed during inference")
        models[variant] = {
            "metadata": stable_meta,
            "elapsed_seconds": time.monotonic() - started,
            "failures": failures,
            "metrics": price_metrics(cases, predictions[variant], sample_intervals=True),
        }
    for name in ("persistence", "momentum"):
        predictions[name] = {}
        for index, c in enumerate(cases):
            p = baseline_paths(c.history, 15)[name]
            predictions[name][c.case_id] = p
            rows[index]["forecasts"][name] = _forecast(p, 0)
        models[name] = {
            "metadata": {"kind": "fixed_past_only_control"},
            "metrics": price_metrics(cases, predictions[name], sample_intervals=False),
        }
    if prospective:
        recorded_at = datetime.now(UTC)
        for symbol in symbols:
            history = tuple(sorted((b for b in bars if b.symbol == symbol), key=lambda b: b.end_time)[-60:])
            future, kind = future_grid(history, metadata["clock"], recorded_at)
            row = {
                "case_id": content_hash({"history": [b.bar_id for b in history], "future": future}),
                "group": "prospective",
                "symbol": symbol,
                "as_of": history[-1].end_time.isoformat(),
                "future_times": [t.isoformat() for t in future],
                "forecast_kind": kind,
                "history_close": [float(b.close) for b in history],
                "actual_close": None,
                "forecasts": {},
            }
            for variant, engine in engines.items():
                t0 = time.monotonic()
                try:
                    p = engine.predict(history, future, seed=42, samples=samples)
                    generated_at = datetime.now(UTC)
                    if generated_at >= future[-1]:
                        raise ValueError("forecast target expired before publication")
                    row["forecasts"][variant] = _forecast(p, time.monotonic() - t0) | {
                        "generated_at": generated_at.isoformat(),
                        "outcome_status": "pending",
                    }
                    raw_paths[f"pending_{variant}_{symbol}"] = p
                except Exception as exc:
                    row["forecasts"][variant] = _forecast(None, time.monotonic() - t0, type(exc).__name__)
            rows.append(row)
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
        "broker_orders": 0,
        "jev_calls": 0,
        "limitations": [
            "Four-session development pilot is not confirmation.",
            "Raw sample bands are uncalibrated.",
            "Historical availability is a bar-end assumption.",
            "Kronos pretraining overlap is not ruled out.",
            "No order policy or profitability claim.",
        ],
    }
    progress_path.write_bytes(_encode({"stage": "complete", "cases": len(cases)}))
    return write_report(
        output_dir,
        report,
        {
            "protocol.json": _encode(protocol),
            "catalog.json": _encode(catalog),
            "source.json": _encode(source),
            "paths.npz": buffer.getvalue(),
        },
    )
