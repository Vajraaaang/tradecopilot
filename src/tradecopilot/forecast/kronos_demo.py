"""Deterministic synthetic contract fixtures for the existing read-only Kronos viewer."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SYMBOLS = ("DEMOA", "DEMOB")
ANCHORS = (datetime(2026, 1, 5, 14, 30, tzinfo=UTC), datetime(2026, 1, 5, 15, 30, tzinfo=UTC))
LOOKBACK, HORIZON, SAMPLES = 60, 15, 5


def _encode(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def _inputs(symbol_index: int, anchor_index: int) -> dict[str, Any]:
    anchor = ANCHORS[anchor_index]
    base, drift = ((100.0, 0.03), (160.0, -0.025))[symbol_index]
    base += 2 * anchor_index
    phase = symbol_index + anchor_index / 2
    values = [
        round(base + drift * i + 0.6 * math.sin(i / 7 + phase) + 0.2 * math.cos(i / 3 + phase), 6)
        for i in range(LOOKBACK + HORIZON)
    ]
    paths = [
        [round(values[LOOKBACK - 1] + drift * k + 0.14 * math.sin(k / 4 + phase)
               + (sample - 2) * 0.05 * math.sqrt(k), 6) for k in range(1, HORIZON + 1)]
        for sample in range(SAMPLES)
    ]
    return {
        "symbol": SYMBOLS[symbol_index],
        "as_of": anchor.isoformat(),
        "history_times": [(anchor + timedelta(minutes=i - LOOKBACK + 1)).isoformat() for i in range(LOOKBACK)],
        "future_times": [(anchor + timedelta(minutes=i)).isoformat() for i in range(1, HORIZON + 1)],
        "history_close": values[:LOOKBACK],
        "actual_close": values[LOOKBACK:],
        "toy_paths": paths,
    }


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (position - lower)


def write_demo(output_dir: Path) -> Path:
    """Seal four fictional cases without market data, credentials or model inference."""
    from tradecopilot.forecast.contracts import content_hash
    from tradecopilot.forecast.kronos_report import write_report

    fixture: dict[str, Any] = {
        "schema_version": "kronos-synthetic-fixture-v1",
        "generator": {
            "name": "deterministic_local_math", "version": 1,
            "symbol_bases": [100.0, 160.0], "symbol_drifts": [0.03, -0.025],
            "case_base": "symbol_base + 2 * anchor_index",
            "phase": "symbol_index + anchor_index / 2",
            "series_formula": "base + drift*i + 0.6*sin(i/7+phase) + 0.2*cos(i/3+phase)",
            "toy_path_formula": "anchor_value + drift*k + 0.14*sin(k/4+phase) + (sample-2)*0.05*sqrt(k)",
            "round_decimal_places": 6,
            "lookback": LOOKBACK, "horizon": HORIZON, "samples": SAMPLES,
            "symbols": list(SYMBOLS), "anchors": [anchor.isoformat() for anchor in ANCHORS],
        },
        "cases": [_inputs(symbol, anchor) for symbol in range(len(SYMBOLS)) for anchor in range(len(ANCHORS))],
    }
    protocol = {
        "schema_version": "kronos-offline-demo-protocol-v1", "fixture": True,
        "lookback": LOOKBACK, "horizon": HORIZON, "samples": SAMPLES, "flat_bps": 10,
        "symbols": list(SYMBOLS), "anchors": [anchor.isoformat() for anchor in ANCHORS],
        "source_data_id": content_hash(fixture), "evaluated": False,
    }
    protocol["protocol_id"] = content_hash(protocol)
    generated = datetime.now(UTC).isoformat()
    cases = []
    for inputs in fixture["cases"]:
        paths = inputs["toy_paths"]
        columns = [list(values) for values in zip(*paths, strict=True)]
        common = {"status": "ok", "generated_at": generated, "intervals_calibrated": False}
        cases.append({
            **{key: value for key, value in inputs.items() if key != "toy_paths"},
            "case_id": content_hash(inputs), "group": "historical", "synthetic": True,
            "forecast_kind": "synthetic_path_fixture", "outcome_status": "simulated",
            "forecasts": {
                "synthetic_fixture": common | {
                    "execution": "fixture", "samples": SAMPLES,
                    "mean_close": [sum(values) / SAMPLES for values in columns],
                    "lower_close": [_quantile(values, 0.1) for values in columns],
                    "upper_close": [_quantile(values, 0.9) for values in columns],
                },
                "persistence": common | {
                    "execution": "local_arithmetic", "samples": 1,
                    "mean_close": [inputs["history_close"][-1]] * HORIZON,
                    "lower_close": None, "upper_close": None,
                },
            },
        })
    metrics = {
        "evaluated": False, "eligible": len(cases), "scored": 0, "errors": 0,
        "terminal_mae_bps": None, "path_mae_bps": None, "direction_accuracy": None,
        "direction_accuracy_errors_as_incorrect": None, "raw_interval_coverage": None,
    }
    report = {
        "generated_at": generated, "evidence_mode": "synthetic_contract_fixture",
        "connection": {"mode": "synthetic"}, "source_data_id": protocol["source_data_id"],
        "protocol": protocol, "catalog_counts": {"planned": len(cases), "eligible": len(cases), "excluded": 0},
        "models": {
            "synthetic_fixture": {
                "execution": "fixture", "metrics": dict(metrics),
                "metadata": {"kind": "synthetic_fixture", "execution": "fixture",
                             "model": {"name": "Synthetic path fixture"}},
            },
            "persistence": {
                "execution": "local_arithmetic", "metrics": dict(metrics),
                "metadata": {"kind": "fixed_past_only_control", "execution": "local_arithmetic",
                             "model": {"name": "Persistence (synthetic context)"}},
            },
        },
        "cases": cases, "prospective_unavailable": [],
        "provider_calls": 0, "model_inference_calls": 0, "broker_orders": 0, "jev_calls": 0,
        "limitations": [
            "Offline synthetic contract fixture: DEMOA and DEMOB are fictional symbols.",
            "All context, target values and toy paths come from deterministic local math; no market data is used.",
            "No broker connection, pretrained model execution, fitting or downloads.",
            "Toy paths and the persistence calculation are not evaluated for forecast quality.",
            "No prospective forecasts, calibrated intervals or profitability claim.",
        ],
    }
    return write_report(output_dir, report, {"fixture.json": _encode(fixture), "protocol.json": _encode(protocol)})
