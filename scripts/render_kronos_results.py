"""Publish only aggregate Kronos pilot metrics from a verified private bundle."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from tradecopilot.forecast.kronos_report import load_report

Validator = Callable[[Any], Any]
_LIMITATIONS = (
    "Four-session development pilot is not confirmation.",
    "Raw sample bands are uncalibrated.",
    "Historical availability is a bar-end assumption.",
    "Kronos pretraining overlap is not ruled out.",
    "No order policy or profitability claim.",
)
_UNAVAILABLE_REASONS = (
    "history and a verified paper market clock are required", "context has not completed", "intraday context is stale",
    "intraday anchor is outside the session", "intraday horizon extends beyond session close",
    "next session is not a future XNYS opening", "invalid prospective target grid",
)


def _project(value: Any, fields: Mapping[str, Validator], required: Sequence[str] = ()) -> dict[str, Any]:
    if not isinstance(value, dict) or not set(required) <= value.keys():
        raise ValueError("invalid public summary fields")
    return {key: check(value[key]) for key, check in fields.items() if key in value}


def _integer(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("invalid public summary count")
    return value


def _number(value: Any) -> int | float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid public summary number")
    return value


def _probability(value: Any) -> int | float:
    result = _number(value)
    if result > 1:
        raise ValueError("invalid public summary probability")
    return result


def _boolean(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError("invalid public summary flag")
    return value


def _enum(*choices: str) -> Validator:
    def check(value: Any) -> str:
        if not isinstance(value, str) or value not in choices:
            raise ValueError("invalid public summary label")
        return value
    return check


def _pattern(pattern: str) -> Validator:
    def check(value: Any) -> str:
        if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
            raise ValueError("invalid public summary identifier")
        return value
    return check


_HASH = _pattern(r"[0-9a-f]{64}")
_REVISION = _pattern(r"[0-9a-f]{40}")
_SYMBOL = _pattern(r"[A-Z][A-Z0-9.-]{0,9}")
_VERSION = _pattern(r"[0-9]{1,4}(?:\.[0-9]{1,4}){1,3}(?:(?:a|b|rc|post|dev)[0-9]{1,4})?(?:\+(?:cpu|cu[0-9]{2,3}))?")


def _optional(check: Validator) -> Validator:
    return lambda value: None if value is None else check(value)


def _list(value: Any, check: Validator) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError("invalid public summary list")
    return [check(item) for item in value]


def _date(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("invalid public summary date")
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        raise ValueError("invalid public summary date") from None


def _timestamp(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("invalid public summary timestamp")
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError("invalid public summary timestamp") from None
    if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
        raise ValueError("invalid public summary timestamp")
    return timestamp.isoformat()


def _dependencies(value: Any, names: Sequence[str]) -> dict[str, Any]:
    return _project(value, {name: _VERSION for name in names})


def _file(value: Any, *names: str) -> dict[str, Any]:
    return _project(value, {"name": _enum(*names), "bytes": _integer, "sha256": _HASH}, ("name", "bytes", "sha256"))


def _checkpoint(value: Any, *names: str) -> dict[str, Any]:
    result = _project(value, {
        "name": _enum(*names), "repository": _enum(*(f"NeoQuasar/{name}" for name in names)), "revision": _REVISION,
        "files": lambda items: _list(items, lambda item: _file(item, "config.json", "model.safetensors")),
    }, ("name", "repository", "revision", "files"))
    if result["repository"] != f"NeoQuasar/{result['name']}" or sorted(f["name"] for f in result["files"]) != [
        "config.json", "model.safetensors",
    ]:
        raise ValueError("invalid public checkpoint pins")
    return result


def _source(value: Any) -> dict[str, Any]:
    names = ("LICENSE", "kronos.py", "module.py")
    return _project(value, {
        "repository": _enum("https://github.com/shiyu-coder/Kronos"), "commit": _REVISION, "license": _enum("MIT"),
        "files": lambda items: _project(items, {
            "LICENSE": lambda item: _file(item, "LICENSE"), "kronos.py": lambda item: _file(item, "kronos.py"),
            "module.py": lambda item: _file(item, "module.py"),
        }, names),
    }, ("repository", "commit", "license", "files"))


def _metadata(value: Any) -> dict[str, Any]:
    return _project(value, {
        "schema_version": _enum("kronos-local-engine-v1"), "variant": _enum("mini", "small"),
        "kind": _enum("fixed_past_only_control"), "initialization_status": _enum("ok", "error"),
        "model": lambda item: _checkpoint(item, "Kronos-mini", "Kronos-small"),
        "tokenizer": lambda item: _checkpoint(item, "Kronos-Tokenizer-2k", "Kronos-Tokenizer-base"),
        "source": _source, "device": _enum("cpu"), "dtype": _enum("float32"), "threads": _integer,
        "max_context": _integer, "model_parameter_count": _integer, "tokenizer_parameter_count": _integer,
        "input_bars": _integer, "calendar": _enum("XNYS"), "artifact_timezone": _enum("UTC"),
        "model_timezone": _enum("America/New_York"), "timestamp_semantics": _enum("completed minute bar ends"),
        "columns": _enum_columns,
        "dependencies": lambda item: _dependencies(item, (
            "torch", "pandas", "huggingface-hub", "safetensors", "einops",
        )),
        "sampler": lambda item: _project(item, {
            "temperature": _number, "top_p": _probability, "top_k": _integer, "sample_count_per_series": _integer,
            "paths": _enum("individual duplicated-input batch series"), "seed_policy": _enum("one CPU seed per call"),
        }),
        "amount_policy": _enum("derived_proxy: volume * mean(open,high,low,close); not provider dollar volume"),
        "fine_tuned": _boolean, "intervals_calibrated": _boolean,
    })


def _enum_columns(value: Any) -> list[str]:
    expected = ["open", "high", "low", "close", "volume", "amount"]
    if value != expected:
        raise ValueError("invalid public model columns")
    return expected


def _classes(value: Any) -> list[str]:
    if value != ["DOWN", "FLAT", "UP"]:
        raise ValueError("invalid public metric classes")
    return ["DOWN", "FLAT", "UP"]


def _confusion(value: Any) -> list[list[int]]:
    rows = _list(value, lambda row: _list(row, _integer))
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("invalid public confusion matrix")
    return rows


def _metrics(value: Any) -> dict[str, Any]:
    result = _project(value, {
        "eligible": _integer, "scored": _integer, "errors": _integer, "forecast_availability": _probability,
        "terminal_mae_bps": _optional(_number), "path_mae_bps": _optional(_number),
        "direction_accuracy": _optional(_probability),
        "direction_accuracy_errors_as_incorrect": _optional(_probability),
        "raw_interval_coverage": _optional(_probability), "interval_nominal_mass": _optional(_probability),
        "intervals_calibrated": _boolean, "uncertainty_interval": _optional(_enum()),
        "class_order": _classes, "confusion_matrix": _confusion,
        "evidence": _enum("retrospective_development_pilot; no confirmation or promotion"),
    }, ("eligible", "scored", "errors"))
    if result["scored"] + result["errors"] != result["eligible"] or (
        "confusion_matrix" in result and sum(map(sum, result["confusion_matrix"])) != result["scored"]
    ):
        raise ValueError("invalid public metric cohort counts")
    return result


def _protocol(value: Any) -> dict[str, Any]:
    return _project(value, {
        "schema_version": _enum("kronos-pilot-protocol-v1"), "registered_at": _timestamp,
        "dates": lambda items: _list(items, _date), "symbols": lambda items: _list(items, _SYMBOL),
        "lookback": _integer, "horizon": _integer, "offsets": lambda items: _list(items, _integer),
        "samples": _integer, "seed": _integer, "flat_bps": _number, "max_seconds_per_model": _number,
        "source_data_id": _HASH, "case_ids_hash": _HASH, "protocol_id": _HASH,
        "dependency_lock_sha256": _optional(_HASH), "fine_tuning": _boolean, "prospectively_requested": _boolean,
        "broker_orders": _integer, "jev_calls": _integer, "evidence": _enum("retrospective_development_pilot"),
        "implementation": lambda item: _project(item, {
            "source_sha256": _HASH, "package_version": _VERSION, "python": _VERSION,
            "dependencies": lambda item: _dependencies(item, (
                "numpy", "scikit-learn", "exchange-calendars", "pydantic",
            )),
        }),
    }, ("dates", "symbols", "samples"))


def _label(value: Any, known: Sequence[str], unknown: str) -> str:
    if not isinstance(value, str):
        raise ValueError("invalid private reason/limitation label")
    return value if value in known else unknown


def _error_type(value: Any) -> str:
    return _label(value, ("RuntimeError", "ValueError", "TypeError", "TimeoutError", "MemoryError", "OSError",
                          "ImportError", "ModuleNotFoundError", "FileNotFoundError", "PermissionError",
                          "IndexError", "OverflowError", "KeyError", "AssertionError"), "OtherError")


def public_summary(path: Path) -> dict[str, Any]:
    report = load_report(path)
    if report["evidence_mode"] != "retrospective_development_pilot":
        raise ValueError("real development evidence is required")
    models = {}
    for name, model in report["models"].items():
        _enum("mini", "small", "persistence", "momentum")(name)
        models[name] = {
            "metrics": _metrics(model["metrics"]),
            "diagnostics": None if model.get("diagnostics") is None else _project(model["diagnostics"], {
                key: _integer for key in ("paths", "rows", "nonphysical_ohlc_rows", "negative_volume_rows",
                                          "negative_amount_rows")
            }),
            "elapsed_seconds": _optional(_number)(model.get("elapsed_seconds")),
            "metadata": _metadata(model["metadata"]),
        }
        if "initialization" in model:
            models[name]["initialization"] = _project(model["initialization"], {
                "status": _enum("ok", "error"), "phase": _enum("initialization"), "error_type": _optional(_error_type),
            }, ("status", "phase", "error_type"))
        if "timing" in model:
            models[name]["timing"] = _project(model["timing"], {
                key: _number for key in ("initialization_seconds", "historical_seconds", "prospective_seconds")
            })
    pending = [row for row in report["cases"] if row["group"] == "prospective"]
    counts = _project(report["catalog_counts"], {key: _integer for key in ("planned", "eligible", "excluded")},
                      ("planned", "eligible", "excluded"))
    if counts["planned"] != counts["eligible"] + counts["excluded"]:
        raise ValueError("invalid public catalog counts")
    return {
        "schema_version": "kronos-paper-public-summary-v1",
        "report_id": _HASH(report["report_id"]),
        "published_at": _timestamp(report["published_at"]),
        "evidence_mode": "retrospective_development_pilot",
        "feed": _enum("iex", "sip")(report["connection"]["feed"]),
        "source_data_id": _HASH(report["source_data_id"]),
        "protocol": _protocol(report["protocol"]),
        "catalog_counts": counts,
        "models": models,
        "prospective": {
            "records": len(pending),
            "unavailable": _list(report.get("prospective_unavailable", []), lambda item: {
                "symbol": _SYMBOL(item["symbol"]),
                "reason": _label(item.get("reason", "unavailable_details_private"), _UNAVAILABLE_REASONS,
                                 "unavailable_details_private"),
            }),
            "observed": sum(r.get("outcome_status") == "observed" for r in pending),
            "target_times": sorted({_timestamp(r["future_times"][-1]) for r in pending}),
        },
        "broker_orders": _integer(report["broker_orders"]),
        "jev_calls": _integer(report["jev_calls"]),
        "limitations": _list(report["limitations"], lambda item: _label(
            item, _LIMITATIONS, "Additional private limitation recorded."
        )),
    }


def render(path: Path, output: Path) -> Path:
    summary = public_summary(path)
    output.mkdir(parents=True, exist_ok=False)
    if "MPLCONFIGDIR" not in os.environ:
        os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="kronos-plots-")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(summary["models"])
    values = [summary["models"][name]["metrics"] for name in names]
    colors = ["#187f75", "#375f6e", "#bb7726", "#77796a"][: len(names)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, key, title, percent in (
        (axes[0], "terminal_mae_bps", "Terminal price error (lower is better)", False),
        (axes[1], "direction_accuracy_errors_as_incorrect", "Direction accuracy (all eligible)", True),
    ):
        heights = [None if m[key] is None else m[key] * (100 if percent else 1) for m in values]
        bars = ax.bar(names, [h or 0 for h in heights], color=colors)
        for bar, value, m in zip(bars, heights, values, strict=True):
            label = "N/A" if value is None else f"{value:.1f}" + ("%" if percent else " bps")
            ax.annotate(
                label + f"\n{m['scored']}/{m['eligible']} scored",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, 100 if percent else max((h or 0 for h in heights), default=1) * 1.3 + 1)
        ax.set_ylabel("Percent; fixed +/-10 bps FLAT" if percent else "MAE / anchor price x 10,000")
        ax.spines[["top", "right"]].set_visible(False)
    counts = summary["catalog_counts"]
    stocks = len(summary["protocol"]["symbols"])
    fig.suptitle(
        "Kronos + Alpaca paper API: fixed development pilot\n"
        f"{counts['planned']} planned / {counts['eligible']} eligible | {stocks} stock" + ("s" if stocks != 1 else ""),
        fontsize=15,
    )
    fig.supxlabel(
        ", ".join(summary["protocol"]["dates"]) + " | 60 input candles -> 15 future candles\n"
        "Descriptive pilot; no tuning, confidence interval, profitability or promotion claim.",
        fontsize=10,
    )
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)
    result = output / "summary.json"
    result.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(render(args.report, args.output_dir))


if __name__ == "__main__":
    main()
