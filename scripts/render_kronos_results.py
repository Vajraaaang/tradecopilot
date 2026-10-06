"""Publish only aggregate Kronos pilot metrics from a verified private bundle."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from tradecopilot.forecast.kronos_report import load_report


def public_summary(path: Path) -> dict[str, Any]:
    report = load_report(path)
    if report["evidence_mode"] != "retrospective_development_pilot":
        raise ValueError("real development evidence is required")
    models = {}
    for name, model in report["models"].items():
        metadata = model["metadata"]
        models[name] = {
            "metrics": model["metrics"],
            "diagnostics": model.get("diagnostics"),
            "elapsed_seconds": model.get("elapsed_seconds"),
            "metadata": {
                key: metadata[key]
                for key in (
                    "variant",
                    "model",
                    "tokenizer",
                    "source",
                    "device",
                    "dtype",
                    "threads",
                    "sampler",
                    "model_parameter_count",
                    "tokenizer_parameter_count",
                    "dependencies",
                    "amount_policy",
                    "fine_tuned",
                )
                if key in metadata
            },
        }
    pending = [row for row in report["cases"] if row["group"] == "prospective"]
    return {
        "schema_version": "kronos-paper-public-summary-v1",
        "report_id": report["report_id"],
        "published_at": report["published_at"],
        "evidence_mode": report["evidence_mode"],
        "feed": report["connection"]["feed"],
        "source_data_id": report["source_data_id"],
        "protocol": report["protocol"],
        "catalog_counts": report["catalog_counts"],
        "models": models,
        "prospective": {
            "records": len(pending),
            "unavailable": report.get("prospective_unavailable", []),
            "observed": sum(r.get("outcome_status") == "observed" for r in pending),
            "target_times": sorted({r["future_times"][-1] for r in pending}),
        },
        "broker_orders": report["broker_orders"],
        "jev_calls": report["jev_calls"],
        "limitations": report["limitations"],
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
