"""Render integrity-checked retrospective aggregates without publishing private market-data rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from render_historical_results import configure

from tradecopilot.forecast.experiment import load_report

MODEL_NAMES = {
    "prior": "Class prior",
    "price-only-control": "Price-only control",
    "selected-raw": "Selected winner\n(raw probabilities)",
    "selected-calibrated": "Same winner\n(calibrated probabilities)",
}
METRIC_KEYS = (
    "eligible",
    "scored",
    "attempted",
    "errors",
    "abstained",
    "missing_predictions",
    "coverage",
    "accuracy",
    "selective_accuracy",
    "macro_f1",
    "log_loss",
    "brier_score",
    "confusion_matrix",
    "labels",
    "class_counts",
)
SOURCE_KEYS = (
    "schema_version",
    "provenance",
    "provider",
    "provider_url",
    "documentation_url",
    "endpoint",
    "feed",
    "adjustment_policy",
    "selected_symbols",
    "start_date",
    "end_date_exclusive",
    "downloaded_bar_count",
    "bar_count",
    "calendar",
    "calendar_version",
    "timezone",
    "interval_seconds",
    "observed_sessions",
    "first_bar_start_utc",
    "last_bar_end_utc",
    "timestamp_semantics",
    "availability_assumption",
    "retrieved_at",
    "raw_data_policy",
)


def pick(value: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: value[key] for key in keys if key in value}


def interval(value: dict[str, Any] | None) -> dict[str, Any] | None:
    return pick(value, ("method", "low", "high", "sessions")) if value else None


def metrics(value: dict[str, Any]) -> dict[str, Any]:
    return {**pick(value, METRIC_KEYS), "accuracy_interval": interval(value.get("accuracy_interval"))}


def selective_metrics(value: dict[str, Any], *, sessions: bool = False) -> dict[str, Any]:
    result = pick(value, ("eligible", "accuracy", "selected", "coverage", "selective_accuracy", "selected_sessions"))
    result["by_predicted_label"] = {
        label: pick(value["by_predicted_label"][label], ("count", "precision")) for label in ("DOWN", "FLAT", "UP")
    }
    if sessions:
        result["selective_accuracy_interval"] = interval(value.get("selective_accuracy_interval"))
        result["by_session"] = {
            day: pick(row, ("eligible", "selected", "accuracy", "selective_accuracy"))
            for day, row in value["by_session"].items()
        }
    return result


def public_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Explicitly select aggregates; never copy cases, protocol block IDs or API request parameters."""
    protocol = report["protocol"]
    source = report["data_source"]
    selection = report["selection"]
    gate = selection["gate"]
    requirement_keys = (
        "target_accuracy",
        "minimum_coverage",
        "minimum_selected",
        "minimum_direction_count",
        "minimum_selected_sessions",
    )
    settings_keys = (
        "C",
        "l2_regularization",
        "max_iter",
        "max_leaf_nodes",
        "max_depth",
        "categorical_features",
        "learning_rate",
        "random_state",
        "class_weight",
        "min_samples_leaf",
        "early_stopping",
    )

    def candidate(value: dict[str, Any]) -> dict[str, Any]:
        return {**pick(value, ("key", "kind")), "settings": pick(value["settings"], settings_keys)}

    return {
        "schema_version": "derived-selective-retrospective-v1",
        **pick(
            report,
            (
                "report_id",
                "created_at",
                "evaluation_kind",
                "bar_dataset_id",
                "experiment_id",
                "target_status",
                "production_promoted",
            ),
        ),
        "source": {
            **pick(report["source"], ("source_sha256", "package_version", "python")),
            "dependencies": pick(
                report["source"]["dependencies"], ("numpy", "scikit-learn", "exchange-calendars", "pydantic")
            ),
        },
        "data_source": {
            **pick(source, SOURCE_KEYS),
            "downloads": [pick(row, ("sha256", "bytes", "retrieved_at")) for row in source.get("downloads", [])],
        },
        "dataset": pick(report["dataset"], ("dataset_id", "provenance", "example_count", "labeled_count", "sessions")),
        "protocol": {
            **pick(
                protocol,
                (
                    "version",
                    "selection_objective",
                    "temperature_grid",
                    "gate_thresholds",
                    "paid_api_calls",
                    "test_policy",
                ),
            ),
            "config": pick(
                protocol["config"],
                (
                    "schema_version",
                    "feature_version",
                    "horizon_minutes",
                    "flat_threshold_bps",
                    "lookback_minutes",
                    "anchor_seconds",
                    "max_source_age_seconds",
                    "max_outcome_delay_seconds",
                    "min_history_points",
                    "max_history_gap_seconds",
                    "seed",
                    "abstention_threshold",
                    "symbols",
                    "config_id",
                ),
            ),
            "candidates": [candidate(row) for row in protocol["candidates"]],
            "hgb_defaults": pick(protocol["hgb_defaults"], settings_keys),
            "gate_requirements": pick(protocol["gate_requirements"], requirement_keys),
        },
        "split": {
            key: pick(report["split"][key], ("examples", "sessions"))
            for key in ("development", "calibration", "gate", "test")
        },
        "tuning": [
            {
                "candidate": candidate(row["candidate"]),
                "pooled_log_loss": row["pooled_log_loss"],
                "fold_metrics": [metrics(fold) for fold in row["fold_metrics"]],
            }
            for row in report["tuning"]
        ],
        "selection": {
            **pick(selection, ("model_id", "temperature", "calibration_ids_hash", "gate_ids_hash")),
            "candidate": candidate(selection["candidate"]),
            "gate": {
                **pick(gate, ("enabled", "threshold", "reason")),
                "requirements": pick(gate["requirements"], requirement_keys),
                "curve_scope": "pretest_gate_block_only",
                "curve": [
                    {**pick(point, ("threshold", "qualifies")), **selective_metrics(point)} for point in gate["curve"]
                ],
            },
        },
        "selection_metrics": selective_metrics(report["selection_metrics"], sessions=True),
        "models": [
            {
                **pick(model, ("key", "label", "model_id", "execution")),
                "metrics": metrics(model["metrics"]),
                "by_symbol": {symbol: metrics(row) for symbol, row in model["by_symbol"].items()},
            }
            for model in report["models"]
        ],
        "limitations": report["limitations"],
    }


def percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def date_range(block: dict[str, Any]) -> str:
    days = block["sessions"]
    return f"{days[0]} to {days[-1]} ({len(days)} sessions)" if days else "no sessions"


def final_text(report: dict[str, Any]) -> str:
    value = report["selection_metrics"]
    gate = report["selection"]["gate"]
    directions = value["by_predicted_label"]
    return (
        f"Status: {report['target_status']}  |  Gate: {'enabled' if gate['enabled'] else 'disabled / abstain'}\n"
        f"Final test: {value['eligible']:,} eligible; {value['selected']:,} selected; "
        f"coverage {percent(value['coverage'])}; selected accuracy {percent(value['selective_accuracy'])}\n"
        f"UP precision {percent(directions['UP']['precision'])} (n={directions['UP']['count']:,}); "
        f"DOWN precision {percent(directions['DOWN']['precision'])} (n={directions['DOWN']['count']:,}); "
        f"selected sessions {value['selected_sessions']}/{len(report['split']['test']['sessions'])}"
    )


def comparison(report: dict[str, Any], path: Path) -> None:
    models = {model["key"]: model for model in report["models"]}
    fig, axes = plt.subplots(1, 3, figsize=(16, 8.2))
    names = list(MODEL_NAMES.values())
    for ax, key, title in zip(
        axes,
        ("accuracy", "log_loss", "brier_score"),
        ("All-case accuracy · higher is better", "Log loss · lower is better", "Brier score · lower is better"),
        strict=True,
    ):
        values = [models[name]["metrics"][key] for name in MODEL_NAMES]
        bars = ax.bar(
            range(4),
            [value if value is not None else 0 for value in values],
            color=["#94a3a0", "#6e9386", "#34745f", "#b16d2b"],
            width=0.62,
        )
        for bar, value in zip(bars, values, strict=True):
            label = percent(value) if key == "accuracy" else "N/A" if value is None else f"{value:.3f}"
            ax.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontweight="bold",
            )
        ax.set_xticks(range(4), names, rotation=15, ha="right", fontsize=9)
        ax.set_title(title, loc="left", fontsize=11, pad=15)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 1.08 if key == "accuracy" else max([v for v in values if v is not None] + [0.1]) * 1.25)
        if key == "accuracy":
            ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle("Selective forecasting · retrospective final test", x=0.04, ha="left", fontsize=20, fontweight="bold")
    fig.text(
        0.04,
        0.91,
        f"Final test: {date_range(report['split']['test'])}\n"
        f"Development winner: {report['selection']['candidate']['key']} · "
        f"temperature {report['selection']['temperature']:.3f}; calibration preserves argmax accuracy",
        fontsize=10,
    )
    fig.text(0.04, 0.24, final_text(report), fontsize=10, va="top", linespacing=1.6)
    fig.text(
        0.04,
        0.07,
        "Source: Alpaca SIP / raw adjustment · historical replay; all-case probability scores include abstentions.\n"
        "Required: accuracy ≥80%, coverage ≥10%, selected ≥100; UP and DOWN each n≥30 and precision ≥80%; "
        "selected sessions ≥5.\nRetrospective results do not establish future accuracy or profitability. "
        "No Jev calls or trading; no test retuning.\n"
        f"Development: {date_range(report['split']['development'])}; "
        f"calibration: {date_range(report['split']['calibration'])}.",
        fontsize=8.5,
        linespacing=1.4,
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.79, bottom=0.38, wspace=0.3)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def gate_plot(report: dict[str, Any], path: Path) -> None:
    gate = report["selection"]["gate"]
    points = gate["curve"]
    requirements = gate["requirements"]
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.plot(
        [p["threshold"] for p in points],
        [p["coverage"] for p in points],
        "o-",
        label="Gate-block coverage",
        color="#34745f",
    )
    valid = [p for p in points if p["selective_accuracy"] is not None]
    ax.plot(
        [p["threshold"] for p in valid],
        [p["selective_accuracy"] for p in valid],
        "s-",
        label="Gate-block selected accuracy",
        color="#b16d2b",
    )
    ax.axhline(requirements["target_accuracy"], linestyle="--", color="#b85151", label="80% accuracy criterion")
    ax.axhline(requirements["minimum_coverage"], linestyle=":", color="#64748b", label="10% coverage minimum")
    if gate["enabled"]:
        ax.axvline(gate["threshold"], linestyle="--", color="#263b35", label="Frozen gate threshold")
    for p in valid:
        ax.annotate(
            f"n={p['selected']:,}",
            (p["threshold"], p["selective_accuracy"]),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8,
        )
    ax.set(xlim=(0.3, 1), ylim=(0, 1.12), xlabel="Fixed candidate confidence threshold", ylabel="Fraction")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=0.15)
    ax.legend(loc="lower left", fontsize=9)
    fig.suptitle("Gate fitting · pretest gate block only", x=0.06, ha="left", fontsize=19, fontweight="bold")
    fig.text(
        0.06,
        0.91,
        f"Gate block: {date_range(report['split']['gate'])} · "
        f"{report['split']['gate']['examples']:,} eligible\n"
        "These curves fitted the frozen gate before final test; they are not final-test evidence.",
        fontsize=10,
    )
    fig.text(0.06, 0.16, final_text(report), fontsize=10, va="top", linespacing=1.5)
    fig.text(
        0.06,
        0.035,
        "Alpaca SIP / raw · fixed threshold grid; qualification also requires selected ≥100, "
        "UP/DOWN each n≥30 and precision ≥80%, ≥5 selected sessions.\n"
        "No qualifying gate means abstention: 0% coverage and N/A selected accuracy. "
        "Final test is scored once without retuning.",
        fontsize=9,
        linespacing=1.4,
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.79, bottom=0.3)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Absolute path to integrity-checked retrospective report.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.report.is_absolute():
        parser.error("report must be an absolute input path")
    report = load_report(args.report)
    if report.get("evaluation_kind") != "selective_accuracy_retrospective":
        raise ValueError(
            "only real selective_accuracy_retrospective reports may be rendered; synthetic fixtures rejected"
        )
    source = report["data_source"]
    if (
        source.get("fixture")
        or source.get("provenance") != "historical"
        or source.get("provider") != "Alpaca"
        or source.get("feed") != "sip"
        or source.get("adjustment_policy") != "raw"
    ):
        raise ValueError("renderer requires historical Alpaca SIP / raw provenance")
    if {model["key"] for model in report["models"]} != set(MODEL_NAMES) or len(report["models"]) != 4:
        raise ValueError("exactly the four recorded study models are required")
    selected = report["selection_metrics"]
    if not report["selection"]["gate"]["enabled"] and (
        selected["selected"] != 0 or selected["coverage"] != 0 or selected["selective_accuracy"] is not None
    ):
        raise ValueError("disabled gate must have zero selection and N/A selective accuracy")
    destination = args.output_dir.resolve()
    if destination == args.report.parent.resolve():
        raise ValueError("derived output must be separate from the immutable report bundle")
    summary = public_summary(report)
    destination.mkdir(parents=True, exist_ok=True)
    configure()
    comparison(summary, destination / "comparison.png")
    gate_plot(summary, destination / "gate-accuracy-coverage.png")
    (destination / "derived-results.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(destination)


if __name__ == "__main__":
    main()
