"""Render derived historical evaluation charts; export no raw prices or source rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tradecopilot.forecast.experiment import load_report

NAMES = {
    "prior": "Class prior",
    "momentum": "Momentum",
    "logistic": "Logistic",
    "logistic-calibrated": "Logistic + calibration",
    "jev-retrospective": "Jev (retrospective)",
    "v1-balanced-logistic": "Price LR / balanced",
    "v1-unweighted-logistic": "Price LR / unweighted",
    "ohlcv-logistic": "OHLCV LR",
    "ohlcv-hgb": "OHLCV boosting",
}
COLORS = ["#94a3a0", "#6e9386", "#367363", "#184f42", "#b16d2b"]
LABEL_COLORS = {"DOWN": "#b85151", "FLAT": "#a68b50", "UP": "#34745f"}


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "figure.facecolor": "#f5f7f4",
            "axes.facecolor": "#ffffff",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#263b35",
            "text.color": "#263b35",
            "xtick.color": "#475b54",
            "ytick.color": "#475b54",
        }
    )


def comparison(report: dict, path: Path, title: str, subtitle: str) -> None:
    models = report["models"]
    names = [NAMES[m["key"]] for m in models]
    figure, axes = plt.subplots(1, 3, figsize=(15, 6.5))
    for ax, metric, heading in zip(
        axes,
        ("accuracy", "log_loss", "brier_score"),
        ("Accuracy · higher is better", "Log loss · lower is better", "Brier score · lower is better"),
        strict=True,
    ):
        values = [m["metrics"][metric] for m in models]
        x = np.arange(len(models))
        bars = ax.bar(x, [v if v is not None else 0 for v in values], color=COLORS[: len(models)], width=0.62)
        for bar, value in zip(bars, values, strict=True):
            label = "unscored" if value is None else f"{100 * value:.1f}%" if metric == "accuracy" else f"{value:.3f}"
            ax.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )
        ax.set_xticks(x, names, rotation=25, ha="right")
        ax.set_title(heading, fontsize=11, loc="left", pad=14)
        ax.grid(axis="y", alpha=0.16)
        ax.set_axisbelow(True)
        if metric == "accuracy":
            ax.set_ylim(0, 1.06)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
        else:
            ax.set_ylim(0, max([v for v in values if v is not None] + [0.1]) * 1.28)
    figure.suptitle(title, fontsize=19, fontweight="bold", x=0.045, ha="left", y=0.99)
    figure.text(0.045, 0.91, subtitle, fontsize=10)
    summaries = [
        f"{NAMES[m['key']]}: {m['metrics']['scored']}/{m['metrics']['eligible']} scored, "
        f"{m['metrics']['coverage']:.0%} coverage, {m['metrics']['errors']} errors"
        for m in models
    ]
    availability = "\n".join("   |   ".join(summaries[index : index + 2]) for index in range(0, len(summaries), 2))
    figure.text(0.045, 0.17, availability, fontsize=8.2, va="top")
    figure.text(
        0.045,
        0.035,
        "Source: FirstRate Data free minute samples. Bar-end replay availability; "
        "fixed 15-minute target / ±10 bps neutral band.\n"
        "Probability scores include valid abstentions. "
        "Historical results do not establish future accuracy or profitability.",
        fontsize=8.5,
    )
    figure.subplots_adjust(left=0.06, right=0.98, top=0.83, bottom=0.32, wspace=0.28)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def diagnostics(report: dict, path: Path) -> None:
    model = next(m for m in report["models"] if m["key"] == "logistic-calibrated")
    metrics = model["metrics"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 6.3))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "--", color="#87928d", label="Reference")
    for label, points in metrics["reliability"].items():
        ax.plot(
            [p["mean_probability"] for p in points],
            [p["observed_frequency"] for p in points],
            "o-",
            color=LABEL_COLORS[label],
            label=label,
            markersize=4,
        )
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted class probability", ylabel="Observed class frequency")
    ax.set_title("Class reliability", loc="left", pad=12)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
    ax = axes[1]
    matrix = np.asarray(metrics["confusion_matrix"])
    ax.imshow(matrix, cmap="Greens")
    for row in range(3):
        for col in range(3):
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > matrix.max() * 0.55 else "#1c382e",
                fontweight="bold",
            )
    ax.set_xticks(range(3), metrics["labels"])
    ax.set_yticks(range(3), metrics["labels"])
    ax.set(xlabel="Predicted argmax", ylabel="Actual outcome")
    ax.set_title("Confusion matrix · all valid forecasts", loc="left", pad=12)
    ax = axes[2]
    curve = metrics["coverage_curve"]
    ax.plot([p["threshold"] for p in curve], [p["coverage"] for p in curve], "o-", color="#34745f", label="Coverage")
    valid = [p for p in curve if p["accuracy"] is not None]
    ax.plot(
        [p["threshold"] for p in valid],
        [p["accuracy"] for p in valid],
        "s-",
        color="#b16d2b",
        label="Selective accuracy",
    )
    for point in valid:
        ax.annotate(
            f"n={point['count']}",
            (point["threshold"], point["accuracy"]),
            xytext=(4, -14),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set(xlim=(0, 1), ylim=(0, 1.05), xlabel="Confidence threshold", ylabel="Fraction")
    ax.set_title("Coverage and accuracy", loc="left", pad=12)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
    figure.suptitle(
        "Historical holdout diagnostics · calibrated logistic baseline",
        fontsize=18,
        fontweight="bold",
        x=0.045,
        ha="left",
        y=0.99,
    )
    figure.text(
        0.045,
        0.91,
        f"{metrics['eligible']:,} labeled test cases · thresholds fixed before evaluation · test used only for scoring",
        fontsize=10,
    )
    figure.text(
        0.045,
        0.04,
        "Source: FirstRate Data. Calibration fitted on validation sessions only. "
        "Curves describe this holdout; do not tune thresholds from it.\n"
        f"Overlapping targets are correlated. Test sessions: {len(report['split']['test']['sessions'])}. "
        "Session-bootstrap intervals require at least five test sessions.",
        fontsize=8.5,
    )
    figure.subplots_adjust(left=0.06, right=0.98, top=0.81, bottom=0.19, wspace=0.34)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def jev_cases(report: dict, path: Path) -> None:
    cases = report["cases"]
    figure, ax = plt.subplots(figsize=(13.5, 7))
    bottoms = np.zeros(len(cases))
    x = np.arange(len(cases))
    for label in ("DOWN", "FLAT", "UP"):
        values = []
        for case in cases:
            prediction = case["predictions"]["jev-retrospective"]
            values.append((prediction["probabilities"] or {}).get(label, 0))
        ax.bar(x, values, bottom=bottoms, color=LABEL_COLORS[label], label=label, width=0.65)
        bottoms += np.asarray(values)
    labels = []
    for index, case in enumerate(cases):
        prediction = case["predictions"]["jev-retrospective"]
        labels.append(
            f"{index + 1}. {case['symbol']}\n{case['as_of'][5:16].replace('T', ' ')} UTC\nActual: {case['actual']}"
        )
        ax.text(index, 1.03, prediction["status"], ha="center", fontsize=8)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set(ylim=(0, 1.14), ylabel="Recorded Jev probability")
    ax.legend(loc="upper right", ncol=3)
    ax.grid(axis="y", alpha=0.16)
    ax.set_axisbelow(True)
    figure.suptitle(
        "Jev retrospective calls · every preselected case", x=0.045, ha="left", fontsize=18, fontweight="bold", y=0.98
    )
    figure.text(
        0.045,
        0.9,
        "Selections were frozen using chronological quantile midpoints; outcome labels were not sent to the model.",
        fontsize=10,
    )
    figure.text(
        0.045,
        0.035,
        "Retrospective inference: requests occurred after these outcomes. "
        "Model training contamination cannot be ruled out.\n"
        f"{len(cases)} cases are a small integration sample. "
        "Failed calls remain in the cohort. Source: FirstRate Data.",
        fontsize=8.5,
    )
    figure.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.21)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def public_summary(base: dict, retrospective: dict | None) -> dict:
    metrics = (
        "eligible",
        "scored",
        "errors",
        "abstained",
        "coverage",
        "accuracy",
        "macro_f1",
        "log_loss",
        "brier_score",
        "selective_accuracy",
        "class_counts",
        "api_cost_estimate_usd",
        "input_tokens",
    )

    def summarize(report: dict) -> dict:
        return {
            "report_id": report["report_id"],
            "evaluation_kind": report["evaluation_kind"],
            "split": report["split"],
            "limitations": report["limitations"],
            "models": [
                {
                    "key": m["key"],
                    "model_id": m["model_id"],
                    "execution": m["execution"],
                    "metrics": {k: m["metrics"][k] for k in metrics},
                    "by_symbol": {s: {k: values[k] for k in metrics} for s, values in m["by_symbol"].items()},
                }
                for m in report["models"]
            ],
        }

    return {
        "schema_version": "derived-historical-results-v1",
        "dataset_id": base["dataset"]["dataset_id"],
        "config": base["dataset"]["config"],
        "sessions": base["dataset"]["sessions"],
        "observation_count": base["dataset"]["observation_count"],
        "example_count": base["dataset"]["example_count"],
        "labeled_count": base["dataset"]["labeled_count"],
        "data_source": base["data_source"],
        "source": base["source"],
        "baseline": summarize(base),
        "retrospective_sample": summarize(retrospective) if retrospective else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--retrospective", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    base = load_report(args.baseline)
    if base["evaluation_kind"] != "historical_benchmark" or base["dataset"]["provenance"] != "historical":
        raise ValueError("publication requires a verified real historical report")
    retrospective = load_report(args.retrospective) if args.retrospective else None
    if retrospective and retrospective["dataset"]["dataset_id"] != base["dataset"]["dataset_id"]:
        raise ValueError("historical comparison reports must use the same dataset")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure()
    test = base["split"]["test"]
    comparison(
        base,
        args.output_dir / "historical-baselines.png",
        "Real historical stock data · full chronological holdout",
        f"AAPL / MSFT / AMZN / NFLX / TSLA · {test['examples']:,} test cases · "
        f"{test['sessions'][0]} to {test['sessions'][-1]} · all CPU models use the same holdout",
    )
    diagnostics(base, args.output_dir / "historical-diagnostics.png")
    if retrospective:
        comparison(
            retrospective,
            args.output_dir / "jev-retrospective-comparison.png",
            "Jev versus CPU baselines · retrospective sample",
            f"Same {len(retrospective['cases'])} preselected "
            f"{' / '.join(sorted({c['symbol'] for c in retrospective['cases']}))} cases; "
            "calls occurred after outcomes. Small-N results are descriptive.",
        )
        jev_cases(retrospective, args.output_dir / "jev-retrospective-cases.png")
    (args.output_dir / "derived-results.json").write_text(
        json.dumps(public_summary(base, retrospective), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"published_derived_artifacts": str(args.output_dir.resolve())}))


if __name__ == "__main__":
    main()
