"""Render validated private RL study outcomes as aggregate-only public artifacts."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from statistics import mean
from typing import Any

from tradecopilot.forecast.contracts import content_hash
from tradecopilot.rl.evaluation import paired_interval
from tradecopilot.rl.study import load_report

_METRICS = ("valid", "episodes", "dates", "invalid_episodes", "trade_count", "turnover", "fees", "execution_drag")


def _metrics(value: dict[str, Any]) -> dict[str, Any]:
    result = {key: value[key] for key in _METRICS if key in value}
    net = value.get("mean_daily_return")
    drawdown = value.get("max_episode_drawdown")
    result["mean_daily_return_bps"] = net * 10000 if net is not None else None
    result["max_episode_drawdown_bps"] = drawdown * 10000 if drawdown is not None else None
    return result


def _interval(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": value["method"],
        "dates": value["dates"],
        **{name: value[name] * 10000 for name in ("mean", "low", "high")},
    }


def public_summary(report: dict[str, Any], registration: dict[str, Any]) -> dict[str, Any]:
    """Explicit allowlist: never copy private vectors, ledgers, checkpoint paths or source metadata."""
    training = {(r["candidate_key"], r["seed"]): r for r in report["training"]}
    architectures = []
    keys = list(dict.fromkeys(row["candidate"] for row in report["tune"]))
    for key in keys:
        seeds = []
        for row in report["tune"]:
            if row["candidate"] != key:
                continue
            run = training[(key, row["seed"])]
            seeds.append(
                {
                    "seed": row["seed"],
                    "metrics": _metrics(row["metrics"]),
                    **{
                        name: run[name]
                        for name in ("n_parameters", "actual_timesteps", "elapsed_seconds", "status", "device")
                    },
                }
            )
        returns = [row["metrics"]["mean_daily_return_bps"] for row in seeds]
        valid = all(row["metrics"]["valid"] and value is not None for row, value in zip(seeds, returns, strict=True))
        architectures.append(
            {
                "candidate": key,
                "seeds": seeds,
                "all_seeds_valid": valid,
                "mean_seed_daily_return_bps": mean(returns) if valid else None,
                "min_seed_daily_return_bps": min(returns) if valid else None,
                "max_seed_daily_return_bps": max(returns) if valid else None,
            }
        )
    selection = report["selection"]
    primary = _metrics(report["test_selected_architecture_seeds"][str(selection["seed"])])
    stress = {}
    for cost, outcome in report["cost_stress"].items():
        value = {"neural": _metrics(outcome["neural"]), "control": _metrics(outcome["control"])}
        if outcome["neural"]["valid"] and outcome["control"]["valid"]:
            value["incremental_interval_bps"] = _interval(
                paired_interval(outcome["neural"]["daily_returns"], outcome["control"]["daily_returns"])
            )
        else:
            value["incremental_interval_bps"] = None
        stress[cost] = value
    gate = report["improvement_gate"]
    result = {
        "schema_version": "rl-capacity-public-v1",
        "scope": report["scope"],
        "status": report["status"],
        "metric": "mean daily net policy return in basis points",
        "interpretation": (
            "Retrospective independent symbol-day books with daily capital resets; "
            "no annualized operational profit claim."
        ),
        "identities": {key: report[key] for key in ("report_id", "registration_id", "prepared_data_id", "budget_id")},
        "splits": {key: list(registration["splits"][key]) for key in ("train", "tune", "test")},
        "selected_candidate": selection["candidate"],
        "primary_seed": selection["seed"],
        "selection_id": selection["selection_id"],
        "strongest_tune_control": selection["strongest_control"],
        "tune_architectures": architectures,
        "primary_test": primary,
        "test_seed_metrics": {
            key: _metrics(value) for key, value in report["test_selected_architecture_seeds"].items()
        },
        "test_controls": {key: _metrics(value) for key, value in report["test_controls"].items()},
        "cost_stress": stress,
        "improvement_gate": {
            "passed": gate["passed"],
            "checks": {key: bool(value) for key, value in gate.get("checks", {}).items()},
            "paired_intervals_bps": {key: _interval(value) for key, value in gate.get("intervals", {}).items()},
        },
        "constraints": {"new_jev_calls": 0, "broker_orders": 0},
    }
    result["public_id"] = content_hash(result)
    return result


def render(report_path: Path, output: Path) -> Path:
    if output.exists():
        raise ValueError("render output requires a fresh directory")
    report = load_report(report_path)
    seal = json.loads((report_path.parent / "study-seal.json").read_text())
    public = public_summary(report, seal["registration"])
    # Matplotlib remains optional until rendering; its cache lives in a temporary directory.
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="rl-render-mpl-"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(public, indent=2, sort_keys=True, allow_nan=False))
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#f8fafc",
            "axes.facecolor": "#f8fafc",
        }
    )
    blue, grey = "#2563eb", "#64748b"
    subtitle = "MARKET_ONLY · Retrospective policy returns · " + public["status"].replace("_", " ")

    def finish(fig: Any, filename: str) -> None:
        fig.text(0.5, 0.015, subtitle, ha="center", fontsize=9, color=grey)
        fig.tight_layout(rect=(0, 0.055, 1, 1))
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for index, row in enumerate(public["tune_architectures"]):
        values = [seed["metrics"]["mean_daily_return_bps"] for seed in row["seeds"]]
        for offset, seed, value in zip((-0.08, 0, 0.08), row["seeds"], values, strict=True):
            if value is not None:
                ax.scatter(
                    index + offset,
                    value,
                    label=f"Seed {seed['seed']}" if index == 0 else None,
                    color=("#2563eb", "#7c3aed", "#0891b2")[seed["seed"] - 42],
                    s=65,
                )
        if row["mean_seed_daily_return_bps"] is not None:
            ax.plot([index - 0.2, index + 0.2], [row["mean_seed_daily_return_bps"]] * 2, color="#111827", lw=2)
    ax.set_xticks(range(len(public["tune_architectures"])), [row["candidate"] for row in public["tune_architectures"]])
    ax.axhline(0, color=grey, linewidth=0.8)
    ax.set_ylabel("Mean daily net policy return (bps)")
    ax.set_title("February TUNE: all registered architectures and seeds\nBlack lines show the three-seed means")
    ax.legend()
    finish(fig, "tune-architectures.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [f"{public['selected_candidate']} / seed {seed}" for seed in public["test_seed_metrics"]]
    values = [row["mean_daily_return_bps"] for row in public["test_seed_metrics"].values()]
    labels += list(public["test_controls"])
    values += [row["mean_daily_return_bps"] for row in public["test_controls"].values()]
    ax.bar(
        range(len(labels)),
        [value if value is not None else 0 for value in values],
        color=[blue] * len(public["test_seed_metrics"]) + [grey] * len(public["test_controls"]),
    )
    for index, value in enumerate(values):
        if value is None:
            ax.text(index, 0, "INVALID", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    ax.axhline(0, color=grey, linewidth=0.8)
    ax.set_ylabel("Mean daily net policy return (bps)")
    ax.set_title(f"March cold TEST at 2 bps/side · TUNE-frozen primary seed {public['primary_seed']}")
    finish(fig, "cold-test-controls.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    costs = [2] + [int(cost) for cost in public["cost_stress"]]
    for kind, label, color in [
        ("neural", "TUNE-frozen neural checkpoint", blue),
        ("control", public["strongest_tune_control"], grey),
    ]:
        baseline = (
            public["primary_test"] if kind == "neural" else public["test_controls"][public["strongest_tune_control"]]
        )
        values = [baseline["mean_daily_return_bps"]] + [
            row[kind]["mean_daily_return_bps"] for row in public["cost_stress"].values()
        ]
        ax.plot(
            costs, [float("nan") if value is None else value for value in values], marker="o", color=color, label=label
        )
    ax.axhline(0, color=grey, linewidth=0.8)
    ax.set_xticks(costs)
    ax.set_xlabel("Execution cost per side (bps)")
    ax.set_ylabel("Mean daily net policy return (bps)")
    ax.set_title("Cold TEST cost stress: fixed checkpoint and strongest TUNE control")
    ax.legend()
    finish(fig, "cost-stress.png")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis("off")
    rows = []
    for row in public["tune_architectures"]:
        run = row["seeds"][0]
        net = row["mean_seed_daily_return_bps"]
        rows.append(
            [
                row["candidate"],
                f"{net:.3f}" if net is not None else "INVALID",
                f"{run['n_parameters']:,}",
                f"{run['actual_timesteps']:,}",
                f"{mean(seed['elapsed_seconds'] for seed in row['seeds']):.1f}",
            ]
        )
    table = ax.table(
        cellText=rows,
        colLabels=["Architecture", "TUNE mean (bps)", "Parameters", "Steps/seed", "Mean training sec"],
        loc="center",
        cellLoc="center",
        colWidths=[0.25, 0.22, 0.17, 0.17, 0.19],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    ax.set_title(
        f"Registered capacity comparison\nSelected: {public['selected_candidate']} "
        f"· primary seed {public['primary_seed']}",
        pad=15,
    )
    finish(fig, "capacity-table.png")
    return output / "summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(render(args.report, args.output_dir))


if __name__ == "__main__":
    main()
