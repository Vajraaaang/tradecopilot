"""Render a sealed Jev ablation into aggregate-only JSON and scientific PNGs."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from datetime import date
from pathlib import Path
from statistics import mean
from typing import Any

from tradecopilot.forecast.contracts import content_hash
from tradecopilot.forecast.jev_rl import load_cache
from tradecopilot.rl import jev_ablation as ab
from tradecopilot.rl import training
from tradecopilot.rl.evaluation import paired_interval

PROFILES = ab.PROFILES
SEEDS = (42, 43, 44)
CONTROLS = ("cash", "hold", "rule")
STATUSES = ("improved_among_tested", "inconclusive_not_promoted")
CHECKS = (
    "positive_mean_seed_increment_lower_bound",
    "positive_jev_mean_seed_net_return",
    "positive_cash_lower_bound",
    "drawdown_limit",
    "positive_doubled_cost_mean_seed_increment",
)


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("finite numeric aggregate required")
    return float(value)


def _hash(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("SHA256 identity required")
    return value


def _metrics(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("valid") is not True or value.get("invalid_episodes") != 0:
        raise ValueError("invalid or incomplete policy evaluation cannot be plotted")
    daily = value["daily_returns"]
    if not daily or not math.isclose(
        mean(map(_number, daily.values())), _number(value["mean_daily_return"]), abs_tol=1e-12
    ):
        raise ValueError("policy daily mean mismatch")
    result = {
        "mean_daily_return_bps": _number(value["mean_daily_return"]) * 10000,
        "max_episode_drawdown_bps": _number(value["max_episode_drawdown"]) * 10000,
    }
    for key in ("episodes", "dates", "trade_count", "fees", "turnover", "execution_drag"):
        result[key] = _number(value[key])
        if result[key] < 0:
            raise ValueError("negative policy count or cost")
    if result["dates"] != len(daily):
        raise ValueError("policy date count mismatch")
    return result


def _profiles(value: dict[str, Any]) -> dict[str, Any]:
    if set(value) != set(PROFILES):
        raise ValueError("both registered profiles required")
    for group in value.values():
        if set(group) != {str(s) for s in SEEDS}:
            raise ValueError("all three registered seeds required")
        for metrics in group.values():
            _metrics(metrics)
    return {p: ab.mean_seed_metrics(value[p]) for p in PROFILES}


def _mean_metrics(group: dict[str, Any], combined: dict[str, Any]) -> dict[str, Any]:
    rows = [_metrics(group[str(s)]) for s in SEEDS]
    if any((r["episodes"], r["dates"]) != (rows[0]["episodes"], rows[0]["dates"]) for r in rows):
        raise ValueError("seed evaluation scope mismatch")
    return {
        "seeds": list(SEEDS),
        "mean_daily_return_bps": _number(combined["mean_daily_return"]) * 10000,
        "max_episode_drawdown_bps": _number(combined["max_episode_drawdown"]) * 10000,
        "episodes_per_seed": rows[0]["episodes"],
        "dates": rows[0]["dates"],
        **{
            f"mean_{k}_per_seed": mean(r[k] for r in rows)
            for k in ("trade_count", "fees", "turnover", "execution_drag")
        },
    }


def _interval(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result = paired_interval(left["daily_returns"], right["daily_returns"])
    return {
        "method": "paired_date_bootstrap_95pct",
        "dates": result["dates"],
        **{key: _number(result[key]) * 10000 for key in ("mean", "low", "high")},
    }


def _availability(value: dict[str, Any]) -> dict[str, Any]:
    result = {
        k: _number(value[k]) for k in ("cohort_records", "available_records", "unavailable_records", "error_rate")
    }
    cohort, available, unavailable = (result[k] for k in ("cohort_records", "available_records", "unavailable_records"))
    if (
        cohort <= 0
        or any(v < 0 or v % 1 for v in (cohort, available, unavailable))
        or available + unavailable != cohort
        or not math.isclose(result["error_rate"], unavailable / cohort, abs_tol=1e-12)
    ):
        raise ValueError("invalid forecast availability counts or error denominator")
    return result


def _forecast_metrics(row: dict[str, Any]) -> dict[str, Any]:
    out = _availability(row)
    if row["records"] != out["cohort_records"]:
        raise ValueError("forecast record cohort mismatch")
    out["records"] = out["cohort_records"]
    for key in (
        "accuracy",
        "accuracy_available",
        "accuracy_errors_as_incorrect",
        "coverage_0_6",
        "selected_accuracy_0_6",
        "log_loss",
        "Brier",
    ):
        value = row[key]
        out[key] = None if value is None else _number(value)
        if out[key] is not None and (out[key] < 0 or (key not in ("log_loss", "Brier") and out[key] > 1)):
            raise ValueError("forecast accuracy and coverage must be fractions; losses nonnegative")
    if row["loss_scope"] != "available_records_only" or row["coverage_scope"] != "full_cohort":
        raise ValueError("forecast loss/coverage denominator mismatch")
    out["loss_scope"] = "available_records_only"
    out["coverage_scope"] = "full_cohort"
    out["accuracy_scope"] = "conditional_on_available_records"
    available, cohort = out["available_records"], out["cohort_records"]
    if out["accuracy"] != out["accuracy_available"] or out["accuracy_errors_as_incorrect"] is None:
        raise ValueError("forecast conditional accuracy alias mismatch")
    if available:
        if any(out[k] is None for k in ("accuracy_available", "log_loss", "Brier")):
            raise ValueError("available forecast metrics missing")
        expected = out["accuracy_available"] * available / cohort
    else:
        if any(out[k] is not None for k in ("accuracy_available", "log_loss", "Brier")):
            raise ValueError("unavailable forecasts cannot have conditional quality")
        expected = 0
    if not math.isclose(out["accuracy_errors_as_incorrect"], expected, abs_tol=1e-12):
        raise ValueError("forecast full cohort accuracy denominator mismatch")
    if out["coverage_0_6"] is None:
        raise ValueError("forecast full cohort coverage required")
    selected_count = out["coverage_0_6"] * cohort
    out["selected_records_0_6"] = round(selected_count)
    if (
        not math.isclose(selected_count, out["selected_records_0_6"], abs_tol=1e-9)
        or out["selected_records_0_6"] > available
        or (out["selected_records_0_6"] == 0) != (out["selected_accuracy_0_6"] is None)
    ):
        raise ValueError("forecast selected coverage mismatch")
    return out


def _forecast(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("vendor_pretraining_cutoff") != "unknown" or set(value["horizons"]) != set(ab.HORIZONS):
        raise ValueError("forecast grading scope mismatch")
    result: dict[str, Any] = {"availability": _availability(value["availability"]), "horizons": {}}
    totals = dict.fromkeys(("cohort_records", "available_records", "unavailable_records"), 0)
    for horizon in ab.HORIZONS:
        source = value["horizons"][horizon]
        entry: dict[str, Any] = {
            "train_labels": _number(source["train_labels"]),
            "train_availability": _availability(source["train_availability"]),
        }
        if entry["train_labels"] != entry["train_availability"]["cohort_records"]:
            raise ValueError("TRAIN prior label cohort mismatch")
        for key in totals:
            totals[key] += entry["train_availability"][key]
        for role in ("tune", "test"):
            entry[role] = {
                model: _forecast_metrics(source[role][model])
                for model in ("jev", "prior_full_cohort", "prior_same_available")
            }
            jev, prior, matched = (entry[role][m] for m in ("jev", "prior_full_cohort", "prior_same_available"))
            if (
                prior["cohort_records"] != jev["cohort_records"]
                or prior["unavailable_records"] != 0
                or any(
                    matched[key] != jev[key]
                    for key in ("cohort_records", "available_records", "unavailable_records", "error_rate")
                )
            ):
                raise ValueError("forecast prior denominator/cohort mismatch")
            if source[role]["prior"] != source[role]["prior_full_cohort"]:
                raise ValueError("TRAIN prior full cohort alias mismatch")
            for key in totals:
                totals[key] += jev[key]
        result["horizons"][horizon] = entry
    if any(totals[k] != result["availability"][k] for k in totals):
        raise ValueError("forecast total availability scope mismatch")
    return result


def _acquisition(cache: dict[str, Any]) -> dict[str, Any]:
    amendment = cache.get("acquisition_amendment")
    if amendment is None:
        return {"amended": False, "original_abort_rule_overridden": False}
    amendment_id = _hash(cache["amendment_id"])
    if amendment["amendment_id"] != amendment_id:
        raise ValueError("acquisition amendment identity mismatch")
    if (
        amendment["retry_provider_failures"] is not False
        or amendment["failure_policy"] != "preserve_unavailable_and_continue_unattempted"
    ):
        raise ValueError("unsupported acquisition continuation policy")
    counts = {k: _number(cache["batch_counts"][k]) for k in ("valid", "unavailable")}
    if any(v < 0 or v % 1 for v in counts.values()) or sum(counts.values()) != cache["usage"]["attempts"]:
        raise ValueError("acquisition batch counts mismatch")
    caps = {
        k: _number(amendment["api_budget"][k])
        for k in ("max_attempts", "max_conservative_cost_usd", "cooldown_seconds")
    }
    if caps != {"max_attempts": 80, "max_conservative_cost_usd": 0.25, "cooldown_seconds": 30}:
        raise ValueError("acquisition budget changed")
    return {
        "amended": True,
        "amendment_id": amendment_id,
        "batch_counts": counts,
        "original_abort_rule_overridden": True,
        "resume_unattempted_only": True,
        "retry_provider_failures": False,
        "same_original_caps": True,
        "caps": caps,
        "payloads_and_prompts_unchanged": True,
        "initial_attempted_batches": len(amendment["existing_attempted_batch_indices"]),
        "continuation_planned_batches": len(amendment["remaining_unattempted_batch_indices"]),
    }


def public_summary(report: dict[str, Any], seal: dict[str, Any], cache: dict[str, Any]) -> dict[str, Any]:
    """Construct an explicit scalar allowlist, recomputing primary means over all seeds."""
    if report.get("status") not in STATUSES:
        raise ValueError("failed or incomplete report cannot be meaningfully plotted")
    reg = seal["registration"]
    if reg["profiles"] != list(PROFILES) or reg["seeds"] != list(SEEDS):
        raise ValueError("unregistered profiles or seeds")
    splits = {
        role: [date.fromisoformat(v).isoformat() for v in reg["splits"][role]] for role in ("train", "tune", "test")
    }
    symbols = reg["symbols"]
    if not symbols or any(
        not isinstance(v, str) or re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,14}", v) is None for v in symbols
    ):
        raise ValueError("registered symbol scope required")
    tune = _profiles(report["tune_seeds"])
    test = _profiles(report["test_seeds"])
    if any(test[p] != report["test_mean_seed"][p] for p in PROFILES):
        raise ValueError("primary TEST must average all three seeds")
    controls = report["test_controls"]
    if set(controls) != set(CONTROLS):
        raise ValueError("all registered TEST controls required")
    intervals = {
        "jev_minus_context": _interval(test["JEV_ASSISTED"], test["CONTEXT_ONLY"]),
        "jev_minus_cash": _interval(test["JEV_ASSISTED"], controls["cash"]),
    }
    stress = {"2": {p: _mean_metrics(report["test_seeds"][p], test[p]) for p in PROFILES}}
    if set(report["cost_stress"]) != {"4", "8"}:
        raise ValueError("2/4/8 bps cost stress required")
    for cost in ("4", "8"):
        source = report["cost_stress"][cost]
        combined = _profiles(source["profiles"])
        if combined != source["mean_seed"]:
            raise ValueError("cost stress must use same three seed means")
        stress[cost] = {p: _mean_metrics(source["profiles"][p], combined[p]) for p in PROFILES}
        for control in CONTROLS:
            _metrics(source["controls"][control])
    training_rows = report["training"]
    pairs = [(r["profile"], r["seed"]) for r in training_rows]
    if len(pairs) != 6 or set(pairs) != {(p, s) for p in PROFILES for s in SEEDS}:
        raise ValueError("six matched complete training runs required")
    capacity = training_rows[0]["n_parameters"]
    steps = training_rows[0]["actual_timesteps"]
    for row in training_rows:
        if (
            row["status"] != "complete"
            or row["observation_dim"] != 151
            or row["n_parameters"] != capacity
            or row["actual_timesteps"] != steps
            or row["candidate"] != training._CANDIDATES["recurrent-256"]
        ):
            raise ValueError("matched 151-dimensional RecurrentPPO-256 capacity required")
    diagnostics = {}
    for profile in PROFILES:
        selected = report["selection"]["profiles"][profile]
        expected = max(SEEDS, key=lambda s: (report["tune_seeds"][profile][str(s)]["mean_daily_return"], -s))
        if selected["seed"] != expected:
            raise ValueError("checkpoint selection must use TUNE only")
        selected_run = next(r for r in training_rows if (r["profile"], r["seed"]) == (profile, expected))
        if any(selected[k] != selected_run[k] for k in ("training_id", "checkpoint_sha256")):
            raise ValueError("selected diagnostic checkpoint/training identity mismatch")
        diagnostics[profile] = {
            "seed": expected,
            "training_id": _hash(selected["training_id"]),
            "checkpoint_sha256": _hash(selected["checkpoint_sha256"]),
            "role": "TUNE-selected checkpoint diagnostic; not primary TEST estimate",
        }
    gate = report["improvement_gate"]
    passed = gate["passed"]
    if not isinstance(passed, bool) or passed != (report["status"] == "improved_among_tested"):
        raise ValueError("promotion status and gate mismatch")
    if any(not isinstance(gate["checks"][key], bool) for key in CHECKS):
        raise ValueError("promotion checks must be boolean")
    if cache["cache_id"] != report["cache_id"]:
        raise ValueError("cache/report identity mismatch")
    usage = cache["usage"]
    usage = {key: _number(usage[key]) for key in ("attempts", "input_tokens", "conservative_estimated_cost_usd")}
    if any(v < 0 for v in usage.values()) or usage["attempts"] % 1 or usage["input_tokens"] % 1:
        raise ValueError("invalid cache API usage")
    result = {
        "schema_version": "jev-rl-public-v1",
        "status": report["status"],
        "scope": "hypothetical_retrospective_jev_cache",
        "vendor_pretraining_cutoff": "unknown",
        "cache_availability_mode": "hypothetical_as_of_plus_60_seconds",
        "metric": "mean daily net policy return in basis points",
        "interpretation": "Equal-weight independent symbol-day capital resets; primary TEST averages all three seeds.",
        "source_attribution": "Alpaca SIP / raw bars; Jev 1.13 retrospective forecasts",
        "identities": {
            key: _hash(report[key])
            for key in ("report_id", "registration_id", "base_prepared_data_id", "cache_id", "budget_id")
        },
        "selection_id": _hash(report["selection"]["selection_id"]),
        "splits": splits,
        "symbols": list(symbols),
        "profile_contract": {
            "algorithm": "RecurrentPPO",
            "lstm_hidden_size": 256,
            "observation_dim": 151,
            "matched_parameters": capacity,
            "unavailable_forecasts": "All eight forecast fields zero for both profiles; all market episodes retained",
            "CONTEXT_ONLY": "Uniform direction probabilities and zero confidence; same context metadata",
            "JEV_ASSISTED": "Raw Jev probabilities and confidence; same context metadata",
        },
        "tune": {
            p: {
                "mean_daily_return_bps": tune[p]["mean_daily_return"] * 10000,
                "seed_points": [
                    {"seed": s, "mean_daily_return_bps": report["tune_seeds"][p][str(s)]["mean_daily_return"] * 10000}
                    for s in SEEDS
                ],
            }
            for p in PROFILES
        },
        "primary_test": {p: _mean_metrics(report["test_seeds"][p], test[p]) for p in PROFILES},
        "test_controls": {c: _metrics(controls[c]) for c in CONTROLS},
        "paired_intervals_bps": intervals,
        "selected_tune_diagnostics": diagnostics,
        "cost_stress_bps_per_side": stress,
        "forecast_grading": _forecast(report["forecast_grading"]),
        "acquisition": _acquisition(cache),
        "forecast_metric_units": "accuracy/coverage fractions; log loss natural logarithm; multiclass Brier sum",
        "forecast_coverage_definition": (
            "probability and confidence both >= 0.6; denominator is full cohort including unavailable records"
        ),
        "promotion": {"passed": passed, "checks": {k: gate["checks"][k] for k in CHECKS}},
        "training": {
            p: {
                "seeds": list(SEEDS),
                "steps_per_seed": steps,
                "mean_elapsed_seconds": mean(_number(r["elapsed_seconds"]) for r in training_rows if r["profile"] == p),
                "parameters": capacity,
            }
            for p in PROFILES
        },
        "api_usage": {
            **usage,
            "cost_basis": "conservative estimate from actual input tokens; not billed cost",
            "new_jev_calls_during_ablation": 0,
        },
    }
    result["public_id"] = content_hash(result)
    return result


def _validated_report(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    report = ab.load_report(path)
    root = path.parent.resolve()
    inventory = report["inventory"]
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p.name != "report.json"}
    if set(inventory) != actual or "ablation-seal.json" not in inventory:
        raise ValueError("complete sealed artifact inventory required")
    # load_report verifies every inventory entry; additionally check the frozen experiment identities.
    seal = training._read_sealed(root / "ablation-seal.json", "ablation_id")
    ab._registration(seal["registration"])
    ab._verify_hash(seal["budget"], "budget_id")
    ab._verify_hash(seal["manifest"], "data_id")
    identities = {
        "registration_id": seal["registration"]["registration_id"],
        "base_prepared_data_id": seal["manifest"]["data_id"],
        "cache_id": seal["cache_id"],
        "budget_id": seal["budget"]["budget_id"],
    }
    if any(report.get(key) != value for key, value in identities.items()):
        raise ValueError("report/frozen experiment identity mismatch")
    if report.get("status") not in STATUSES:
        raise ValueError("failed or incomplete report cannot be meaningfully plotted")
    ab._verify_hash(report["selection"], "selection_id")
    ab._verify_training(
        report["training"],
        seal["registration"],
        seal["budget"],
        seal["source_provenance"],
        seal["lock_sha256"],
        seal["execution_provenance"],
    )
    for run in report["training"]:
        model = root / f"{run['profile']}-{run['seed']}" / "model.zip"
        if inventory.get(str(model.relative_to(root))) != run["checkpoint_sha256"]:
            raise ValueError("training checkpoint inventory mismatch")
    return report, seal


def render(report_path: Path, output: Path, cache_path: Path) -> Path:
    if not all(p.is_absolute() for p in (report_path, output, cache_path)):
        raise ValueError("absolute report, cache and output paths required")
    if output.exists():
        raise ValueError("render output requires a fresh directory")
    report, seal = _validated_report(report_path)
    cache = load_cache(cache_path)
    public = public_summary(report, seal, cache)
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="jev-render-mpl-"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(public, indent=2, sort_keys=True, allow_nan=False))
    colors = ("#526782", "#177e89")
    labels = ("Context only", "Jev assisted")
    status = "Promotion passed among tested" if public["promotion"]["passed"] else "FAILED PROMOTION / inconclusive"

    def finish(fig: Any, filename: str, kind: str) -> None:
        fig.text(0.5, 0.045, f"Hypothetical retrospective {kind} | {status}", ha="center", fontsize=9)
        fig.text(0.5, 0.012, public["source_attribution"], ha="center", fontsize=8, color="#536170")
        fig.tight_layout(rect=(0, 0.085, 1, 1))
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)

    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(9, 5.4))
        for seed, color in zip(SEEDS, ("#c26a36", "#4d65a6", "#177e89"), strict=True):
            values = [
                next(r["mean_daily_return_bps"] for r in public["tune"][p]["seed_points"] if r["seed"] == seed)
                for p in PROFILES
            ]
            ax.plot(range(2), values, marker="o", color=color, label=f"Paired seed {seed}")
        for i, p in enumerate(PROFILES):
            ax.plot([i - 0.15, i + 0.15], [public["tune"][p]["mean_daily_return_bps"]] * 2, color="black", lw=3)
        ax.set_xticks(range(2), labels)
        ax.set_ylabel("Mean daily net policy return (bps)")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title("TUNE policy returns: all three paired seeds\nBlack marks: three-seed means")
        ax.legend()
        finish(fig, "tune-paired-seeds.png", "policy returns")

        fig, (ax, ci) = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": [1.2, 1]})
        values = [public["primary_test"][p]["mean_daily_return_bps"] for p in PROFILES]
        values += [public["test_controls"][c]["mean_daily_return_bps"] for c in CONTROLS]
        ax.bar(range(5), values, color=[*colors, "#b1b8c2", "#b1b8c2", "#b1b8c2"])
        ax.set_xticks(range(5), ["Context\n3-seed mean", "Jev\n3-seed mean", "Cash", "Hold", "Rule"])
        ax.set_ylabel("Mean daily net policy return (bps)")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title("Primary cold TEST at 2 bps/side")
        for i, key in enumerate(("jev_minus_context", "jev_minus_cash")):
            interval = public["paired_intervals_bps"][key]
            ci.plot([interval["low"], interval["high"]], [i, i], color=colors[1], lw=3)
            ci.scatter(interval["mean"], i, color=colors[1], s=65)
        ci.axvline(0, color="grey", lw=0.8)
        ci.set_yticks([0, 1], ["Jev minus context", "Jev minus cash"])
        ci.set_ylim(-0.5, 1.5)
        ci.set_xlabel("Paired daily increment (bps), 95% CI")
        ci.set_title("Bootstrap over dates, after seed averaging")
        finish(fig, "test-means-controls.png", "policy returns")

        fig, ax = plt.subplots(figsize=(9, 5.4))
        for p, label, color in zip(PROFILES, labels, colors, strict=True):
            ax.plot(
                [2, 4, 8],
                [public["cost_stress_bps_per_side"][c][p]["mean_daily_return_bps"] for c in ("2", "4", "8")],
                marker="o",
                color=color,
                label=label,
            )
        ax.set_xticks([2, 4, 8])
        ax.set_xlabel("Execution cost per side (bps)")
        ax.set_ylabel("Mean daily net policy return (bps)")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title("Cold TEST cost stress: same three fixed seeds per profile")
        ax.legend()
        finish(fig, "cost-stress-means.png", "policy returns")

        fig, axes = plt.subplots(2, 2, figsize=(13, 9.4), sharey=True)
        horizons = public["forecast_grading"]["horizons"]
        for column, role in enumerate(("tune", "test")):
            for row_index, accuracy_key, prior_key, scope in (
                (0, "accuracy_errors_as_incorrect", "prior_full_cohort", "Full cohort; errors counted incorrect"),
                (1, "accuracy_available", "prior_same_available", "Conditional on same available records"),
            ):
                ax = axes[row_index, column]
                for model, label, color, offset in (
                    ("jev", "Jev 1.13", colors[1], -0.18),
                    (prior_key, "TRAIN prior", colors[0], 0.18),
                ):
                    values = [horizons[h][role][model][accuracy_key] for h in ab.HORIZONS]
                    bars = ax.bar(
                        [i + offset for i in range(3)],
                        [v * 100 if v is not None else 0 for v in values],
                        width=0.35,
                        label=label,
                        color=color,
                    )
                    for bar, horizon, value in zip(bars, ab.HORIZONS, values, strict=True):
                        metrics = horizons[horizon][role][model]
                        text = f"a={int(metrics['available_records'])}/{int(metrics['cohort_records'])}"
                        if model == "jev":
                            text += f"\nu={int(metrics['unavailable_records'])} err={metrics['error_rate']:.0%}"
                            text += f"\ncov={metrics['coverage_0_6']:.0%}"
                        if value is None:
                            text += "\nNO AVAILABLE"
                        ax.annotate(
                            text,
                            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 4),
                            textcoords="offset points",
                            ha="center",
                            fontsize=7.5,
                        )
                ax.set_xticks(range(3), ["15 min", "60 min", "Session close"])
                ax.set_ylim(0, 130)
                ax.set_yticks([0, 20, 40, 60, 80, 100])
                ax.set_title(f"{role.upper()} forecast accuracy\n{scope}", fontsize=10)
                ax.legend(loc="upper right", fontsize=8)
        axes[0, 0].set_ylabel("Directional forecast accuracy (%)")
        axes[1, 0].set_ylabel("Directional forecast accuracy (%)")
        fig.suptitle(
            "Forecast quality is separate from policy returns; all market episodes retained\n"
            "a=available/cohort, u=unavailable; coverage uses full cohort; losses use available records only",
            fontsize=11,
        )
        finish(fig, "forecast-accuracy-coverage.png", "forecast accuracy")
    return output / "summary.json"


def absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("absolute path required")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=absolute_path, required=True)
    parser.add_argument("--cache", type=absolute_path, required=True)
    parser.add_argument("--output-dir", type=absolute_path, required=True)
    args = parser.parse_args()
    print(render(args.report, args.output_dir, args.cache))


if __name__ == "__main__":
    main()
