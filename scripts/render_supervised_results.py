"""Render validated supervised forecasts as aggregate-only scientific artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from datetime import date
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from tradecopilot.forecast.contracts import content_hash
from tradecopilot.forecast.selective import GATE_REQUIREMENTS

CLASSES = ("DOWN", "FLAT", "UP")
CANDIDATES = ("prior", "lr-001", "lr-1", "hgb-31", "tcn-32", "lstm-64")
NEURAL = ("tcn-32", "lstm-64")
ROLES = ("TRAIN", "TUNE", "CAL", "GATE", "TEST")


def load_report(path: Path) -> dict[str, Any]:
    loader = import_module("tradecopilot.forecast.supervised_study").load_report
    return cast(dict[str, Any], loader(path))


def _number(value: Any, *, fraction: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("finite numeric aggregate required")
    number = float(value)
    if fraction and not 0 <= number <= 1:
        raise ValueError("accuracy, precision and coverage must use fractions")
    return number


def _count(value: Any) -> int:
    number = _number(value)
    if number < 0 or number % 1:
        raise ValueError("nonnegative integer aggregate required")
    return int(number)


def _optional_fraction(value: Any) -> float | None:
    return None if value is None else _number(value, fraction=True)


def _hash(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("SHA256 identity required")
    return value


def _bounds(value: Any, *, fraction: bool) -> dict[str, float] | None:
    if value is None:
        return None
    result = {key: _number(value[key], fraction=fraction) for key in ("low", "high")}
    if result["low"] > result["high"]:
        raise ValueError("uncertainty interval bounds inverted")
    return result


def _uncertainty(value: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, expected_block_days in (("date_cluster", 1), ("five_date_block_sensitivity", 5)):
        row = value[key]
        out: dict[str, Any] = {
            name: _count(row[name]) for name in ("resamples", "seed", "sessions", "clusters", "block_days")
        }
        if out["block_days"] != expected_block_days or out["sessions"] <= 0 or out["clusters"] <= 0:
            raise ValueError("uncertainty cluster scope mismatch")
        out["method"] = (
            "whole_session_date_bootstrap_95pct"
            if expected_block_days == 1
            else "consecutive_5_date_block_bootstrap_95pct"
        )
        for metric in (
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "coverage",
            "selective_accuracy",
            "log_loss",
            "brier_score",
        ):
            out[metric] = _bounds(row[metric], fraction=metric not in ("log_loss", "brier_score"))
        result[key] = out
    return result


def _selective(value: dict[str, Any]) -> dict[str, Any]:
    eligible, selected = _count(value["eligible"]), _count(value["selected"])
    if not eligible or selected > eligible:
        raise ValueError("nonempty matched eligible/selected counts required")
    coverage = _number(value["coverage"], fraction=True)
    if not math.isclose(coverage, selected / eligible, abs_tol=1e-12):
        raise ValueError("selective coverage denominator mismatch")
    accuracy = _optional_fraction(value["selective_accuracy"])
    if (selected == 0) != (accuracy is None):
        raise ValueError("all abstain requires N/A selective accuracy")
    groups: dict[str, dict[str, Any]] = {}
    for label in CLASSES:
        row = value["by_predicted_label"][label]
        count, precision = _count(row["count"]), _optional_fraction(row["precision"])
        if (count == 0) != (precision is None):
            raise ValueError("selected class precision/count mismatch")
        groups[label] = {"count": count, "precision": precision}
    if sum(row["count"] for row in groups.values()) != selected:
        raise ValueError("selected class counts differ from selected cohort")
    if selected and not math.isclose(
        sum(row["count"] * (row["precision"] or 0) for row in groups.values()) / selected,
        cast(float, accuracy),
        abs_tol=1e-12,
    ):
        raise ValueError("selected accuracy differs from class precision aggregate")
    sessions = _count(value["selected_sessions"])
    if sessions > selected or (selected == 0) != (sessions == 0):
        raise ValueError("selected session counts inconsistent")
    return {
        "eligible": eligible,
        "selected": selected,
        "coverage": coverage,
        "selective_accuracy": accuracy,
        "by_predicted_label": groups,
        "selected_sessions": sessions,
    }


def _group_summary(value: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {key: _count(value[key]) for key in ("eligible", "selected") if key in value}
    for key in ("accuracy", "macro_f1", "balanced_accuracy", "coverage", "selective_accuracy"):
        if key in value:
            result[key] = _optional_fraction(value[key])
    for key in ("log_loss", "brier_score"):
        if key in value:
            result[key] = _number(value[key])
            if result[key] < 0:
                raise ValueError("negative aggregate loss")
    return result


def _metrics(value: dict[str, Any]) -> dict[str, Any]:
    result = _selective(value)
    for key in ("accuracy", "macro_f1", "balanced_accuracy"):
        result[key] = _number(value[key], fraction=True)
    for key in ("log_loss", "brier_score"):
        result[key] = _number(value[key])
        if result[key] < 0:
            raise ValueError("negative aggregate loss")
    result["errors"] = _count(value["errors"])
    result["missing_predictions"] = _count(value["missing_predictions"])
    if result["errors"] or result["missing_predictions"]:
        raise ValueError("complete candidate primary cannot omit eligible predictions")
    matrix = value["confusion_matrix"]
    if (
        not isinstance(matrix, list)
        or len(matrix) != 3
        or any(not isinstance(row, list) or len(row) != 3 for row in matrix)
    ):
        raise ValueError("three-class confusion matrix required")
    matrix = [[_count(n) for n in row] for row in matrix]
    if sum(map(sum, matrix)) != result["eligible"]:
        raise ValueError("confusion matrix cohort mismatch")
    if not math.isclose(sum(matrix[i][i] for i in range(3)) / result["eligible"], result["accuracy"], abs_tol=1e-12):
        raise ValueError("confusion matrix accuracy mismatch")
    result["confusion_matrix"] = matrix
    result["class_order"] = list(CLASSES)
    result["per_class"] = {}
    for index, label in enumerate(CLASSES):
        row = value["per_class"][label]
        support, predicted = _count(row["support"]), _count(row["predicted"])
        if support != sum(matrix[index]) or predicted != sum(r[index] for r in matrix):
            raise ValueError("class confusion counts mismatch")
        result["per_class"][label] = {
            "support": support,
            "predicted": predicted,
            **{k: _optional_fraction(row[k]) for k in ("precision", "recall", "f1")},
        }
    result["target_requirements_met"] = value["target_requirements_met"]
    if not isinstance(result["target_requirements_met"], bool):
        raise ValueError("selective target flag must be boolean")
    if result["target_requirements_met"] != _qualifies(result):
        raise ValueError("selective target flag differs from registered requirements")
    result["by_session"] = {
        date.fromisoformat(key).isoformat(): _group_summary(row) for key, row in value.get("by_session", {}).items()
    }
    result["by_symbol"] = {}
    for key, row in value.get("by_symbol", {}).items():
        if re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,14}", key) is None:
            raise ValueError("invalid aggregate symbol key")
        result["by_symbol"][key] = _group_summary(row)
    if "uncertainty" in value:
        result["uncertainty"] = _uncertainty(value["uncertainty"])
    return result


def _tune_metrics(value: dict[str, Any]) -> dict[str, Any]:
    result = {key: _number(value[key], fraction=True) for key in ("accuracy", "macro_f1", "balanced_accuracy")}
    result.update({key: _number(value[key]) for key in ("log_loss", "brier_score")})
    result["eligible"] = _count(value["eligible"])
    if not result["eligible"] or result["log_loss"] < 0 or result["brier_score"] < 0:
        raise ValueError("invalid TUNE aggregate")
    return result


def _qualifies(value: dict[str, Any]) -> bool:
    req = GATE_REQUIREMENTS
    return bool(
        value["selective_accuracy"] is not None
        and value["selective_accuracy"] >= req["target_accuracy"]
        and value["coverage"] >= req["minimum_coverage"]
        and value["selected"] >= req["minimum_selected"]
        and value["selected_sessions"] >= req["minimum_selected_sessions"]
        and all(
            value["by_predicted_label"][label]["count"] >= req["minimum_direction_count"]
            and value["by_predicted_label"][label]["precision"] is not None
            and value["by_predicted_label"][label]["precision"] >= req["target_accuracy"]
            for label in ("UP", "DOWN")
        )
    )


def _gate(value: dict[str, Any]) -> dict[str, Any]:
    if value["requirements"] != GATE_REQUIREMENTS or not isinstance(value["enabled"], bool):
        raise ValueError("registered gate requirements required")
    threshold = _optional_fraction(value["threshold"])
    if value["enabled"] != (threshold is not None):
        raise ValueError("disabled gate must have no threshold")
    curve = []
    for row in value["curve"]:
        point = {**_selective(row), "threshold": _number(row["threshold"], fraction=True)}
        point["qualifies"] = _qualifies(point)
        if row["qualifies"] != point["qualifies"]:
            raise ValueError("GATE curve qualification mismatch")
        curve.append(point)
    if not curve or len({p["threshold"] for p in curve}) != len(curve):
        raise ValueError("complete unique GATE threshold curve required")
    valid = [p for p in curve if p["qualifies"]]
    expected = max(valid, key=lambda p: (p["coverage"], -p["threshold"])) if valid else None
    if threshold != (expected["threshold"] if expected is not None else None):
        raise ValueError("GATE choice must maximize qualified coverage before TEST")
    metrics = _metrics(value["metrics"])
    if metrics["target_requirements_met"] != _qualifies(metrics) or value["enabled"] != _qualifies(metrics):
        raise ValueError("GATE metrics and enabled state mismatch")
    return {
        "enabled": value["enabled"],
        "threshold": threshold,
        "requirements": dict(GATE_REQUIREMENTS),
        "curve": curve,
        "metrics": metrics,
        "reason": "qualified_on_gate_selection" if value["enabled"] else "target_not_achieved_on_gate_selection",
    }


def public_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Whitelist aggregates; omit source rows, individual probabilities and private paths."""
    scope = "selected_raw_primary_vs_raw_tune_selected_cpu"
    if (
        report.get("schema_version") != "supervised-forecast-report-v1"
        or report.get("evidence_mode") != "historical_project_holdout"
        or report.get("comparison_scope") != scope
        or set(report["candidates"]) != set(CANDIDATES)
    ):
        raise ValueError("unsupported or incomplete registered supervised report")
    evaluation_kind = report["evaluation_kind"]
    if evaluation_kind not in ("historical_project_holdout", "synthetic_control_flow"):
        raise ValueError("unknown evaluation evidence mode")
    selection = report["selection"]
    chosen, reference = selection["selected_candidate_id"], selection["cpu_reference_candidate_id"]
    if chosen not in CANDIDATES or reference not in CANDIDATES[:4] or selection["comparison_scope"] != scope:
        raise ValueError("invalid frozen candidate/reference")
    eligible = {key for key in CANDIDATES if report["candidates"][key]["eligible"] is True}
    if not eligible or chosen not in eligible or reference not in eligible or "prior" not in eligible:
        raise ValueError("complete chosen candidate and CPU reference required")
    if set(report["test"]["candidate_primaries"]) != eligible or set(report["tune"]["candidate_primaries"]) != eligible:
        raise ValueError("failed family cannot have a partial primary ensemble")
    primaries = {key: _metrics(row) for key, row in report["test"]["candidate_primaries"].items()}
    tune = {key: _tune_metrics(row) for key, row in report["tune"]["candidate_primaries"].items()}
    cohorts = {row["eligible"] for row in primaries.values()}
    if len(cohorts) != 1:
        raise ValueError("all TEST candidate primaries must share the identical eligible cohort")
    expected = min((key for key in CANDIDATES if key in eligible), key=lambda key: tune[key]["log_loss"])
    cpu_expected = min((key for key in CANDIDATES[:4] if key in eligible), key=lambda key: tune[key]["log_loss"])
    if chosen != expected or reference != cpu_expected:
        raise ValueError("frozen choice/reference must be selected on TUNE only")
    seeds: dict[str, list[dict[str, Any]]] = {key: [] for key in NEURAL}
    for row in report["training"]["neural_seeds"]:
        key, seed = row["candidate_key"], row["seed"]
        if key not in NEURAL or seed not in (42, 43, 44):
            raise ValueError("unregistered neural seed diagnostic")
        complete = row.get("eligible") is True and row["validation"]["eligible"] is True
        entry: dict[str, Any] = {
            "seed": seed,
            "eligible": complete,
            "status": "complete" if complete else "failed",
            "failure_reason": None if complete else "seed incomplete or invalid",
        }
        for name in ("elapsed_seconds", "epochs_completed", "optimizer_updates", "parameter_count", "best_epoch"):
            entry[name] = None if row.get(name) is None else _number(row[name])
        entry["tune_log_loss"] = None if not complete else _number(row["best_tune_nll"])
        seeds[key].append(entry)
    for key in NEURAL:
        if len(seeds[key]) != 3 or {row["seed"] for row in seeds[key]} != {42, 43, 44}:
            raise ValueError("all three registered neural seed attempts must be reported")
        if report["candidates"][key]["eligible"] != all(row["eligible"] for row in seeds[key]):
            raise ValueError("neural family requires every registered seed")
    candidates = {}
    for key in CANDIDATES:
        row = report["candidates"][key]
        if not isinstance(row["eligible"], bool):
            raise ValueError("candidate eligibility must be boolean")
        representation = (
            "TRAIN class prior"
            if key == "prior"
            else (
                "60x6 causal sequence plus 115 static inputs"
                if key in NEURAL
                else "55 causal OHLCV features plus 5 symbol indicators"
            )
        )
        hashes = [_hash(v) for v in row["artifact_ids"]]
        if key in eligible and len(hashes) != (3 if key in NEURAL else 1):
            raise ValueError("complete primary must bind every registered model artifact")
        candidates[key] = {
            "eligible": row["eligible"],
            "status": "complete" if row["eligible"] else "FAILED",
            "representation": representation,
            "primary_unit": "three-seed probability mean" if key in NEURAL else "single CPU model",
            "artifact_hashes": hashes,
            "input_dimensions": {"sequence": [60, 6], "static": 115}
            if key in NEURAL
            else ({} if key == "prior" else {"static": 60}),
            "failure_reason": None if row["eligible"] else "registered candidate incomplete or invalid",
            "tune": tune.get(key),
            "test": primaries.get(key),
            "seed_diagnostics": seeds.get(key, []),
        }
        if key in eligible and not math.isclose(_number(row["tune_log_loss"]), tune[key]["log_loss"], abs_tol=1e-12):
            raise ValueError("candidate TUNE loss identity mismatch")
    diagnostics = {}
    for key, row in report["test"]["individual_seed_diagnostics"].items():
        if key not in {f"{family}:{seed}" for family in NEURAL for seed in (42, 43, 44)}:
            raise ValueError("unregistered TEST seed diagnostic")
        family, seed = key.split(":")
        if not next(r for r in seeds[family] if r["seed"] == int(seed))["eligible"]:
            raise ValueError("invalid seed cannot have a TEST diagnostic")
        diagnostics[key] = _metrics(row)
        if diagnostics[key]["eligible"] not in cohorts:
            raise ValueError("TEST seed diagnostic scope differs from primary scope")
    raw = _metrics(report["test"]["selected_raw"])
    calibrated = _metrics(report["test"]["selected_calibrated"])
    if raw["eligible"] not in cohorts or calibrated["eligible"] not in cohorts:
        raise ValueError("selected TEST cohort mismatch")
    if (
        raw["confusion_matrix"] != calibrated["confusion_matrix"]
        or raw["confusion_matrix"] != primaries[chosen]["confusion_matrix"]
    ):
        raise ValueError("scalar calibration must preserve chosen raw argmax predictions")
    if any(
        raw[k] != primaries[chosen][k] for k in ("accuracy", "macro_f1", "balanced_accuracy", "log_loss", "brier_score")
    ):
        raise ValueError("selected RAW quality must match frozen chosen candidate primary")
    if report["test"]["prior"] != report["test"]["candidate_primaries"]["prior"]:
        raise ValueError("TRAIN prior identity mismatch")
    if report["test"]["cpu_reference"] != report["test"]["candidate_primaries"][reference]:
        raise ValueError("raw CPU reference identity mismatch")
    paired = report["test"]["paired_reference_improvement"]
    if (
        paired["comparison_scope"] != scope
        or paired["selected_candidate_id"] != chosen
        or paired["reference_candidate_id"] != reference
    ):
        raise ValueError("architecture gain must compare selected RAW primary with RAW CPU reference")
    increment = _number(paired["mean_control_minus_selected_log_loss"])
    if not math.isclose(increment, primaries[reference]["log_loss"] - raw["log_loss"], abs_tol=1e-12):
        raise ValueError("paired raw log loss increment mismatch")
    paired_public: dict[str, Any] = {
        "comparison_scope": scope,
        "selected_candidate": chosen,
        "reference_candidate": reference,
        "mean_reference_minus_selected_log_loss": increment,
    }
    for key, block_days in (("date_cluster", 1), ("five_date_block_sensitivity", 5)):
        row = paired[key]
        if _count(row["block_days"]) != block_days:
            raise ValueError("paired uncertainty block scope mismatch")
        paired_public[key] = {
            **cast(dict[str, float], _bounds(row, fraction=False)),
            **{name: _count(row[name]) for name in ("resamples", "seed", "sessions", "clusters", "block_days")},
        }
    expected_diagnostics = {f"{key}:{row['seed']}" for key, rows in seeds.items() for row in rows if row["eligible"]}
    if set(diagnostics) != expected_diagnostics:
        raise ValueError("every complete TEST seed diagnostic must remain visible")
    gate = _gate(report["gate"])
    flags = report["flags"]
    if (
        any(
            not isinstance(flags[key], bool)
            for key in ("observed_selective_target_met", "improvement_supported", "automatic_production_promotion")
        )
        or flags["automatic_production_promotion"] is not False
    ):
        raise ValueError("no automatic production promotion is permitted")
    observed = gate["enabled"] and gate["metrics"]["target_requirements_met"] and calibrated["target_requirements_met"]
    improvement = increment > 0 and paired_public["date_cluster"]["low"] > 0
    if flags["observed_selective_target_met"] != observed or flags["improvement_supported"] != improvement:
        raise ValueError("outcome flags differ from frozen TEST/GATE evidence")
    temperature = _number(selection["temperature"])
    if not 0.25 <= temperature <= 4 or report["calibration"]["temperature"] != temperature:
        raise ValueError("invalid frozen calibration temperature")
    splits = {
        role: {
            "planned_sessions": _count(report["split"][role]["planned_sessions"]),
            "eligible_cases": _count(report["split"][role]["eligible_cases"]),
            "case_ids_hash": _hash(report["split"][role]["case_ids_hash"]),
        }
        for role in ROLES
    }
    if splits["TEST"]["eligible_cases"] not in cohorts:
        raise ValueError("registered TEST count differs from scored cohort")
    reasons = (
        "sequence_missing_minute",
        "sequence_not_available",
        "target_missing_minute",
        "target_not_available",
        "insufficient_history",
        "stale_source",
        "history_gap",
        "missing_return_endpoint",
        "previous_session_unavailable",
    )
    exclusions = {
        role: {key: _count(value) for key, value in report["exclusions"].get(role, {}).items() if key in reasons}
        for role in ROLES
    }
    catalog = {}
    for role in ROLES:
        rows = report["catalog_counts"].get(role, [])
        catalog[role] = {key: sum(_count(row[key]) for row in rows) for key in ("planned", "eligible", "excluded")}
        catalog[role]["symbol_session_groups"] = len(rows)
    constraints = {"new_jev_calls": 0, "broker_orders": 0, "no_test_retuning": True, "default_model_changed": False}
    if any(report["constraints"].get(key) != value for key, value in constraints.items()):
        raise ValueError("registered display constraints violated")
    result: dict[str, Any] = {
        "schema_version": "supervised-forecast-public-v1",
        "scope": "historical_project_holdout",
        "evaluation_kind": evaluation_kind,
        "interpretation": (
            "Historical project holdout; prospective deployment and profitable execution are unconfirmed."
        ),
        "metric_units": "accuracy/coverage/precision fractions; NLL natural logarithm; multiclass Brier sum",
        "source_attribution": "Alpaca SIP raw market bars; direct supervised forecasts",
        "identities": {
            key: _hash(report[key])
            for key in (
                "report_id",
                "registration_id",
                "prepared_data_id",
                "source_data_id",
                "config_id",
                "normalizer_id",
                "prefit_id",
            )
        },
        "source_provenance_hash": content_hash(report["source_provenance"]),
        "splits": splits,
        "exclusions": exclusions,
        "catalog_counts": catalog,
        "selection": {
            "selection_id": _hash(selection["selection_id"]),
            "candidate": chosen,
            "cpu_reference": reference,
            "criterion": "lowest TUNE raw primary NLL; fixed before TEST",
            "temperature": temperature,
            "temperature_fit_role": "CAL",
            "comparison_scope": scope,
        },
        "candidates": candidates,
        "individual_test_seed_diagnostics": diagnostics,
        "selected_raw": raw,
        "selected_calibrated": calibrated,
        "paired_raw_reference_improvement": paired_public,
        "gate": gate,
        "flags": {
            key: flags[key]
            for key in ("observed_selective_target_met", "improvement_supported", "automatic_production_promotion")
        },
        "constraints": {
            "new_jev_calls": 0,
            "broker_orders": 0,
            "no_test_retuning": True,
            "default_model_changed": False,
        },
    }
    result["public_id"] = content_hash(result)
    return result


def render(report_path: Path, output: Path) -> Path:
    if not report_path.is_absolute() or not output.is_absolute():
        raise ValueError("absolute report and output paths required")
    if output.exists():
        raise ValueError("render output requires a fresh directory")
    if output.resolve().is_relative_to(report_path.parent.resolve()):
        raise ValueError("public output must be outside the sealed study bundle")
    public = public_summary(load_report(report_path))
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="supervised-render-mpl-"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(public, indent=2, sort_keys=True, allow_nan=False) + "\n")
    palette = {key: "#177e89" if key in NEURAL else "#526782" for key in CANDIDATES}
    gate_status = (
        "Selective target observed" if public["flags"]["observed_selective_target_met"] else "SELECTIVE TARGET NOT MET"
    )

    def finish(fig: Any, filename: str) -> None:
        fig.text(
            0.5,
            0.048,
            f"{public['evaluation_kind'].replace('_', ' ')} | {gate_status} | No automatic production promotion",
            ha="center",
            fontsize=8.5,
        )
        fig.text(0.5, 0.017, public["source_attribution"], ha="center", fontsize=8, color="#536170")
        fig.tight_layout(rect=(0, 0.095, 1, 0.94))
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)

    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
        for ax, metric, title in zip(
            axes,
            ("accuracy", "log_loss"),
            ("All-case TEST accuracy (%)", "All-case TEST negative log likelihood"),
            strict=True,
        ):
            for index, key in enumerate(CANDIDATES):
                row = public["candidates"][key]
                if row["eligible"]:
                    value = row["test"][metric] * (100 if metric == "accuracy" else 1)
                    ax.bar(index, value, color=palette[key])
                    ax.annotate(
                        f"n={row['test']['eligible']:,}",
                        (index, value),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center",
                        fontsize=8,
                    )
                else:
                    ax.text(index, 0, "FAILED", ha="center", va="bottom", rotation=90, color="#a33b35", fontsize=9)
            ax.set_xticks(range(6), CANDIDATES)
            ax.set_title(title)
            ax.set_ylabel("Accuracy (%)" if metric == "accuracy" else "NLL (lower is better)")
            ax.margins(y=0.15)
            if metric == "accuracy":
                ax.set_ylim(0, 113)
                ax.set_yticks([0, 20, 40, 60, 80, 100])
        fig.suptitle(
            "Complete candidates share one cold TEST cohort; neural primaries average three probability distributions"
        )
        finish(fig, "test-all-candidates.png")

        fig, (ax, seed_ax) = plt.subplots(1, 2, figsize=(12, 5.8))
        chosen = public["selection"]["candidate"]
        for index, key in enumerate(CANDIDATES):
            row = public["candidates"][key]
            if row["eligible"]:
                ax.bar(
                    index,
                    row["tune"]["log_loss"],
                    color=palette[key],
                    edgecolor="black" if key == chosen else "none",
                    lw=2,
                )
            else:
                ax.text(index, 0, "FAILED", ha="center", va="bottom", rotation=90, color="#a33b35")
        ax.set_xticks(range(6), CANDIDATES)
        ax.set_ylabel("TUNE primary NLL (lower is better)")
        ax.set_title(f"Frozen TUNE choice: {chosen}\nBlack outline identifies chosen primary")
        for family_index, (key, color) in enumerate(zip(NEURAL, ("#177e89", "#a96c32"), strict=True)):
            rows = sorted(public["candidates"][key]["seed_diagnostics"], key=lambda row: row["seed"])
            seed_ax.plot(
                [r["seed"] for r in rows],
                [r["tune_log_loss"] if r["eligible"] else math.nan for r in rows],
                marker="o",
                color=color,
                label=key,
            )
            for row in rows:
                if not row["eligible"]:
                    seed_ax.text(
                        row["seed"],
                        0.02 + family_index * 0.09,
                        f"{key}\nFAILED",
                        transform=seed_ax.get_xaxis_transform(),
                        color=color,
                        ha="center",
                        fontsize=8,
                    )
        seed_losses = [
            r["tune_log_loss"] for key in NEURAL for r in public["candidates"][key]["seed_diagnostics"] if r["eligible"]
        ]
        seed_ax.set_ylim(0, max(seed_losses, default=1) * 1.25)
        seed_ax.set_xticks([42, 43, 44])
        seed_ax.set_xlabel("Registered seed")
        seed_ax.set_ylabel("Individual seed best complete TUNE NLL")
        seed_ax.set_title("All seed attempts; diagnostics only\nIncomplete family never forms a partial ensemble")
        seed_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=9)
        finish(fig, "tune-choice-seed-diagnostics.png")

        fig, (ax, precision_ax) = plt.subplots(1, 2, figsize=(12, 6.5))
        curve = public["gate"]["curve"]
        thresholds = [row["threshold"] for row in curve]
        for key, label, color in (
            ("coverage", "Full-cohort coverage", "#526782"),
            ("selective_accuracy", "Selected accuracy", "#177e89"),
        ):
            ax.plot(
                thresholds,
                [row[key] * 100 if row[key] is not None else math.nan for row in curve],
                marker="o",
                color=color,
                label=label,
            )
        ax.axhline(80, color="#a33b35", ls="--", label="80% accuracy requirement")
        ax.axhline(10, color="#8a623c", ls=":", label="10% coverage requirement")
        for label, color in (("UP", "#177e89"), ("DOWN", "#a96c32")):
            precision_ax.plot(
                thresholds,
                [
                    row["by_predicted_label"][label]["precision"] * 100
                    if row["by_predicted_label"][label]["precision"] is not None
                    else math.nan
                    for row in curve
                ],
                marker="o",
                color=color,
                label=f"Selected {label} precision",
            )
        precision_ax.axhline(80, color="#a33b35", ls="--", label="80% direction requirement")
        if public["gate"]["enabled"]:
            for axis in (ax, precision_ax):
                axis.axvline(public["gate"]["threshold"], color="black", lw=1, ls=":")
        for axis in (ax, precision_ax):
            axis.set_xlim(min(thresholds) - 0.02, max(thresholds) + 0.02)
            axis.set_ylim(0, 105)
            axis.set_xlabel("GATE selection probability threshold")
            axis.set_ylabel("Percent (%)")
            axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=8, ncol=2)
        ax.set_title("GATE coverage and selected accuracy\nMissing accuracy means no selected forecasts")
        precision_ax.set_title("GATE direction precision\nMissing precision means no selected direction forecasts")
        gate = public["gate"]
        active = gate["metrics"]
        accuracy = "N/A" if active["selective_accuracy"] is None else f"{active['selective_accuracy']:.1%}"
        fig.suptitle(
            f"GATE {'ENABLED' if gate['enabled'] else 'DISABLED / ALL ABSTAIN'}: selected {active['selected']:,}/"
            f"{active['eligible']:,}, coverage {active['coverage']:.1%}, selected accuracy {accuracy}\n"
            "Also required: ≥100 selected, ≥30 UP and DOWN each, ≥5 selected sessions; no TEST re-selection",
            fontsize=10,
        )
        finish(fig, "gate-coverage-direction-precision.png")
    return output / "summary.json"


def absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("absolute path required")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=absolute_path, required=True)
    parser.add_argument("--output-dir", type=absolute_path, required=True)
    args = parser.parse_args()
    print(render(args.report, args.output_dir))


if __name__ == "__main__":
    main()
