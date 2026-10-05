"""Render sealed consumed-date Jev fusion aggregates without reading outcomes for scoring."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Any

from tradecopilot.forecast.contracts import content_hash

ARMS = ("TRAIN_PRIOR", "RAW_JEV", "CONTEXT_LR", "JEV_FUSION_LR")
ROLES = ("train", "tune", "test")
FILES = {"protocol.json", "models.json", "predictions.json", "report.json", "inventory.json"}
SETTINGS = {
    "C": 0.01,
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 42,
    "class_weight": None,
    "threads": 1,
    "objective": "unweighted_multinomial_log_loss",
}
EVIDENCE = "exploratory_consumed_dates; not fresh confirmation; no Jev fine-tuning or new calls"


def _number(value: Any, *, fraction: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("finite numeric aggregate required")
    result = float(value)
    if fraction and not 0 <= result <= 1:
        raise ValueError("accuracy and coverage must be fractions")
    return result


def _count(value: Any) -> int:
    result = _number(value)
    if result < 0 or result % 1:
        raise ValueError("nonnegative integer count required")
    return int(result)


def _path(path: Path) -> Path:
    if not path.is_absolute() or ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("absolute artifact path without symlinks required")
    return path.resolve()


def _sealed(path: Path, identity: str) -> dict[str, Any]:
    value: Any = json.loads(_path(path).read_bytes())
    if not isinstance(value, dict) or value.get(identity) != content_hash(
        {k: v for k, v in value.items() if k != identity}
    ):
        raise ValueError("invalid fusion content seal")
    return value


def _descriptor(value: dict[str, Any], root: Path | None = None) -> str:
    path = _path(Path(value["path"]))
    if root is not None and path.parent != root:
        raise ValueError("bundle artifact outside sealed directory")
    blob = path.read_bytes()
    if len(blob) != value["bytes"] or sha256(blob).hexdigest() != value["sha256"]:
        raise ValueError("fusion artifact integrity mismatch")
    return path.name if root is not None else str(path)


def load_bundle(report_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate every seal and inventory binding; never fit, predict or score private rows."""
    report_path = _path(report_path)
    root = report_path.parent
    if report_path.name != "report.json" or {p.name for p in root.iterdir()} != FILES:
        raise ValueError("complete immutable five-file fusion bundle required")
    inventory = _sealed(root / "inventory.json", "inventory_id")
    report = _sealed(report_path, "report_id")
    protocol = _sealed(root / "protocol.json", "protocol_id")
    model = _sealed(root / "models.json", "model_id")
    predictions = _sealed(root / "predictions.json", "predictions_id")
    if inventory.get("schema_version") != "jev-fusion-inventory-v1" or inventory["report_id"] != report["report_id"]:
        raise ValueError("fusion inventory/report binding mismatch")
    names = [_descriptor(item, root) for item in inventory["artifacts"]]
    if len(names) != 4 or set(names) != FILES - {"inventory.json"}:
        raise ValueError("full unique fusion artifact inventory required")
    artifacts = [_descriptor(item, root) for item in report["artifacts"]]
    if len(artifacts) != 3 or set(artifacts) != FILES - {"inventory.json", "report.json"}:
        raise ValueError("report artifact inventory mismatch")
    descriptors = {item["path"]: item for item in inventory["artifacts"]}
    if any(descriptors.get(item["path"]) != item for item in report["artifacts"]):
        raise ValueError("report/inventory descriptor mismatch")
    for name, document, identity in (
        ("protocol_id", protocol, "protocol_id"),
        ("model_id", model, "model_id"),
        ("predictions_id", predictions, "predictions_id"),
    ):
        if report[name] != document[identity]:
            raise ValueError("fusion report identity mismatch")
    if (
        report["cache_id"] != protocol["cache_id"]
        or report["case_order_hash"] != content_hash(protocol["ordered_case_ids"])
        or len(predictions["records"]) != protocol["case_count"]
        or content_hash([row["case_id"] for row in predictions["records"]]) != report["case_order_hash"]
    ):
        raise ValueError("fusion cache or ordered cohort binding mismatch")
    if model["protocol_id"] != protocol["protocol_id"] or predictions["model_id"] != model["model_id"]:
        raise ValueError("fusion model/prediction/protocol binding mismatch")
    if (
        model.get("schema_version") != "jev-fusion-models-v1"
        or predictions.get("schema_version") != "jev-fusion-predictions-v1"
        or model["settings"] != protocol["settings"]
        or model["classes"] != ["DOWN", "FLAT", "UP"]
        or predictions["class_order"] != model["classes"]
    ):
        raise ValueError("fusion model schema/settings mismatch")
    # Hashing source inputs verifies provenance, without accessing providers or grading records.
    if protocol["inputs"] != report["inputs"]:
        raise ValueError("fusion frozen input descriptor mismatch")
    input_paths = [_descriptor(item) for item in report["inputs"]]
    if len(input_paths) != len(set(input_paths)):
        raise ValueError("duplicate frozen source input descriptor")
    return report, protocol, inventory


def _metrics(value: dict[str, Any]) -> dict[str, Any]:
    n, scored, available = (_count(value[k]) for k in ("eligible", "scored", "jev_available"))
    if scored != n or available > n:
        raise ValueError("all eligible fusion cases must remain scored")
    coverage = _number(value["coverage"], fraction=True)
    availability = _number(value["jev_availability_coverage"], fraction=True)
    if coverage != (1 if n else 0) or not math.isclose(availability, available / n if n else 0, abs_tol=1e-12):
        raise ValueError("fusion coverage denominator mismatch")
    result: dict[str, Any] = {
        "eligible": n,
        "scored": scored,
        "coverage": coverage,
        "jev_available": available,
        "jev_availability_coverage": availability,
    }
    for name in ("accuracy", "balanced_accuracy", "macro_f1", "log_loss", "brier_score"):
        metric = value[name]
        if (metric is None) != (n == 0):
            raise ValueError("empty available subset requires N/A quality")
        result[name] = None if metric is None else _number(metric, fraction=name not in ("log_loss", "brier_score"))
        if result[name] is not None and result[name] < 0:
            raise ValueError("negative proper loss")
    if n and not math.isclose(result["accuracy"] * n, round(result["accuracy"] * n), abs_tol=1e-9):
        raise ValueError("accuracy must match an integer correct-case count")
    return result


def public_summary(report: dict[str, Any], protocol: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    """Explicit aggregate allowlist, including every unavailable case and both denominators."""
    if (
        report.get("schema_version") != "jev-fusion-report-v1"
        or protocol.get("schema_version") != "jev-fusion-exploration-v1"
        or any(
            d.get("evidence_mode") != "consumed_date_exploration"
            or d.get("confirmatory") is not False
            or d.get("fresh_model_selection_allowed") is not False
            for d in (report, protocol)
        )
        or protocol.get("all_old_dates_consumed") is not True
    ):
        raise ValueError("fusion evidence must remain descriptive consumed-date exploration")
    if protocol["arms"] != list(ARMS) or protocol["settings"] != SETTINGS or set(report["metrics"]) != set(ARMS):
        raise ValueError("four fixed registered arms/settings required")
    if protocol["target"]["horizon_minutes"] != 15 or protocol["target"]["flat_threshold_bps_inclusive"] != 10:
        raise ValueError("fixed 15-minute / 10-bps target required")
    if protocol["availability_mode"] != "hypothetical_as_of_plus_60_seconds":
        raise ValueError("hypothetical retrospective availability mode required")
    if report["case_count"] != protocol["case_count"]:
        raise ValueError("fusion full cohort identity mismatch")
    splits: dict[str, dict[str, Any]] = {}
    availability = {}
    metrics: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    for role in ROLES:
        n = _count(report["split_counts"][role])
        dates = [date.fromisoformat(v).isoformat() for v in report["split_dates"][role]]
        if not n or not dates or dates != sorted(set(dates)):
            raise ValueError("nonempty unique registered split dates required")
        reference: dict[str, Any] | None = None
        for arm in ARMS:
            source = report["metrics"][arm][role]
            all_cases, matched = _metrics(source["all_cases"]), _metrics(source["matched_available"])
            if (
                all_cases["eligible"] != n
                or matched["eligible"] != all_cases["jev_available"]
                or matched["jev_available"] != matched["eligible"]
            ):
                raise ValueError("matched available subset must retain the same denominator for every arm")
            if reference is not None and all_cases["jev_available"] != reference["jev_available"]:
                raise ValueError("Jev availability cohort differs between fixed arms")
            if matched["eligible"]:
                correct = all_cases["accuracy"] * n
                matched_correct = matched["accuracy"] * matched["eligible"]
                if not -1e-9 <= correct - matched_correct <= n - matched["eligible"] + 1e-9:
                    raise ValueError("matched available accuracy cannot contradict the full cohort")
            reference = all_cases
            metrics[arm][role] = {"all_cases": all_cases, "matched_available": matched}
        assert reference is not None
        availability[role] = {
            "all_cases": n,
            "available": reference["jev_available"],
            "unavailable": n - reference["jev_available"],
            "availability_coverage": reference["jev_availability_coverage"],
        }
        splits[role] = {"cases": n, "sessions": len(dates), "date_range": [dates[0], dates[-1]]}
    if sum(s["cases"] for s in splits.values()) != _count(report["case_count"]):
        raise ValueError("fusion split counts do not sum to the full cohort")
    paired = report["paired_test_nll"]
    if (
        paired["confirmatory"] is not False
        or paired["interpretation"] != "descriptive_consumed_dates_only"
        or paired["metric"] != "per_case_log_loss"
        or paired["unit"] != "whole_date_all_symbols"
        or paired["comparison"] != "JEV_FUSION_LR minus CONTEXT_LR negative favors fusion"
        or _count(paired["cases"]) != splits["test"]["cases"]
        or _count(paired["dates"]) != splits["test"]["sessions"]
    ):
        raise ValueError("paired NLL evidence or denominator mismatch")
    paired_public: dict[str, Any] = {k: _number(paired[k]) for k in ("mean_difference", "low", "high")}
    expected = (
        metrics["JEV_FUSION_LR"]["test"]["all_cases"]["log_loss"]
        - metrics["CONTEXT_LR"]["test"]["all_cases"]["log_loss"]
    )
    if (
        not math.isclose(paired_public["mean_difference"], expected, abs_tol=1e-12)
        or paired_public["low"] > paired_public["high"]
    ):
        raise ValueError("paired mean must be fusion minus context; negative favors fusion")
    paired_public.update({k: _count(paired[k]) for k in ("cases", "dates", "seed", "resamples")})
    paired_public.update(
        comparison="JEV_FUSION_LR minus CONTEXT_LR; negative favors fusion",
        confirmatory=False,
        method="whole_date_paired_bootstrap_95pct; descriptive only",
    )
    identities = {
        k: report[k] for k in ("report_id", "protocol_id", "model_id", "predictions_id", "cache_id", "case_order_hash")
    }
    identities.update({k: protocol[k] for k in ("source_data_id", "old_prepared_data_id", "old_registration_id")})
    identities["inventory_id"] = inventory["inventory_id"]
    if any(
        not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
        for v in identities.values()
    ):
        raise ValueError("SHA256 aggregate identities required")
    result: dict[str, Any] = {
        "schema_version": "jev-fusion-public-v1",
        "evidence": EVIDENCE,
        "confirmatory": False,
        "fresh_model_selection_allowed": False,
        "identities": identities,
        "method": "Fixed TRAIN-only multinomial logistic regression; no tuning, calibration, gate or neural fitting",
        "arms": list(ARMS),
        "settings": dict(SETTINGS),
        "target": {
            "horizon_minutes": 15,
            "flat_threshold_bps_inclusive": 10,
            "anchor": "exact cache as_of minute-bar close",
            "outcome": "exact target minute-bar close",
        },
        "splits": splits,
        "availability": availability,
        "metrics": metrics,
        "paired_test_nll": paired_public,
        "raw_jev_available_only_test": metrics["RAW_JEV"]["test"]["matched_available"],
        "raw_jev_fallback": (
            "Raw Jev uses TRAIN class prior for unavailable forecasts; every case remains in all-case metrics"
        ),
        "matched_available_definition": (
            "The identical ok/abstained Jev subset for every arm; conditional available-only quality"
        ),
        "shared_metadata": "Both LR arms include the same Jev status, availability and timing metadata",
        "fusion_features": "Four raw Jev direction/confidence fields added to context; unavailable fields zero",
        "metric_units": "Accuracy/coverage fractions; log loss natural logarithm; multiclass Brier sum",
        "source_attribution": "Alpaca SIP raw bars; cached Jev 1.13 retrospective forecasts; vendor cutoff unknown",
        "availability_mode": "hypothetical_as_of_plus_60_seconds",
        "new_jev_calls": 0,
        "jev_weights_trained": False,
    }
    result["public_id"] = content_hash(result)
    return result


def render(report_path: Path, output: Path) -> Path:
    report_path, output = _path(report_path), _path(output)
    if output.exists():
        raise ValueError("render output requires a fresh directory")
    if output.is_relative_to(report_path.parent):
        raise ValueError("public output must be outside the sealed private fusion bundle")
    public = public_summary(*load_bundle(report_path))
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="fusion-render-mpl-"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(public, sort_keys=True, indent=2, allow_nan=False) + "\n")
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        labels = ("TRAIN\nprior", "Raw Jev +\nprior fallback", "Context\nLR", "Jev fusion\nLR")
        n = public["availability"]["test"]["all_cases"]
        available = public["availability"]["test"]["available"]
        for ax, metric, title in zip(
            axes,
            ("accuracy", "log_loss"),
            ("All-case TEST accuracy", "All-case TEST negative log likelihood"),
            strict=True,
        ):
            values = [public["metrics"][arm]["test"]["all_cases"][metric] for arm in ARMS]
            scaled = [v * 100 if metric == "accuracy" else v for v in values]
            bars = ax.bar(range(4), scaled, color=["#87909b", "#a96c32", "#526782", "#177e89"])
            for bar in bars:
                ax.annotate(
                    f"n={n:,}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                )
            ax.set_xticks(range(4), labels)
            ax.set_title(title)
            ax.set_ylabel("Accuracy (%)" if metric == "accuracy" else "NLL (lower is better)")
            if metric == "accuracy":
                ax.set_ylim(0, 112)
                ax.set_yticks([0, 20, 40, 60, 80, 100])
            else:
                ax.margins(y=0.2)
        raw = public["raw_jev_available_only_test"]["accuracy"]
        raw_text = "N/A" if raw is None else f"{raw:.1%}"
        paired = public["paired_test_nll"]
        fig.suptitle(
            f"Consumed-date 15-minute forecasts: same {n:,} TEST cases in all four fixed arms\n"
            f"Matched available subset n={available:,}; unavailable n={n - available:,} retained; "
            f"raw Jev available-only accuracy {raw_text}",
            fontsize=11,
        )
        fig.text(
            0.5,
            0.098,
            f"TEST NLL fusion minus context = {paired['mean_difference']:+.4f} "
            f"(descriptive 95% CI {paired['low']:+.4f}, {paired['high']:+.4f}); negative favors fusion",
            ha="center",
            fontsize=9,
        )
        fig.text(
            0.5,
            0.056,
            "Exploratory consumed dates; no fresh 80% confirmation, neural claim or Jev fine-tuning; no new calls",
            ha="center",
            fontsize=8.5,
        )
        fig.text(0.5, 0.018, public["source_attribution"], ha="center", fontsize=8, color="#536170")
        fig.tight_layout(rect=(0, 0.145, 1, 0.9))
        fig.savefig(output / "test-four-fixed-arms.png", dpi=180)
        plt.close(fig)
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
