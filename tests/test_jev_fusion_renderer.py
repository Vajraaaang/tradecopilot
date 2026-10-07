"""Synthetic-only fusion rendering: integrity, cohort denominators and exploratory privacy."""

from __future__ import annotations

import copy
import importlib.util
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from tradecopilot.forecast.contracts import content_hash


def renderer() -> Any:
    path = Path(__file__).resolve().parents[1] / "scripts" / "render_jev_fusion_results.py"
    spec = importlib.util.spec_from_file_location("fusion_renderer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def seal(value: dict[str, Any], identity: str) -> dict[str, Any]:
    return {**value, identity: content_hash(value)}


def descriptor(path: Path) -> dict[str, Any]:
    blob = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(blob), "sha256": sha256(blob).hexdigest()}


def metric(n: int, available: int, accuracy: float, loss: float) -> dict[str, Any]:
    return {
        "eligible": n,
        "scored": n,
        "coverage": 1 if n else 0,
        "jev_available": available,
        "jev_availability_coverage": available / n if n else 0,
        "accuracy": accuracy if n else None,
        "balanced_accuracy": 0.5 if n else None,
        "macro_f1": 0.5 if n else None,
        "log_loss": loss if n else None,
        "brier_score": 0.6 if n else None,
        "prices": [123.45],
        "labels": ["UP"],
        "probabilities": [[0.1, 0.2, 0.7]],
        "feature_rows": [[3, 4]],
        "source_url": "https://private.test/source",
        "page_token": "fixture-secret",
        "private_path": "/private/source.json",
    }


@pytest.fixture
def bundle(tmp_path: Path) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    module = renderer()
    root = tmp_path.resolve() / "private-fusion"
    root.mkdir()
    private_input = tmp_path.resolve() / "private-input.json"
    private_input.write_text('{"price":123,"source_url":"https://private.test","page_token":"fixture-secret"}')
    inputs = [descriptor(private_input)]
    ordered = [digest(f"case-{i}") for i in range(20)]
    protocol = seal(
        {
            "schema_version": "jev-fusion-exploration-v1",
            "evidence_mode": "consumed_date_exploration",
            "confirmatory": False,
            "fresh_model_selection_allowed": False,
            "all_old_dates_consumed": True,
            "case_count": 20,
            "target": {"horizon_minutes": 15, "flat_threshold_bps_inclusive": 10},
            "arms": list(module.ARMS),
            "settings": dict(module.SETTINGS),
            "availability_mode": "hypothetical_as_of_plus_60_seconds",
            "cache_id": digest("cache"),
            **{k: digest(k) for k in ("source_data_id", "old_prepared_data_id", "old_registration_id")},
            "ordered_case_ids": ordered,
            "inputs": inputs,
            "private_path": "/private/registry.json",
        },
        "protocol_id",
    )
    model = seal(
        {
            "schema_version": "jev-fusion-models-v1",
            "protocol_id": protocol["protocol_id"],
            "classes": ["DOWN", "FLAT", "UP"],
            "settings": dict(module.SETTINGS),
            "normalizer": {"mean": [9, 8, 7]},
            "coefficients": [[1, 2, 3]],
            "training_ids": ordered[:10],
        },
        "model_id",
    )
    predictions = seal(
        {
            "schema_version": "jev-fusion-predictions-v1",
            "model_id": model["model_id"],
            "class_order": ["DOWN", "FLAT", "UP"],
            "records": [
                {
                    "case_id": identity,
                    "label": "UP",
                    "probabilities": {a: [0.1, 0.2, 0.7] for a in module.ARMS},
                    "price": 123,
                    "features": [3, 4],
                    "page_token": "fixture-secret",
                }
                for identity in ordered
            ],
        },
        "predictions_id",
    )
    for filename, value in (("protocol.json", protocol), ("models.json", model), ("predictions.json", predictions)):
        (root / filename).write_text(json.dumps(value))
    metrics: dict[str, dict[str, Any]] = {arm: {} for arm in module.ARMS}
    for arm, accuracy, loss in zip(module.ARMS, (0.2, 0.6, 0.4, 0.6), (1.3, 1.2, 1.1, 0.9), strict=True):
        for role, n in (("train", 10), ("tune", 5), ("test", 5)):
            metrics[arm][role] = {
                "all_cases": metric(n, n - 1, accuracy, loss),
                "matched_available": metric(n - 1, n - 1, round(accuracy * n) / (n - 1), loss),
            }
    report = seal(
        {
            "schema_version": "jev-fusion-report-v1",
            "evidence_mode": "consumed_date_exploration",
            "confirmatory": False,
            "fresh_model_selection_allowed": False,
            "protocol_id": protocol["protocol_id"],
            "model_id": model["model_id"],
            "predictions_id": predictions["predictions_id"],
            "cache_id": digest("cache"),
            "case_count": 20,
            "case_order_hash": content_hash(ordered),
            "split_counts": {"train": 10, "tune": 5, "test": 5},
            "split_dates": {"train": ["2024-01-02", "2024-01-03"], "tune": ["2024-02-01"], "test": ["2024-03-01"]},
            "metrics": metrics,
            "paired_test_nll": {
                "comparison": "JEV_FUSION_LR minus CONTEXT_LR negative favors fusion",
                "metric": "per_case_log_loss",
                "mean_difference": -0.2,
                "low": -0.3,
                "high": -0.1,
                "resamples": 1000,
                "seed": 42,
                "dates": 1,
                "cases": 5,
                "unit": "whole_date_all_symbols",
                "interpretation": "descriptive_consumed_dates_only",
                "confirmatory": False,
            },
            "inputs": inputs,
            "artifacts": [
                descriptor(root / filename) for filename in ("protocol.json", "models.json", "predictions.json")
            ],
            "provenance": {"private_path": "/private/python", "api_key": "fixture-secret"},
        },
        "report_id",
    )
    report_path = root / "report.json"
    report_path.write_text(json.dumps(report))
    inventory = seal(
        {
            "schema_version": "jev-fusion-inventory-v1",
            "report_id": report["report_id"],
            "artifacts": [descriptor(p) for p in sorted(root.iterdir())],
        },
        "inventory_id",
    )
    (root / "inventory.json").write_text(json.dumps(inventory))
    return report_path, report, protocol, inventory


def test_sealed_bundle_and_public_allowlist_evidence_and_denominators(bundle: Any) -> None:
    path, _, _, _ = bundle
    module = renderer()
    public = module.public_summary(*module.load_bundle(path))
    encoded = json.dumps(public)
    for forbidden in (
        "fixture-secret",
        "/private/",
        "page_token",
        "source_url",
        "private_path",
        "coefficients",
        "normalizer",
        "probabilities",
        "ordered_case_ids",
        "training_ids",
        "feature_rows",
        '"prices":',
        '"labels":',
    ):
        assert forbidden not in encoded
    assert set(public) == {
        "schema_version",
        "evidence",
        "confirmatory",
        "fresh_model_selection_allowed",
        "identities",
        "method",
        "arms",
        "settings",
        "target",
        "splits",
        "availability",
        "metrics",
        "paired_test_nll",
        "raw_jev_available_only_test",
        "raw_jev_fallback",
        "matched_available_definition",
        "shared_metadata",
        "fusion_features",
        "metric_units",
        "source_attribution",
        "availability_mode",
        "new_jev_calls",
        "jev_weights_trained",
        "public_id",
    }
    assert public["evidence"] == "exploratory_consumed_dates; not fresh confirmation; no Jev fine-tuning or new calls"
    assert not public["confirmatory"] and not public["fresh_model_selection_allowed"]
    assert public["new_jev_calls"] == 0 and not public["jev_weights_trained"]
    assert public["raw_jev_available_only_test"]["accuracy"] == 0.75
    assert public["metrics"]["RAW_JEV"]["test"]["all_cases"]["accuracy"] == 0.6
    assert public["availability"]["test"] == {
        "all_cases": 5,
        "available": 4,
        "unavailable": 1,
        "availability_coverage": 0.8,
    }
    assert all(public["metrics"][arm]["test"]["all_cases"]["eligible"] == 5 for arm in module.ARMS)
    assert all(public["metrics"][arm]["test"]["matched_available"]["eligible"] == 4 for arm in module.ARMS)
    assert public["paired_test_nll"]["mean_difference"] == pytest.approx(-0.2)
    assert "negative favors fusion" in public["paired_test_nll"]["comparison"]
    assert "TRAIN class prior" in public["raw_jev_fallback"]


@pytest.mark.parametrize(
    "mutation",
    [
        "fresh_confirmation",
        "selection",
        "unavailable_dropped",
        "unmatched_available",
        "positive_difference",
        "percent_accuracy",
        "empty_accuracy",
        "settings",
    ],
)
def test_false_evidence_or_denominators_refused(bundle: Any, mutation: str) -> None:
    _, report, protocol, inventory = bundle
    if mutation == "fresh_confirmation":
        report["confirmatory"] = True
    elif mutation == "selection":
        protocol["fresh_model_selection_allowed"] = True
    elif mutation == "unavailable_dropped":
        report["metrics"]["RAW_JEV"]["test"]["all_cases"]["scored"] = 4
    elif mutation == "unmatched_available":
        report["metrics"]["JEV_FUSION_LR"]["test"]["matched_available"] = metric(3, 3, 0.5, 1)
    elif mutation == "positive_difference":
        report["paired_test_nll"]["mean_difference"] = 0.2
    elif mutation == "percent_accuracy":
        report["metrics"]["RAW_JEV"]["test"]["all_cases"]["accuracy"] = 60
    elif mutation == "empty_accuracy":
        row = metric(0, 0, 0, 0)
        row["accuracy"] = 0
        report["metrics"]["RAW_JEV"]["test"]["matched_available"] = row
    else:
        protocol["settings"]["C"] = 1
    with pytest.raises(ValueError):
        renderer().public_summary(report, protocol, inventory)


@pytest.mark.parametrize("mutation", ["artifact_bytes", "seal", "extra_file", "source_input", "symlink"])
def test_integrity_failure_prevents_output(bundle: Any, mutation: str) -> None:
    path, _, _, _ = bundle
    root = path.parent
    if mutation == "artifact_bytes":
        (root / "predictions.json").write_text("tampered")
    elif mutation == "seal":
        value = json.loads((root / "models.json").read_text())
        value["coefficients"] = [[9]]
        (root / "models.json").write_text(json.dumps(value))
    elif mutation == "extra_file":
        (root / "private-extra.json").write_text("secret")
    elif mutation == "source_input":
        (root.parent / "private-input.json").write_text("changed source")
    else:
        target = root / "models.json"
        destination = root.parent / "relocated-model.json"
        target.rename(destination)
        target.symlink_to(destination)
    output = root.parent / "public"
    with pytest.raises(ValueError):
        renderer().render(path, output)
    assert not output.exists()


def test_render_png_summary_and_no_overwrite(bundle: Any) -> None:
    path, _, _, _ = bundle
    module = renderer()
    output = path.parent.parent / "public"
    assert module.render(path, output) == output / "summary.json"
    assert {p.name for p in output.iterdir()} == {"summary.json", "test-four-fixed-arms.png"}
    assert (output / "test-four-fixed-arms.png").read_bytes().startswith(b"\x89PNG")
    assert "fixture-secret" not in (output / "summary.json").read_text()
    with pytest.raises(ValueError, match="fresh"):
        module.render(path, output)
    with pytest.raises(ValueError, match="outside"):
        module.render(path, path.parent / "nested-public")


def test_empty_available_subset_is_na_not_zero_accuracy(bundle: Any) -> None:
    _, report, protocol, inventory = copy.deepcopy(bundle)
    for arm in renderer().ARMS:
        report["metrics"][arm]["test"]["all_cases"]["jev_available"] = 0
        report["metrics"][arm]["test"]["all_cases"]["jev_availability_coverage"] = 0
        report["metrics"][arm]["test"]["matched_available"] = metric(0, 0, 0, 0)
    public = renderer().public_summary(report, protocol, inventory)
    assert public["raw_jev_available_only_test"]["accuracy"] is None
    assert public["availability"]["test"]["unavailable"] == 5
    assert public["metrics"]["RAW_JEV"]["test"]["all_cases"]["scored"] == 5


def test_conditional_accuracy_cannot_hide_incorrect_full_cohort(bundle: Any) -> None:
    _, report, protocol, inventory = bundle
    report["metrics"]["RAW_JEV"]["test"]["all_cases"]["accuracy"] = 0.2
    with pytest.raises(ValueError, match="cannot contradict"):
        renderer().public_summary(report, protocol, inventory)
