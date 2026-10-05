"""Synthetic supervised renderer privacy, selection, failure and scientific-unit contracts."""

from __future__ import annotations

import copy
import importlib.util
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest


def renderer() -> Any:
    path = Path(__file__).resolve().parents[1] / "scripts" / "render_supervised_results.py"
    spec = importlib.util.spec_from_file_location("supervised_renderer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def selective(selected: int = 300, precision: float = 0.5) -> dict[str, Any]:
    return {
        "eligible": 300,
        "selected": selected,
        "coverage": selected / 300,
        "selective_accuracy": precision if selected else None,
        "by_predicted_label": {
            label: {"count": selected // 3, "precision": precision if selected else None}
            for label in ("DOWN", "FLAT", "UP")
        },
        "selected_sessions": 10 if selected else 0,
    }


def metrics(loss: float, selected: int = 300, uncertainty: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {
        **selective(selected),
        "accuracy": 0.5,
        "macro_f1": 0.5,
        "balanced_accuracy": 0.5,
        "log_loss": loss,
        "brier_score": 0.6,
        "confusion_matrix": [[50, 25, 25], [25, 50, 25], [25, 25, 50]],
        "per_class": {
            label: {"support": 100, "predicted": 100, "precision": 0.5, "recall": 0.5, "f1": 0.5}
            for label in ("DOWN", "FLAT", "UP")
        },
        "errors": 0,
        "missing_predictions": 0,
        "target_requirements_met": False,
        "by_session": {
            "2024-03-01": {
                "eligible": 300,
                "selected": selected,
                "accuracy": 0.5,
                "log_loss": loss,
                "selective_accuracy": 0.5 if selected else None,
            }
        },
        "by_symbol": {"AAA": {"eligible": 300, "selected": selected, "accuracy": 0.5, "log_loss": loss}},
        "prices": [123, 456],
        "features": [[9, 8]],
        "targets": [0, 1, 2],
        "probabilities": [[0.2, 0.3, 0.5]],
        "private_path": "/private/checkpoint.pkl",
        "api_key": "secret-token",
    }
    if uncertainty:
        result["uncertainty"] = {}
        for name, block in (("date_cluster", 1), ("five_date_block_sensitivity", 5)):
            result["uncertainty"][name] = {
                "resamples": 1000,
                "seed": 42,
                "sessions": 10,
                "clusters": 10 // block,
                "block_days": block,
                **{key: {"low": 0.4, "high": 0.6} for key in ("accuracy", "macro_f1", "balanced_accuracy")},
                "coverage": {"low": selected / 300, "high": selected / 300},
                "selective_accuracy": {"low": 0.4, "high": 0.6} if selected else None,
                "log_loss": {"low": max(0, loss - 0.1), "high": loss + 0.1},
                "brier_score": {"low": 0.5, "high": 0.7},
                "private_path": "/private/bootstrap.npy",
            }
    return result


@pytest.fixture
def report() -> dict[str, Any]:
    module = renderer()
    candidate_keys = module.CANDIDATES
    tune_losses = {"prior": 1.2, "lr-001": 1.0, "lr-1": 0.8, "hgb-31": 0.9, "tcn-32": 0.7}
    test_losses = {"prior": 1.2, "lr-001": 1.0, "lr-1": 0.9, "hgb-31": 0.4, "tcn-32": 0.8}
    candidates = {
        key: {
            "eligible": key != "lstm-64",
            "representation": "private_path_should_not_copy",
            "artifact_ids": [digest(f"{key}:{seed}") for seed in (42, 43, 44)]
            if key in module.NEURAL
            else [digest(key)],
            "tune_log_loss": tune_losses.get(key),
            "failure_reason": "/private/checkpoint.pkl secret-token" if key == "lstm-64" else None,
        }
        for key in candidate_keys
    }
    train_seeds = []
    for key in module.NEURAL:
        for seed, loss in zip((42, 43, 44), (0.65, 0.75, 0.85), strict=True):
            eligible = not (key == "lstm-64" and seed == 44)
            train_seeds.append(
                {
                    "candidate_key": key,
                    "seed": seed,
                    "eligible": eligible,
                    "validation": {"eligible": eligible, "reason": "/private/worker.log"},
                    "status": "complete" if eligible else "failed",
                    "elapsed_seconds": 12.5,
                    "epochs_completed": 5,
                    "optimizer_updates": 50,
                    "parameter_count": 1000,
                    "best_epoch": 3,
                    "best_tune_nll": loss,
                    "training_ids": ["raw-case-001"],
                    "probabilities": [[0.1, 0.2, 0.7]],
                    "private_path": "/private/model.pkl",
                }
            )
    curve = []
    for threshold, selected, precision in ((0.5, 150, 0.6), (0.9, 0, 0.5)):
        curve.append({**selective(selected, precision), "threshold": threshold, "qualifies": False})
    gate = {
        "enabled": False,
        "threshold": None,
        "requirements": dict(module.GATE_REQUIREMENTS),
        "curve": curve,
        "metrics": metrics(0.7, 0, uncertainty=False),
        "reason": "/private/gate-output.json",
    }
    selection = {
        "selected_candidate_id": "tcn-32",
        "cpu_reference_candidate_id": "lr-1",
        "temperature": 1.25,
        "comparison_scope": "selected_raw_primary_vs_raw_tune_selected_cpu",
        "selection_id": digest("selection"),
        "selected_models": [{"path": "/private/model.pkl"}],
        "training_ids": ["raw-case-001"],
    }
    primaries = {key: metrics(loss) for key, loss in test_losses.items()}
    paired = {
        "mean_control_minus_selected_log_loss": 0.1,
        "comparison_scope": "selected_raw_primary_vs_raw_tune_selected_cpu",
        "selected_candidate_id": "tcn-32",
        "reference_candidate_id": "lr-1",
    }
    for key, block in (("date_cluster", 1), ("five_date_block_sensitivity", 5)):
        paired[key] = {
            "low": -0.05,
            "high": 0.2,
            "resamples": 1000,
            "seed": 42,
            "sessions": 10,
            "clusters": 10 // block,
            "block_days": block,
        }
    return {
        "schema_version": "supervised-forecast-report-v1",
        "evidence_mode": "historical_project_holdout",
        "evaluation_kind": "synthetic_control_flow",
        "comparison_scope": "selected_raw_primary_vs_raw_tune_selected_cpu",
        **{
            key: digest(key)
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
        "source_provenance": {"private_path": "/private/code.py", "source_urls": ["https://private.test/source"]},
        "candidates": candidates,
        "training": {"cpu_arms": [], "neural_seeds": train_seeds},
        "selection": selection,
        "tune": {"candidate_primaries": {key: metrics(loss) for key, loss in tune_losses.items()}},
        "calibration": {"temperature": 1.25},
        "gate": gate,
        "test": {
            "candidate_primaries": primaries,
            "individual_seed_diagnostics": {
                f"{key}:{seed}": metrics(0.6)
                for key in module.NEURAL
                for seed in (42, 43, 44)
                if not (key == "lstm-64" and seed == 44)
            },
            "selected_raw": metrics(0.8, 0),
            "selected_calibrated": metrics(0.7, 0),
            "prior": primaries["prior"],
            "cpu_reference": primaries["lr-1"],
            "paired_reference_improvement": paired,
        },
        "flags": {
            "observed_selective_target_met": False,
            "improvement_supported": False,
            "automatic_production_promotion": False,
        },
        "split": {
            role: {"planned_sessions": 10, "eligible_cases": 300, "case_ids_hash": digest(role)}
            for role in module.ROLES
        },
        "catalog_counts": {
            role: [{"planned": 310, "eligible": 300, "excluded": 10, "private_path": "/private/catalog.jsonl"}]
            for role in module.ROLES
        },
        "exclusions": {role: {"sequence_missing_minute": 10, "secret-token": 1} for role in module.ROLES},
        "constraints": {
            "new_jev_calls": 0,
            "broker_orders": 0,
            "no_test_retuning": True,
            "default_model_changed": False,
        },
    }


def test_public_allowlist_privacy_failed_family_and_raw_comparison(report: dict[str, Any]) -> None:
    public = renderer().public_summary(report)
    encoded = json.dumps(public)
    for forbidden in (
        "/private/",
        "secret-token",
        "raw-case-001",
        "source_urls",
        "private_path",
        "probabilities",
        '"features":',
        '"targets":',
        '"prices":',
        "training_ids",
        "checkpoint.pkl",
        "selected_models",
    ):
        assert forbidden not in encoded
    assert set(public) == {
        "schema_version",
        "scope",
        "evaluation_kind",
        "interpretation",
        "metric_units",
        "source_attribution",
        "identities",
        "source_provenance_hash",
        "splits",
        "exclusions",
        "catalog_counts",
        "selection",
        "candidates",
        "individual_test_seed_diagnostics",
        "selected_raw",
        "selected_calibrated",
        "paired_raw_reference_improvement",
        "gate",
        "flags",
        "constraints",
        "public_id",
    }
    assert public["candidates"]["lstm-64"]["status"] == "FAILED"
    assert public["candidates"]["lstm-64"]["test"] is None
    assert public["candidates"]["tcn-32"]["primary_unit"] == "three-seed probability mean"
    assert len(public["candidates"]["tcn-32"]["seed_diagnostics"]) == 3
    assert public["selection"]["candidate"] == "tcn-32"  # HGB has better TEST NLL, never reselect.
    assert public["selection"]["cpu_reference"] == "lr-1"
    assert public["paired_raw_reference_improvement"]["mean_reference_minus_selected_log_loss"] == pytest.approx(0.1)
    assert public["selected_calibrated"]["log_loss"] == 0.7
    assert not public["flags"]["improvement_supported"]  # Raw interval includes zero despite calibration gain.
    assert public["candidates"]["tcn-32"]["test"]["accuracy"] == 0.5
    assert public["gate"]["metrics"]["coverage"] == 0
    assert public["gate"]["metrics"]["selective_accuracy"] is None
    assert public["gate"]["metrics"]["by_predicted_label"]["UP"]["precision"] is None
    assert not public["flags"]["automatic_production_promotion"]


@pytest.mark.parametrize(
    "mutation",
    [
        "reselect_test",
        "partial_family",
        "different_scope",
        "coverage_denominator",
        "fake_gate",
        "fake_promotion",
        "calibrated_architecture_gain",
        "percent_units",
        "missing_seed",
        "missing_predictions",
        "confusion_count",
        "fake_target",
    ],
)
def test_invalid_or_misleading_summary_rejected(report: dict[str, Any], mutation: str) -> None:
    if mutation == "reselect_test":
        report["selection"]["selected_candidate_id"] = "hgb-31"
    elif mutation == "partial_family":
        report["test"]["candidate_primaries"]["lstm-64"] = metrics(0.6)
    elif mutation == "different_scope":
        report["test"]["candidate_primaries"]["prior"]["eligible"] = 299
    elif mutation == "coverage_denominator":
        report["test"]["selected_raw"]["coverage"] = 0.1
    elif mutation == "fake_gate":
        report["gate"]["enabled"] = True
    elif mutation == "fake_promotion":
        report["flags"]["automatic_production_promotion"] = True
    elif mutation == "calibrated_architecture_gain":
        report["test"]["paired_reference_improvement"]["mean_control_minus_selected_log_loss"] = 0.2
    elif mutation == "percent_units":
        report["test"]["candidate_primaries"]["tcn-32"]["accuracy"] = 50
    elif mutation == "missing_seed":
        report["training"]["neural_seeds"].pop()
    elif mutation == "missing_predictions":
        report["test"]["candidate_primaries"]["lr-1"]["missing_predictions"] = 1
    elif mutation == "fake_target":
        report["test"]["selected_calibrated"]["target_requirements_met"] = True
    else:
        report["test"]["candidate_primaries"]["tcn-32"]["confusion_matrix"][0][0] = 49
    with pytest.raises(ValueError):
        renderer().public_summary(report)


def test_render_synthetic_pngs_only_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: Any, report: dict[str, Any]
) -> None:
    module = renderer()
    bundle = tmp_path / "sealed"
    bundle.mkdir()
    report_path = bundle / "report.json"
    report_path.write_text("{}")
    monkeypatch.setattr(module, "load_report", lambda path: copy.deepcopy(report))
    output = tmp_path / "public"
    result = module.render(report_path, output)
    assert result == output / "summary.json"
    assert sorted(p.name for p in output.iterdir()) == [
        "gate-coverage-direction-precision.png",
        "summary.json",
        "test-all-candidates.png",
        "tune-choice-seed-diagnostics.png",
    ]
    assert all(p.read_bytes().startswith(b"\x89PNG") for p in output.glob("*.png"))
    assert "secret-token" not in result.read_text()
    with pytest.raises(ValueError, match="fresh"):
        module.render(report_path, output)
    with pytest.raises(ValueError, match="outside"):
        module.render(report_path, bundle / "nested-public")


def test_integrity_loader_failure_prevents_output(tmp_path: Path, monkeypatch: Any) -> None:
    module = renderer()

    def invalid(_: Path) -> dict[str, Any]:
        raise ValueError("sealed inventory mismatch")

    monkeypatch.setattr(module, "load_report", invalid)
    output = tmp_path / "public"
    with pytest.raises(ValueError, match="sealed inventory"):
        module.render(tmp_path / "bundle" / "report.json", output)
    assert not output.exists()


def test_absolute_path_cli_and_all_abstain_validation() -> None:
    module = renderer()
    with pytest.raises(Exception, match="absolute"):
        module.absolute_path("report.json")
    row = selective(0)
    assert module._selective(row)["selective_accuracy"] is None
    row["selective_accuracy"] = 0
    with pytest.raises(ValueError, match="N/A"):
        module._selective(row)


def test_real_integrity_loader_on_synthetic_sealed_bundle(tmp_path: Path, report: dict[str, Any]) -> None:
    from tradecopilot.forecast.contracts import content_hash

    module = renderer()
    bundle = tmp_path / "sealed"
    bundle.mkdir()
    selection = copy.deepcopy(report["selection"])
    selection.update(registration_id=report["registration_id"], prepared_data_id=report["prepared_data_id"])
    del selection["selection_id"]
    selection["selection_id"] = content_hash(selection)
    prefit: dict[str, Any] = {"fixture": True}
    prefit["prefit_id"] = content_hash(prefit)
    report["prefit_id"] = prefit["prefit_id"]
    report["selection"] = selection
    (bundle / "frozen-selection.json").write_text(json.dumps(selection))
    (bundle / "prefit.json").write_text(json.dumps(prefit))
    for name in (
        "registration.json",
        "prepared-data.json",
        "source.json",
        "frozen-weights.json",
        "calibration.json",
        "gate.json",
        "predictions.jsonl",
        "examples.jsonl",
    ):
        (bundle / name).write_text('{"synthetic_private_material": "secret-token"}\n')
    inventory = {path.name: sha256(path.read_bytes()).hexdigest() for path in bundle.iterdir()}
    report.update(inventory=inventory, inventory_id=content_hash(inventory))
    del report["report_id"]
    report["report_id"] = content_hash(report)
    path = bundle / "report.json"
    path.write_text(json.dumps(report))
    assert module.load_report(path)["report_id"] == report["report_id"]
    (bundle / "predictions.jsonl").write_text("tampered synthetic fixture")
    output = tmp_path / "public"
    with pytest.raises(ValueError, match="integrity"):
        module.render(path, output)
    assert not output.exists()
