"""Synthetic fixtures only: sealed inputs, aggregate privacy and honest plot semantics."""

import copy
import importlib.util
import json
from hashlib import sha256
from pathlib import Path

import pytest

from tradecopilot.forecast.contracts import content_hash
from tradecopilot.rl import jev_ablation as ab
from tradecopilot.rl import training


def renderer():
    path = Path(__file__).resolve().parents[1] / "scripts" / "render_jev_rl_results.py"
    spec = importlib.util.spec_from_file_location("jev_rl_renderer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sealed(value, key):
    return {**value, key: content_hash(value)}


def metrics(net):
    return {
        "valid": True,
        "episodes": 25,
        "dates": 5,
        "invalid_episodes": 0,
        "mean_daily_return": net,
        "max_episode_drawdown": 0.001,
        "trade_count": 10,
        "fees": 0,
        "execution_drag": 2,
        "turnover": 2000,
        "daily_returns": {f"private-day-{i}": net for i in range(5)},
        "ledger": [{"price": 123}],
        "private_path": "/private/checkpoint/model.zip",
        "normalized_inputs": [1.2, 3.4],
        "credential": "fixture-secret",
    }


@pytest.fixture
def inputs(tmp_path):
    h = lambda s: sha256(s.encode()).hexdigest()  # noqa: E731
    reg = sealed(
        {
            "symbols": ["A", "B", "C", "D", "E"],
            "seeds": [42, 43, 44],
            "profiles": list(ab.PROFILES),
            "splits": {
                "train": ["2024-01-01", "2024-01-31"],
                "tune": ["2024-02-01", "2024-02-29"],
                "test": ["2024-03-01", "2024-03-31"],
            },
            "candidates": [training._CANDIDATES["recurrent-256"]],
            "optimizer": training.OPTIMIZER,
            "jev": {
                "model": "jev-1.13.0",
                "prompt_version": "masked-v1",
                "horizons": list(ab.HORIZONS),
                "as_of_minutes_after_open": 61,
                "simulated_latency_seconds": 60,
            },
            "environment": {
                "initial_cash": "10000",
                "notional_cap": "1000",
                "loss_lock": "100",
                "base_cost_bps_per_side": 2,
                "fee_per_fill": "0",
                "capacity_fraction": "0.01",
                "quantity_quantum": "0.000001",
                "warmup_minutes": 60,
                "latency_minutes": 1,
                "forced_close_buffer_minutes": 5,
            },
            "rule": {"enter_when_return_5m_bps_above": 5},
        },
        "registration_id",
    )
    manifest = sealed(
        {"registration_id": reg["registration_id"], "source_data_id": h("source"), "private_normalizer": [1, 2, 3]},
        "data_id",
    )
    cache = {
        "cache_id": h("cache"),
        "usage": {"attempts": 80, "input_tokens": 12000, "conservative_estimated_cost_usd": 0.024},
        "records": [{"prediction": "PRIVATE", "anchor_price": "123"}],
    }
    budget = sealed(
        {
            "registration_id": reg["registration_id"],
            "base_prepared_data_id": manifest["data_id"],
            "cache_id": cache["cache_id"],
            "shared_steps": 10240,
            "max_seconds": 600,
            "device": "cpu",
        },
        "budget_id",
    )
    provenance = {"code": h("code")}
    execution = {"python": "/private/python", "scripts": {"run_jev_rl_ablation.py": h("script")}}
    seal = sealed(
        {
            "registration": reg,
            "budget": budget,
            "manifest": manifest,
            "cache_id": cache["cache_id"],
            "source_provenance": provenance,
            "lock_sha256": h("lock"),
            "execution_provenance": execution,
        },
        "ablation_id",
    )
    report = {
        "schema_version": "jev-rl-ablation-report-v1",
        "status": "inconclusive_not_promoted",
        "registration_id": reg["registration_id"],
        "base_prepared_data_id": manifest["data_id"],
        "cache_id": cache["cache_id"],
        "budget_id": budget["budget_id"],
        "training": [],
        "source_provenance": provenance,
        "lock_sha256": h("lock"),
        "execution_provenance": execution,
    }
    (tmp_path / "ablation-seal.json").write_text(json.dumps(seal))
    for profile in ab.PROFILES:
        for seed in (42, 43, 44):
            directory = tmp_path / f"{profile}-{seed}"
            directory.mkdir()
            model = directory / "model.zip"
            model.write_bytes(f"synthetic checkpoint {profile} {seed}".encode())
            run = sealed(
                {
                    "profile": profile,
                    "seed": seed,
                    "candidate": training._CANDIDATES["recurrent-256"],
                    "status": "complete",
                    "actual_timesteps": 10240,
                    "n_updates": 800,
                    "elapsed_seconds": 10.0,
                    "n_parameters": 100,
                    "observation_dim": 151,
                    "registration_id": reg["registration_id"],
                    "base_prepared_data_id": manifest["data_id"],
                    "cache_id": cache["cache_id"],
                    "budget_id": budget["budget_id"],
                    "config_hash": training.registered_config(reg).content_hash,
                    "source_provenance": provenance,
                    "lock_sha256": h("lock"),
                    "execution_provenance": execution,
                    "versions": {"torch": "synthetic"},
                    "checkpoint_sha256": sha256(model.read_bytes()).hexdigest(),
                },
                "training_id",
            )
            (directory / "result.json").write_text(json.dumps(run))
            report["training"].append(run)
    report["tune_seeds"] = {p: {str(s): metrics((45 - s) * 0.001) for s in (42, 43, 44)} for p in ab.PROFILES}
    report["test_seeds"] = {
        "CONTEXT_ONLY": {str(s): metrics(0.001) for s in (42, 43, 44)},
        "JEV_ASSISTED": {str(s): metrics(v) for s, v in zip((42, 43, 44), (-0.003, 0.003, 0.006), strict=True)},
    }
    report["test_mean_seed"] = {p: ab.mean_seed_metrics(report["test_seeds"][p]) for p in ab.PROFILES}
    report["test_controls"] = {c: metrics(0) for c in ("cash", "hold", "rule")}
    report["cost_stress"] = {}
    for cost in ("4", "8"):
        profiles = copy.deepcopy(report["test_seeds"])
        report["cost_stress"][cost] = {
            "profiles": profiles,
            "mean_seed": {p: ab.mean_seed_metrics(profiles[p]) for p in ab.PROFILES},
            "controls": copy.deepcopy(report["test_controls"]),
        }
    chosen = {
        p: {
            k: next(r for r in report["training"] if r["profile"] == p and r["seed"] == 42)[k]
            for k in ("seed", "training_id", "checkpoint_sha256")
        }
        for p in ab.PROFILES
    }
    report["selection"] = sealed({"profiles": chosen, "stage": "selection_frozen"}, "selection_id")
    report["improvement_gate"] = {"passed": False, "checks": {key: False for key in renderer().CHECKS}}

    def availability(cohort, available):
        return {
            "cohort_records": cohort,
            "available_records": available,
            "unavailable_records": cohort - available,
            "error_rate": (cohort - available) / cohort,
        }

    def grade(available=8, accuracy=0.5, coverage=0.2):
        return {
            **availability(10, available),
            "records": 10,
            "accuracy": accuracy,
            "accuracy_available": accuracy,
            "accuracy_errors_as_incorrect": accuracy * available / 10 if accuracy is not None else 0,
            "accuracy_scope": "conditional_on_available_records",
            "log_loss": 1.1 if available else None,
            "Brier": 0.7 if available else None,
            "loss_scope": "available_records_only",
            "coverage_0_6": coverage,
            "coverage_scope": "full_cohort",
            "selected_accuracy_0_6": 0.5 if coverage else None,
            "prediction": "PRIVATE",
        }

    full_prior = grade(10, 0.6, 0.3)
    report["forecast_grading"] = {
        "vendor_pretraining_cutoff": "unknown",
        "availability": availability(210, 168),
        "horizons": {
            horizon: {
                "train_labels": 50,
                "train_prior": [0.3, 0.4, 0.3],
                "train_availability": availability(50, 40),
                **{
                    role: {
                        "jev": grade(),
                        "prior": copy.deepcopy(full_prior),
                        "prior_full_cohort": copy.deepcopy(full_prior),
                        "prior_same_available": grade(8, 0.625, 0.3),
                    }
                    for role in ("tune", "test")
                },
            }
            for horizon in ab.HORIZONS
        },
    }
    cache["acquisition_amendment"] = sealed(
        {
            "existing_attempted_batch_indices": list(range(6)),
            "remaining_unattempted_batch_indices": list(range(6, 80)),
            "retry_provider_failures": False,
            "failure_policy": "preserve_unavailable_and_continue_unattempted",
            "api_budget": {"max_attempts": 80, "max_conservative_cost_usd": 0.25, "cooldown_seconds": 30},
            "private_path": "/private/amendment.json",
            "private_notes": "fixture-secret",
        },
        "amendment_id",
    )
    cache["amendment_id"] = cache["acquisition_amendment"]["amendment_id"]
    cache["batch_counts"] = {"valid": 78, "unavailable": 2}
    report["inventory"] = {
        str(p.relative_to(tmp_path)): sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()
    }
    report = sealed(report, "report_id")
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    return report, seal, cache, report_path


def test_public_allowlist_units_and_all_seed_primary(inputs):
    report, seal, cache, _ = inputs
    public = renderer().public_summary(report, seal, cache)
    encoded = json.dumps(public)
    for forbidden in (
        "private-day",
        "daily_returns",
        "/private/",
        "model.zip",
        "fixture-secret",
        "ledger",
        "normalized_inputs",
        "anchor_price",
        "prediction",
        "PRIVATE",
        "train_prior",
    ):
        assert forbidden not in encoded
    assert set(public) == {
        "schema_version",
        "status",
        "scope",
        "vendor_pretraining_cutoff",
        "cache_availability_mode",
        "metric",
        "interpretation",
        "source_attribution",
        "identities",
        "selection_id",
        "splits",
        "symbols",
        "profile_contract",
        "tune",
        "primary_test",
        "test_controls",
        "paired_intervals_bps",
        "selected_tune_diagnostics",
        "cost_stress_bps_per_side",
        "forecast_grading",
        "acquisition",
        "forecast_metric_units",
        "forecast_coverage_definition",
        "promotion",
        "training",
        "api_usage",
        "public_id",
    }
    assert public["primary_test"]["JEV_ASSISTED"]["mean_daily_return_bps"] == pytest.approx(20)
    assert public["paired_intervals_bps"]["jev_minus_context"]["mean"] == pytest.approx(10)
    assert public["paired_intervals_bps"]["jev_minus_cash"]["mean"] == pytest.approx(20)
    assert public["primary_test"]["JEV_ASSISTED"]["seeds"] == [42, 43, 44]
    assert public["selected_tune_diagnostics"]["JEV_ASSISTED"]["seed"] == 42
    assert "not primary" in public["selected_tune_diagnostics"]["JEV_ASSISTED"]["role"]
    assert public["forecast_grading"]["horizons"]["15m"]["test"]["jev"]["selected_records_0_6"] == 2
    assert public["status"] == "inconclusive_not_promoted"
    assert public["api_usage"]["attempts"] == 80
    assert public["cost_stress_bps_per_side"]["8"]["JEV_ASSISTED"]["mean_daily_return_bps"] == pytest.approx(20)


@pytest.mark.parametrize("mutation", ["failed", "missing_seed", "fake_primary", "bad_accuracy", "status", "cache"])
def test_incomplete_or_misleading_summary_rejected(inputs, mutation):
    report, seal, cache, _ = inputs
    if mutation == "failed":
        report["status"] = "failed"
    elif mutation == "missing_seed":
        del report["test_seeds"]["JEV_ASSISTED"]["44"]
    elif mutation == "fake_primary":
        report["test_mean_seed"]["JEV_ASSISTED"]["mean_daily_return"] = 0.006
    elif mutation == "bad_accuracy":
        report["forecast_grading"]["horizons"]["15m"]["test"]["jev"]["accuracy"] = 50
    elif mutation == "status":
        report["status"] = "improved_among_tested"
    else:
        cache["cache_id"] = "wrong"
    with pytest.raises(ValueError):
        renderer().public_summary(report, seal, cache)


def test_validated_renderer_png_and_no_private_output(tmp_path, monkeypatch, inputs):
    _, _, cache, report_path = inputs
    module = renderer()
    monkeypatch.setattr(module, "load_cache", lambda _: cache)
    output = tmp_path / "public"
    assert module.render(report_path, output, tmp_path / "cache.json") == output / "summary.json"
    assert sorted(p.name for p in output.iterdir()) == [
        "cost-stress-means.png",
        "forecast-accuracy-coverage.png",
        "summary.json",
        "test-means-controls.png",
        "tune-paired-seeds.png",
    ]
    assert all(p.read_bytes().startswith(b"\x89PNG") for p in output.glob("*.png"))
    assert "private-day" not in (output / "summary.json").read_text()
    with pytest.raises(ValueError, match="fresh"):
        module.render(report_path, output, tmp_path / "cache.json")


@pytest.mark.parametrize("mutation", ["artifact", "report", "extra_inventory", "incomplete_inventory"])
def test_full_seal_and_inventory_validation_before_output(tmp_path, monkeypatch, inputs, mutation):
    report, _, cache, report_path = inputs
    module = renderer()
    monkeypatch.setattr(module, "load_cache", lambda _: cache)
    if mutation == "artifact":
        (tmp_path / "CONTEXT_ONLY-42" / "model.zip").write_bytes(b"tampered")
    elif mutation == "report":
        report["status"] = "failed"
        report_path.write_text(json.dumps(report))
    elif mutation == "extra_inventory":
        (tmp_path / "unsealed-private-file").write_text("secret")
    else:
        del report["inventory"]["CONTEXT_ONLY-42/model.zip"]
        del report["report_id"]
        report_path.write_text(json.dumps(sealed(report, "report_id")))
    output = tmp_path / "public"
    with pytest.raises(ValueError):
        module.render(report_path, output, tmp_path / "cache.json")
    assert not output.exists()


def test_cli_absolute_paths_and_render_enforces_paths(tmp_path):
    module = renderer()
    assert module.absolute_path(str(tmp_path)) == tmp_path
    with pytest.raises(Exception, match="absolute"):
        module.absolute_path("relative.json")
    with pytest.raises(ValueError, match="absolute"):
        module.render(Path("report.json"), tmp_path / "new", tmp_path / "cache.json")


def test_cache_binding_mismatch_rejected_before_creating_output(tmp_path, monkeypatch, inputs):
    _, _, cache, report_path = inputs
    module = renderer()
    cache["cache_id"] = "wrong"
    monkeypatch.setattr(module, "load_cache", lambda _: cache)
    output = tmp_path / "public"
    with pytest.raises(ValueError, match="cache/report identity"):
        module.render(report_path, output, tmp_path / "cache.json")
    assert not output.exists()


def test_selected_checkpoint_diagnostic_must_bind_to_training(inputs):
    report, seal, cache, _ = inputs
    report["selection"]["profiles"]["JEV_ASSISTED"]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint/training identity"):
        renderer().public_summary(report, seal, cache)


def test_forecast_failures_keep_full_cohort_and_matched_prior_denominators(inputs):
    report, seal, cache, _ = inputs
    public = renderer().public_summary(report, seal, cache)
    forecast = public["forecast_grading"]
    assert forecast["availability"]["unavailable_records"] == 42
    assert forecast["availability"]["error_rate"] == pytest.approx(0.2)
    row = forecast["horizons"]["15m"]["test"]
    assert row["jev"]["accuracy_available"] == 0.5
    assert row["jev"]["accuracy_errors_as_incorrect"] == 0.4
    assert row["jev"]["cohort_records"] == 10
    assert row["jev"]["available_records"] == 8
    assert row["jev"]["unavailable_records"] == 2
    assert row["prior_same_available"]["available_records"] == row["jev"]["available_records"]
    assert row["prior_full_cohort"]["available_records"] == 10
    assert row["jev"]["coverage_0_6"] == 0.2
    assert row["jev"]["loss_scope"] == "available_records_only"
    amendment = public["acquisition"]
    assert amendment["initial_attempted_batches"] == 6
    assert amendment["continuation_planned_batches"] == 74
    assert amendment["batch_counts"] == {"valid": 78, "unavailable": 2}
    assert amendment["original_abort_rule_overridden"]
    assert amendment["resume_unattempted_only"]
    assert not amendment["retry_provider_failures"]
    assert amendment["same_original_caps"]


@pytest.mark.parametrize(
    "mutation", ["conditional_as_full", "unmatched_prior", "available_coverage", "lost_failure", "amendment_retry"]
)
def test_misleading_failure_denominators_and_retry_policy_rejected(inputs, mutation):
    report, seal, cache, _ = inputs
    row = report["forecast_grading"]["horizons"]["15m"]["test"]
    if mutation == "conditional_as_full":
        row["jev"]["accuracy_errors_as_incorrect"] = row["jev"]["accuracy_available"]
    elif mutation == "unmatched_prior":
        row["prior_same_available"] = copy.deepcopy(row["prior_full_cohort"])
    elif mutation == "available_coverage":
        row["jev"]["coverage_scope"] = "available_records_only"
    elif mutation == "lost_failure":
        report["forecast_grading"]["availability"]["unavailable_records"] = 0
    else:
        cache["acquisition_amendment"]["retry_provider_failures"] = True
    with pytest.raises(ValueError):
        renderer().public_summary(report, seal, cache)


def test_all_unavailable_forecasts_have_no_conditional_quality():
    row = {
        "cohort_records": 10,
        "available_records": 0,
        "unavailable_records": 10,
        "error_rate": 1,
        "records": 10,
        "accuracy": None,
        "accuracy_available": None,
        "accuracy_errors_as_incorrect": 0,
        "log_loss": None,
        "Brier": None,
        "loss_scope": "available_records_only",
        "coverage_0_6": 0,
        "coverage_scope": "full_cohort",
        "selected_accuracy_0_6": None,
    }
    public = renderer()._forecast_metrics(row)
    assert public["accuracy_available"] is None
    assert public["accuracy_errors_as_incorrect"] == 0
    assert public["selected_records_0_6"] == 0
