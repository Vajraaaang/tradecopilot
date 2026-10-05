"""Synthetic forecast-correction contracts; no provider or market-quality runs."""

import copy
import importlib.util
import json
from dataclasses import fields, replace
from datetime import date, timedelta
from decimal import Decimal
from hashlib import sha256

import numpy as np
import pytest

from tradecopilot.forecast.bars import HistoricalBar, write_bar_dataset
from tradecopilot.forecast.contracts import LABELS, content_hash
from tradecopilot.forecast.jev_rl import _BUDGET
from tradecopilot.forecast.ohlcv_features import OHLCV_FEATURE_NAMES, OHLCV_FEATURE_VERSION
from tradecopilot.forecast.sessions import session_bounds
from tradecopilot.rl.training import _CANDIDATES, OPTIMIZER


def sealed(value, key):
    return {**value, key: content_hash(value)}


def test_fusion_api_exists():
    assert importlib.util.find_spec("tradecopilot.forecast.jev_fusion") is not None


@pytest.fixture
def bundle(tmp_path):
    symbols = ["AAPL", "AMZN", "MSFT", "NVDA", "TSLA"]
    reg = sealed(
        {
            "fixture": True,
            "forecast_profile": "MARKET_ONLY",
            "symbols": symbols,
            "source_range": {
                "start": "2025-11-03",
                "end_exclusive": "2025-11-07",
                "provider": "Alpaca",
                "feed": "sip",
                "adjustment": "raw",
            },
            "splits": {"train": ["2025-11-04"] * 2, "tune": ["2025-11-05"] * 2, "test": ["2025-11-06"] * 2},
            "profiles": ["CONTEXT_ONLY", "JEV_ASSISTED"],
            "seeds": [42, 43, 44],
            "candidates": [_CANDIDATES["recurrent-256"]],
            "optimizer": OPTIMIZER,
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
            "jev": {
                "model": "jev-1.13.0",
                "prompt_version": "masked-normalized-ohlcv-3h-v1",
                "as_of_minutes_after_open": 61,
                "simulated_latency_seconds": 60,
                "horizons": ["15m", "60m", "close"],
                "flat_threshold_bps": 10,
                "abstention_threshold": 0.6,
            },
            "api_budget": _BUDGET,
        },
        "registration_id",
    )
    bars = []
    episodes = []
    records = []
    for day_number in range(3, 7):
        day = date(2025, 11, day_number)
        opening, closing = session_bounds(day)
        for ordinal, symbol in enumerate(symbols):
            for minute in range(80):
                if day_number == 4 and ordinal == 0 and minute == 30:
                    continue
                start = opening + timedelta(minutes=minute)
                price = Decimal(100)
                if minute >= 61:
                    price += [Decimal("-.2"), Decimal(0), Decimal(".2"), Decimal(".1"), Decimal("-.1")][ordinal]
                bars.append(
                    HistoricalBar(
                        symbol=symbol,
                        start_time=start,
                        end_time=start + timedelta(minutes=1),
                        available_at=start + timedelta(minutes=1),
                        opening=price,
                        high=price,
                        low=price,
                        close=price,
                        volume=Decimal(100 + minute),
                        source="alpaca_sip_1min_bar",
                    )
                )
            if day_number == 3:
                continue
            role = {4: "train", 5: "tune", 6: "test"}[day_number]
            episodes.append(
                {
                    "symbol": symbol,
                    "session_date": day.isoformat(),
                    "role": role,
                    "session_open": int(opening.timestamp()),
                    "session_close": int(closing.timestamp()),
                }
            )
            as_of = int(opening.timestamp()) + 61 * 60
            status = "unavailable" if ordinal == 4 else "abstained" if ordinal == 3 else "ok"
            for horizon in ("15m", "60m", "close"):
                probabilities = dict(
                    zip(LABELS, [0.0, 0.0, 0.0] if status == "unavailable" else [0.1, 0.2, 0.7], strict=True)
                )
                records.append(
                    {
                        "symbol": symbol,
                        "session_date": day.isoformat(),
                        "horizon_key": horizon,
                        "as_of": as_of,
                        "target_time": int(closing.timestamp())
                        if horizon == "close"
                        else as_of + int(horizon[:-1]) * 60,
                        "anchor_price": "100",
                        "replay_available_at": as_of + 60,
                        "generated_at": "2026-10-04T00:00:00+00:00",
                        "probabilities": probabilities,
                        "model_confidence": 0.0 if status == "unavailable" else 0.5 if status == "abstained" else 0.8,
                        "status": status,
                        "request_id": str(day_number),
                        "payload_id": f"payload-{day_number}",
                        "input_id": f"input-{day_number}-{symbol}-{horizon}",
                    }
                )
    metadata = {
        "fixture": True,
        "provider": "Alpaca",
        "feed": "sip",
        "adjustment_policy": "raw",
        "start_date": "2025-11-03",
        "end_date_exclusive": "2025-11-07",
        "selected_symbols": symbols,
    }
    source_dir = tmp_path / "source-bars"
    source_path = write_bar_dataset(source_dir, bars, metadata)
    source = json.loads(source_path.read_text())
    manifest = sealed(
        {
            "schema_version": "rl-prepared-market-v1",
            "registration": reg,
            "registration_id": reg["registration_id"],
            "source_manifest": source,
            "source_metadata": metadata,
            "source_data_id": source["data_id"],
            "config_id": content_hash(
                {"registration_id": reg["registration_id"], "feature_version": OHLCV_FEATURE_VERSION}
            ),
            "feature_names": [
                *OHLCV_FEATURE_NAMES,
                *(f"missing_{n}" for n in OHLCV_FEATURE_NAMES),
                *(f"symbol_{s}" for s in symbols),
            ],
            "episodes": episodes,
        },
        "data_id",
    )
    cache = sealed(
        {
            "schema_version": "jev-rl-cache-v1",
            "registration_id": reg["registration_id"],
            "source_data_id": source["data_id"],
            "base_prepared_data_id": manifest["data_id"],
            "model": reg["jev"]["model"],
            "prompt_version": reg["jev"]["prompt_version"],
            "availability_mode": "hypothetical_as_of_plus_60_seconds",
            "records": records,
            "inventory": [],
        },
        "cache_id",
    )
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    (prepared_dir / "manifest.json").write_text(json.dumps(manifest))
    registration_path, cache_path = tmp_path / "registration.json", tmp_path / "cache.json"
    registration_path.write_text(json.dumps(reg))
    cache_path.write_text(json.dumps(cache))
    return bars, source, manifest, reg, cache, source_dir, prepared_dir, registration_path, cache_path


def cases_from(bundle):
    from tradecopilot.forecast.jev_fusion import build_cases

    return build_cases(*bundle[:5])


def test_label_free_complete_ordered_cases_preserve_missingness_and_abstention(bundle):
    from tradecopilot.forecast.jev_fusion import FusionCase

    cases = cases_from(bundle)
    assert len(cases) == 15
    assert [(c.session_date, c.symbol) for c in cases] == [
        (date(2025, 11, day), s) for day in (4, 5, 6) for s in bundle[3]["symbols"]
    ]
    assert not {"label", "target_close", "return_bps"} & {f.name for f in fields(FusionCase)}
    assert cases[0].context[OHLCV_FEATURE_NAMES.index("return_60m_bps")] is None
    assert sum(c.available for c in cases) == 12
    assert cases[3].available and cases[3].status == "abstained"
    assert cases[4].jev_probabilities == (0.0, 0.0, 0.0) and cases[4].model_confidence == 0


def test_exact_target_decimal_boundaries_and_missing_target_refusal(bundle):
    from tradecopilot.forecast.jev_fusion import labels_for_cases

    cases = cases_from(bundle)
    np.testing.assert_array_equal(labels_for_cases(bundle[0], cases[:5]), [0, 1, 2, 1, 1])
    missing = [
        b
        for b in bundle[0]
        if not (b.symbol == cases[0].symbol and int(b.end_time.timestamp()) == cases[0].target_time)
    ]
    with pytest.raises(ValueError, match="exact target"):
        labels_for_cases(missing, cases)


def test_future_outcome_mutation_cannot_enter_context(bundle):
    from tradecopilot.forecast.jev_fusion import build_cases, labels_for_cases

    cases = cases_from(bundle)
    first = cases[0]
    changed = [
        b.model_copy(
            update={
                "opening": Decimal(999),
                "high": Decimal(999),
                "low": Decimal(999),
                "close": Decimal(999),
                "volume": Decimal(999),
            }
        )
        if b.symbol == first.symbol and int(b.end_time.timestamp()) == first.target_time
        else b
        for b in bundle[0]
    ]
    revised = build_cases(changed, *bundle[1:5])
    assert revised[:5] == cases[:5]
    assert labels_for_cases(changed, [first])[0] != labels_for_cases(bundle[0], [first])[0]


@pytest.mark.parametrize("mutation", ["anchor", "missing_slot", "duplicate_slot", "late_anchor", "target"])
def test_cache_source_slot_and_timing_mismatch_refused(bundle, mutation):
    from tradecopilot.forecast.jev_fusion import build_cases

    bars, source, manifest, reg, cache = copy.deepcopy(bundle[:5])
    if mutation == "anchor":
        cache["records"][0]["anchor_price"] = "101"
    elif mutation == "missing_slot":
        cache["records"].pop()
    elif mutation == "duplicate_slot":
        cache["records"].append(cache["records"][0])
    elif mutation == "target":
        cache["records"][0]["target_time"] += 60
    else:
        key = cache["records"][0]
        bars = [
            b.model_copy(update={"available_at": b.available_at + timedelta(minutes=1)})
            if b.symbol == key["symbol"] and int(b.end_time.timestamp()) == key["as_of"]
            else b
            for b in bars
        ]
    cache.pop("cache_id")
    cache = sealed(cache, "cache_id")
    with pytest.raises(ValueError):
        build_cases(bars, source, manifest, reg, cache)


def test_train_only_medians_scaler_shared_context_zero_missing_jev_and_json_parity(bundle):
    from tradecopilot.forecast.jev_fusion import ARMS, _design_matrices, fit_models, labels_for_cases, predict_models

    cases = cases_from(bundle)
    model = fit_models(cases[:5], labels_for_cases(bundle[0], cases[:5]), tuple(bundle[3]["symbols"]))
    before = json.dumps(model, allow_nan=False)
    context, fusion = _design_matrices(cases, model["normalizer"], tuple(bundle[3]["symbols"]))
    np.testing.assert_array_equal(context, fusion[:, : context.shape[1]])
    assert context.shape == (15, 121) and fusion.shape == (15, 125)
    np.testing.assert_array_equal(fusion[4, -4:], [0.0, 0.0, 0.0, 0.0])
    assert context[0, 55 + OHLCV_FEATURE_NAMES.index("return_60m_bps")] == 1
    assert np.isfinite(fusion).all()
    predictions = predict_models(json.loads(before), cases)
    assert tuple(predictions) == ARMS
    for p in predictions.values():
        assert p.shape == (15, 3) and np.isfinite(p).all()
        np.testing.assert_allclose(p.sum(axis=1), 1)
    np.testing.assert_array_equal(predictions["RAW_JEV"][4], model["class_prior"])
    assert model["normalizer"]["fit_case_ids"] == [c.case_id for c in cases[:5]]
    changed = [*cases[:5], *(replace(c, context=(None,) * 55) for c in cases[5:])]
    predict_models(model, changed)
    assert json.dumps(model, allow_nan=False) == before
    with pytest.raises(ValueError, match="TRAIN"):
        fit_models(cases, labels_for_cases(bundle[0], cases), tuple(bundle[3]["symbols"]))


def test_consumed_exploration_runner_freezes_models_before_outcomes_and_inventories_every_file(
    bundle, tmp_path, monkeypatch
):
    from tradecopilot.forecast import jev_fusion as module
    from tradecopilot.forecast import jev_rl

    monkeypatch.setattr(module, "load_cache", lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(jev_rl, "_post", lambda *args: pytest.fail("provider access forbidden"))
    output = tmp_path / "fusion"
    original_labels = module.labels_for_cases
    visits = []

    def checked_labels(bars, cases):
        visits.append([c.split for c in cases])
        if any(c.split != "train" for c in cases):
            assert (output / "models.json").is_file()
        return original_labels(bars, cases)

    monkeypatch.setattr(module, "labels_for_cases", checked_labels)
    path = module.run_jev_fusion(*bundle[5:], output)
    assert path == output / "report.json"
    report = json.loads(path.read_text())
    assert report["evidence_mode"] == "consumed_date_exploration"
    assert report["confirmatory"] is False and report["fresh_model_selection_allowed"] is False
    assert report["case_count"] == 15 and report["split_counts"] == {"train": 5, "tune": 5, "test": 5}
    assert visits[0] == ["train"] * 5 and len(visits) == 2
    for arm in module.ARMS:
        assert report["metrics"][arm]["test"]["all_cases"]["scored"] == 5
        assert report["metrics"][arm]["test"]["all_cases"]["coverage"] == 1
        assert report["metrics"][arm]["test"]["matched_available"]["scored"] == 4
    assert report["paired_test_nll"]["resamples"] == 1000
    assert report["paired_test_nll"]["seed"] == 42
    assert report["paired_test_nll"]["interpretation"] == "descriptive_consumed_dates_only"
    records = json.loads((output / "predictions.json").read_text())["records"]
    assert len(records) == 15 and all(set(r["probabilities"]) == set(module.ARMS) for r in records)
    inventory = json.loads((output / "inventory.json").read_text())["artifacts"]
    assert {i["path"] for i in inventory} == {str(p) for p in output.iterdir() if p.name != "inventory.json"}
    for item in inventory:
        from pathlib import Path

        assert sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]
    assert report["inference_parity"]["max_abs_error"] < 1e-12
    with pytest.raises(ValueError, match="immutable"):
        module.run_jev_fusion(*bundle[5:], output)


def test_metrics_class_coverage_and_paired_date_bootstrap(bundle):
    from tradecopilot.forecast.jev_fusion import paired_test_nll, score_probabilities

    cases = cases_from(bundle)[10:]
    labels = np.array([0, 1, 2, 1, 1], dtype=np.int64)
    probabilities = np.eye(3)[labels]
    metrics = score_probabilities(probabilities, labels, np.array([c.available for c in cases]))
    assert metrics["accuracy"] == metrics["balanced_accuracy"] == metrics["macro_f1"] == 1
    assert metrics["brier_score"] == metrics["log_loss"] == 0
    assert metrics["coverage"] == 1 and metrics["jev_availability_coverage"] == 0.8
    paired = paired_test_nll(cases, labels, probabilities, probabilities)
    assert paired["mean_difference"] == paired["low"] == paired["high"] == 0
    assert paired["dates"] == 1 and paired["cases"] == 5


def test_model_artifact_seal_and_finite_weights_are_checked(bundle):
    from tradecopilot.forecast.jev_fusion import fit_models, labels_for_cases, predict_models

    cases = cases_from(bundle)
    model = fit_models(cases[:5], labels_for_cases(bundle[0], cases[:5]), tuple(bundle[3]["symbols"]))
    model["models"]["CONTEXT_LR"]["intercept"][0] += 10
    with pytest.raises(ValueError, match="model_id"):
        predict_models(model, cases)


@pytest.mark.parametrize("where", ["normalizer", "CONTEXT_LR", "JEV_FUSION_LR"])
def test_resealed_wrong_feature_order_is_refused(bundle, where):
    from tradecopilot.forecast.jev_fusion import fit_models, labels_for_cases, predict_models

    cases = cases_from(bundle)
    model = fit_models(cases[:5], labels_for_cases(bundle[0], cases[:5]), tuple(bundle[3]["symbols"]))
    block = model["normalizer"] if where == "normalizer" else model["models"][where]
    block["feature_names"][0] = "unregistered_feature"
    model.pop("model_id")
    model = sealed(model, "model_id")
    with pytest.raises(ValueError, match="feature order"):
        predict_models(model, cases)
