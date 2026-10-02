from __future__ import annotations

import copy
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from threading import Barrier

import pytest

from tradecopilot.forecast.contracts import FEATURE_NAMES, ForecastConfig, ForecastExample
from tradecopilot.jev import INPUT_TOKEN_PRICE_USD, JEV_MODEL, MAX_INPUT_TOKENS, JevAdvisor

NOW = datetime(2026, 10, 1, 15, 0, tzinfo=UTC)
WORST_COST = MAX_INPUT_TOKENS * INPUT_TOKEN_PRICE_USD


def example(config=None, *, as_of=NOW, symbol="AAPL", labeled=False, **updates):
    config = config or ForecastConfig()
    features = dict.fromkeys(FEATURE_NAMES, 0.0)
    features.update(return_1m_bps=5, return_5m_bps=12, history_points=6)
    values = dict(
        config_id=config.config_id,
        symbol=symbol,
        as_of=as_of,
        target_time=as_of + timedelta(minutes=config.horizon_minutes),
        session_date=as_of.date(),
        anchor_price=Decimal("100"),
        features=features,
        observation_ids=("observation",),
        provenance="market",
    )
    if labeled:
        values.update(
            label="DOWN",
            target_price=Decimal("91.12345"),
            label_observed_at=values["target_time"],
            target_return_bps=-887.655,
        )
    return ForecastExample(**(values | updates))


def response(*, confidence=0.8, probabilities=None, choice="UP", tokens=1500):
    return {
        "model": JEV_MODEL,
        "answers": {
            "direction": {
                "type": "choice",
                "choice": choice,
                "confidence": confidence,
                "probabilities": probabilities or {"DOWN": 0.1, "FLAT": 0.2, "UP": 0.7},
            },
        },
        "usage": {"input_tokens": tokens},
    }


def service(tmp_path, *, reply=None, clock=None, **kwargs):
    from tradecopilot.forecast.jev import JevForecaster

    calls = []

    def transport(payload, key):
        calls.append((copy.deepcopy(payload), key))
        if isinstance(reply, Exception):
            raise reply
        return response() if reply is None else reply

    return JevForecaster(
        "TEST_SECRET",
        tmp_path / "jev.sqlite3",
        transport=transport,
        clock=clock or (lambda: NOW),
        **kwargs,
    ), calls


def test_adapter_available():
    import importlib.util

    assert importlib.util.find_spec("tradecopilot.forecast.jev") is not None


def test_prospective_freshness_includes_time_since_input_anchor(tmp_path):
    row = example()
    row = row.model_copy(update={"features": row.features | {"source_age_seconds": 29}})
    advisor, calls = service(tmp_path, clock=lambda: NOW + timedelta(seconds=29))
    result = advisor.predict(row, ForecastConfig(), "dataset")
    assert result.status == "error" and result.reason == "stale_source"
    assert result.estimated_cost_usd == 0 and calls == []


def test_historical_inputs_cannot_be_labeled_prospective(tmp_path):
    row = example().model_copy(update={"provenance": "historical"})
    advisor, calls = service(tmp_path)
    result = advisor.predict(row, ForecastConfig(), "historical-dataset")
    assert result.reason == "historical_input_requires_retrospective_mode"
    assert result.estimated_cost_usd == 0 and calls == []


@pytest.mark.parametrize("wait_seconds", [1, 2])
def test_target_expiry_during_reservation_wait_never_spends_credit(tmp_path, monkeypatch, wait_seconds):
    now = [NOW + timedelta(seconds=59)]
    config = ForecastConfig(horizon_minutes=1, max_source_age_seconds=120)
    advisor, calls = service(tmp_path, clock=lambda: now[0])
    original = advisor._reserve_forecast

    def delayed(*args, **kwargs):
        now[0] += timedelta(seconds=wait_seconds)
        return original(*args, **kwargs)

    monkeypatch.setattr(advisor, "_reserve_forecast", delayed)
    result = advisor.predict(example(config), config, "dataset")
    assert result.reason == "target_already_elapsed"
    assert result.estimated_cost_usd == 0 and result.request_id is None and calls == []
    with sqlite3.connect(tmp_path / "jev.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM jev_attempts").fetchone()[0] == 0


def test_prompt_uses_only_causal_context_and_exact_last_price_outcome(tmp_path):
    from tradecopilot.forecast.jev import FORECAST_PROMPT_VERSION

    config = ForecastConfig(horizon_minutes=7, flat_threshold_bps=12)
    row = example(config, labeled=True)
    advisor, calls = service(tmp_path)
    before = row.model_dump()
    result = advisor.predict(row, config, "dataset", prospective=False)
    assert result.status == "ok"
    assert result.model_id == result.model_version == JEV_MODEL
    assert result.prompt_version == FORECAST_PROMPT_VERSION
    assert result.execution == "live_api" and result.reason == "retrospective_api_inference"
    assert result.generated_at == NOW and result.model_confidence == 0.8
    assert result.input_tokens == 1500 and result.request_id
    assert result.latency_ms >= 0
    assert result.estimated_cost_usd == pytest.approx(1500 * INPUT_TOKEN_PRICE_USD)
    assert row.model_dump() == before
    assert len(calls) == 1
    payload = calls[0][0]
    assert payload["model"] == JEV_MODEL and set(payload["questions"]) == {"direction"}
    state = payload["state"]
    assert set(state) == {"features", "symbol", "as_of", "anchor_price", "target_time"}
    assert state["features"] == row.features and state["anchor_price"] == "100"
    encoded = json.dumps(payload)
    for forbidden in ("91.12345", "-887.655", "target_price", "target_return_bps", "label_observed_at", "TEST_SECRET"):
        assert forbidden not in encoded
    assert "7 minutes" in encoded and "12" in encoded and "last price" in encoded
    assert set(payload["questions"]["direction"]["criteria"]) == {"DOWN", "FLAT", "UP"}
    with sqlite3.connect(tmp_path / "jev.sqlite3") as connection:
        stored = connection.execute("SELECT result_json, input_tokens FROM jev_attempts").fetchone()
    assert stored[1] == 1500 and result.request_id in stored[0]
    assert "TEST_SECRET" not in stored[0]


def test_outcome_arrival_does_not_change_payload_or_cached_provenance(tmp_path):
    config = ForecastConfig()
    first, calls = service(tmp_path)
    original = first.predict(example(config), config, "dataset")
    later, later_calls = service(tmp_path, clock=lambda: NOW + timedelta(seconds=10))
    cached = later.predict(example(config, labeled=True), config, "dataset")
    assert original.status == "ok" and cached.cached
    assert cached.model_copy(update={"cached": False}) == original
    assert len(calls) == 1 and not later_calls


@pytest.mark.parametrize("age", [-1, 31, 901])
def test_prospective_future_and_stale_examples_never_spend(tmp_path, age):
    advisor, calls = service(tmp_path)
    result = advisor.predict(example(as_of=NOW - timedelta(seconds=age)), ForecastConfig(), "dataset")
    assert result.status == "error" and not calls


@pytest.mark.parametrize("change", ["config", "horizon", "source_age", "symbol"])
def test_invalid_forecast_contract_never_spends(tmp_path, change):
    config = ForecastConfig()
    row = example(config)
    if change == "config":
        row = row.model_copy(update={"config_id": "different"})
    elif change == "horizon":
        row = row.model_copy(update={"target_time": row.target_time + timedelta(seconds=1)})
    elif change == "source_age":
        row = row.model_copy(update={"features": row.features | {"source_age_seconds": 31}})
    else:
        row = row.model_copy(update={"symbol": "OTHER"})
    advisor, calls = service(tmp_path)
    result = advisor.predict(row, config, "dataset", prospective=False)
    assert result.status == "error" and not calls


def test_retrospective_inference_is_explicit_and_cache_separate(tmp_path):
    config = ForecastConfig()
    row = example(as_of=NOW - timedelta(days=30), labeled=True)
    advisor, calls = service(tmp_path)
    result = advisor.predict(row, config, "dataset", prospective=False)
    assert result.status == "ok" and result.reason == "retrospective_api_inference"
    assert advisor.predict(row, config, "dataset").status == "error"
    assert len(calls) == 1


def test_response_arriving_after_target_is_not_a_prospective_prediction(tmp_path):
    now = [NOW]
    advisor, calls = service(tmp_path, clock=lambda: now[0])
    original_transport = advisor._transport

    def transport(payload, key):
        result = original_transport(payload, key)
        now[0] += timedelta(minutes=16)
        return result

    advisor._transport = transport
    result = advisor.predict(example(), ForecastConfig(), "dataset")
    assert result.status == "error" and result.reason == "response_after_target"
    assert result.input_tokens == 1500 and len(calls) == 1


@pytest.mark.parametrize(
    "confidence,probs",
    [
        (0.2, {"DOWN": 0.1, "FLAT": 0.2, "UP": 0.7}),
        (0.8, {"DOWN": 0.3, "FLAT": 0.3, "UP": 0.4}),
    ],
)
def test_both_model_confidence_and_probability_gate_abstention(tmp_path, confidence, probs):
    advisor, _ = service(tmp_path, reply=response(confidence=confidence, probabilities=probs))
    result = advisor.predict(example(), ForecastConfig(), "dataset")
    assert result.status == "abstained" and result.probabilities == probs


@pytest.mark.parametrize(
    "field,value",
    [
        ("probabilities", {"DOWN": 0.1, "FLAT": 0.2, "UP": 0.8}),
        ("probabilities", {"DOWN": -0.1, "FLAT": 0.4, "UP": 0.7}),
        ("probabilities", {"DOWN": 0.1, "FLAT": float("nan"), "UP": 0.9}),
        ("probabilities", {"DOWN": 0.1, "FLAT": float("inf"), "UP": 0.9}),
        ("probabilities", {"DOWN": 0, "FLAT": False, "UP": 1}),
        ("probabilities", {"DOWN": 0.1, "FLAT": 0.2, "UP": 0.6999}),
        ("probabilities", {"DOWN": 0.1, "UP": 0.9}),
        ("probabilities", [0.1, 0.2, 0.7]),
        ("choice", "DOWN"),
        ("choice", "BUY"),
        ("type", "Choice"),
        ("confidence", float("nan")),
        ("confidence", 1.1),
        ("confidence", True),
    ],
)
def test_invalid_responses_fail_closed_without_echoing_provider_content(tmp_path, field, value):
    raw = response()
    raw["answers"]["direction"][field] = value
    raw["private"] = "SECRET_SENTINEL"
    advisor, calls = service(tmp_path, reply=raw)
    result = advisor.predict(example(), ForecastConfig(), "dataset")
    assert result.status == "error" and result.probabilities is None
    assert "SECRET" not in result.model_dump_json() and len(calls) == 1
    assert result.estimated_cost_usd == WORST_COST


@pytest.mark.parametrize(
    "raw",
    [
        response() | {"model": "unpinned"},
        response() | {"usage": {"input_tokens": True}},
        response() | {"usage": {"input_tokens": -1}},
        response() | {"usage": {"input_tokens": MAX_INPUT_TOKENS + 1}},
        response() | {"usage": {"input_tokens": 1.5}},
        {},
    ],
)
def test_model_and_usage_are_strictly_validated(tmp_path, raw):
    advisor, calls = service(tmp_path, reply=raw)
    assert advisor.predict(example(), ForecastConfig(), "dataset").status == "error"
    assert len(calls) == 1


def test_timeout_counts_persistently_without_retry_or_secret_leak(tmp_path):
    advisor, calls = service(tmp_path, reply=TimeoutError("SECRET_SENTINEL"), max_requests=1)
    result = advisor.predict(example(), ForecastConfig(), "dataset")
    assert result.status == "error" and result.estimated_cost_usd == WORST_COST
    assert result.input_tokens is None and "SECRET" not in result.model_dump_json()
    later, later_calls = service(tmp_path, max_requests=1, clock=lambda: NOW + timedelta(seconds=31))
    blocked = later.predict(example(as_of=NOW + timedelta(seconds=31)), ForecastConfig(), "dataset")
    assert blocked.reason == "run_request_limit"
    assert len(calls) == 1 and not later_calls
    assert later.metadata()["requests_used"] == 1


def test_run_caps_cannot_expand_after_recreation_or_from_existing_instance(tmp_path):
    wide, calls = service(tmp_path, max_requests=2)
    first = wide.predict(example(), ForecastConfig(), "dataset")
    assert first.status == "ok"
    service(tmp_path, max_requests=1)
    with pytest.raises(ValueError, match="expand"):
        service(tmp_path, max_requests=2)
    wide._clock = lambda: NOW + timedelta(seconds=31)
    assert wide.predict(example(as_of=wide._clock()), ForecastConfig(), "dataset").reason == "run_request_limit"
    assert len(calls) == 1


def test_run_spend_reserves_unknown_attempts_at_worst_case_and_settles_valid_usage(tmp_path):
    config = ForecastConfig()
    budget = WORST_COST + 1500 * INPUT_TOKEN_PRICE_USD
    advisor, calls = service(tmp_path, max_cost_usd=budget)
    assert advisor.predict(example(), config, "dataset").status == "ok"
    later, later_calls = service(tmp_path, max_cost_usd=budget, clock=lambda: NOW + timedelta(seconds=31))
    assert later.predict(example(as_of=NOW + timedelta(seconds=31)), config, "dataset").status == "ok"
    final, final_calls = service(tmp_path, max_cost_usd=budget, clock=lambda: NOW + timedelta(seconds=62))
    assert final.predict(example(as_of=NOW + timedelta(seconds=62)), config, "dataset").reason == "run_cost_limit"
    assert len(calls) == len(later_calls) == 1 and not final_calls
    with pytest.raises(ValueError, match="expand"):
        service(tmp_path, max_cost_usd=budget + 0.001)


def test_insufficient_worst_case_budget_never_sends(tmp_path):
    advisor, calls = service(tmp_path, max_cost_usd=WORST_COST / 2)
    result = advisor.predict(example(), ForecastConfig(), "dataset")
    assert result.reason == "run_cost_limit" and not calls


def test_concurrent_instances_cannot_overrun_persisted_budget(tmp_path):
    config = ForecastConfig()
    start = Barrier(2)

    def predict(symbol):
        advisor, calls = service(tmp_path, max_requests=1)
        start.wait(timeout=5)
        return advisor.predict(example(symbol=symbol), config, "dataset"), calls

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(predict, ("AAPL", "MSFT")))
    assert sum(len(calls) for _, calls in results) == 1
    assert sorted(result.status for result, _ in results) == ["error", "ok"]


def test_global_ledger_cap_and_cooldown_include_legacy_advisor_calls(tmp_path):
    path = tmp_path / "jev.sqlite3"
    market_calls = []

    def transport(payload, key):
        market_calls.append(payload)
        return {
            "model": JEV_MODEL,
            "usage": {"input_tokens": 100},
            "answers": {
                "action": {
                    "type": "choice",
                    "choice": "WAIT",
                    "confidence": 0.8,
                    "probabilities": {"BUY": 0, "HOLD": 0, "SELL": 0, "WAIT": 1},
                }
            },
        }

    market = JevAdvisor("TEST_SECRET", path, transport=transport, clock=lambda: NOW)
    assert market.check()["status"] == "available"
    advisor, calls = service(tmp_path)
    assert advisor.predict(example(), ForecastConfig(), "dataset").reason == "cooldown"
    assert not calls
    with sqlite3.connect(path) as connection:
        connection.executemany(
            "INSERT INTO jev_attempts(requested_at, fingerprint, request_json) VALUES (?, ?, ?)",
            [(NOW.timestamp() - 60, str(index), "{}") for index in range(99)],
        )
    later, later_calls = service(tmp_path, clock=lambda: NOW + timedelta(seconds=31))
    assert (
        later.predict(example(as_of=NOW + timedelta(seconds=31)), ForecastConfig(), "dataset").reason
        == "global_request_limit"
    )
    assert len(market_calls) == 1 and not later_calls and later.metadata()["requests_used"] == 100


def test_forecast_attempt_blocks_market_advisor_during_shared_cooldown(tmp_path):
    advisor, calls = service(tmp_path)
    assert advisor.predict(example(), ForecastConfig(), "dataset").status == "ok"
    market = JevAdvisor(
        "TEST_SECRET",
        tmp_path / "jev.sqlite3",
        transport=lambda *_: pytest.fail("network"),
        clock=lambda: NOW + timedelta(seconds=2),
    )
    assert market.check()["status"] == "cooldown" and len(calls) == 1


def test_fixture_is_explicit_offline_deterministic_and_label_independent(monkeypatch):
    from tradecopilot.forecast.jev import fixture_predictions

    monkeypatch.setattr("tradecopilot.jev._post", lambda *_: pytest.fail("network"))
    config = ForecastConfig()
    first, second = fixture_predictions([example(config), example(config, labeled=True)], "dataset", config)
    assert first.probabilities == second.probabilities
    assert first.execution == second.execution == "fixture"
    assert first.model_id == "jev-contract-fixture"
    assert first.reason == "offline_contract_fixture_not_model_performance"
    assert first.estimated_cost_usd == 0 and first.input_tokens == 0 and first.request_id is None
    invalid = fixture_predictions([example().model_copy(update={"config_id": "different"})], "dataset", config)[0]
    assert invalid.status == "error" and invalid.execution == "fixture"


def test_pending_transport_is_charged_conservatively_for_concurrent_instances(tmp_path):
    from threading import Event

    from tradecopilot.forecast.jev import JevForecaster

    entered, release = Event(), Event()

    def transport(payload, key):
        entered.set()
        assert release.wait(timeout=5)
        return response()

    first = JevForecaster(
        "TEST_SECRET",
        tmp_path / "jev.sqlite3",
        transport=transport,
        clock=lambda: NOW,
        max_cost_usd=WORST_COST * 1.5,
    )
    second, calls = service(
        tmp_path,
        clock=lambda: NOW + timedelta(seconds=31),
        max_cost_usd=WORST_COST * 1.5,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(first.predict, example(), ForecastConfig(), "dataset")
        try:
            assert entered.wait(timeout=5)
            result = second.predict(example(as_of=NOW + timedelta(seconds=31)), ForecastConfig(), "dataset")
            assert result.status == "error" and result.reason == "run_cost_limit" and not calls
        finally:
            release.set()
        assert pending.result().status == "ok"


def test_default_run_hard_cap_is_ten_after_many_restarts(tmp_path):
    for index in range(11):
        current = NOW + timedelta(seconds=31 * index)
        advisor, calls = service(tmp_path, clock=lambda current=current: current)
        result = advisor.predict(example(as_of=current), ForecastConfig(), "dataset")
        assert len(calls) == int(index < 10)
        assert result.status == ("ok" if index < 10 else "error")
    assert result.reason == "run_request_limit"
    assert advisor.metadata()["requests_used"] == 10


def test_delayed_response_completion_is_preserved_on_cache_hit(tmp_path):
    now = [NOW]
    advisor, calls = service(tmp_path, clock=lambda: now[0])
    original_transport = advisor._transport

    def transport(payload, key):
        result = original_transport(payload, key)
        now[0] += timedelta(seconds=4)
        return result

    advisor._transport = transport
    prediction = advisor.predict(example(), ForecastConfig(), "dataset")
    assert prediction.generated_at == NOW + timedelta(seconds=4)
    now[0] += timedelta(seconds=4)
    cached = advisor.predict(example(), ForecastConfig(), "dataset")
    assert cached.cached and cached.generated_at == prediction.generated_at and len(calls) == 1


def test_diagnostic_after_forecast_preserves_legacy_response_contract(tmp_path):
    advisor, _ = service(tmp_path)
    assert advisor.predict(example(), ForecastConfig(), "dataset").status == "ok"
    market = JevAdvisor(
        "TEST_SECRET",
        tmp_path / "jev.sqlite3",
        clock=lambda: NOW + timedelta(seconds=31),
        transport=lambda *_: {
            "model": JEV_MODEL,
            "usage": {"input_tokens": 25},
            "answers": {
                "action": {
                    "type": "choice",
                    "choice": "WAIT",
                    "confidence": 0.8,
                    "probabilities": {"BUY": 0, "HOLD": 0, "SELL": 0, "WAIT": 1},
                }
            },
        },
    )
    result = market.check()
    assert result["status"] == "available" and result["requests_used"] == 2
    assert result["input_tokens"] == 25


def test_failed_transport_records_completion_time_without_exception_text(tmp_path):
    now = [NOW]
    advisor, _ = service(tmp_path, clock=lambda: now[0])

    def transport(payload, key):
        now[0] += timedelta(seconds=5)
        raise TimeoutError("SECRET_SENTINEL")

    advisor._transport = transport
    result = advisor.predict(example(), ForecastConfig(), "dataset")
    assert result.status == "error" and result.generated_at == NOW + timedelta(seconds=5)
    assert "SECRET" not in result.model_dump_json()


def test_preflight_and_cooldown_have_explicit_zero_api_usage(tmp_path):
    advisor, calls = service(tmp_path)
    rejected = advisor.predict(example(as_of=NOW - timedelta(minutes=1)), ForecastConfig(), "dataset")
    assert rejected.status == "error" and rejected.input_tokens == 0 and rejected.estimated_cost_usd == 0
    assert not calls
    assert advisor.predict(example(), ForecastConfig(), "dataset").status == "ok"
    cooldown = advisor.predict(example(symbol="MSFT"), ForecastConfig(), "dataset")
    assert cooldown.reason == "cooldown" and cooldown.input_tokens == 0 and cooldown.estimated_cost_usd == 0
    assert len(calls) == 1


def test_settlement_failure_retains_reservation_and_records_error_completion(tmp_path):
    now = [NOW]
    advisor, calls = service(tmp_path, clock=lambda: now[0])
    connect = advisor._connect
    connections = []

    def unavailable_settlement():
        connections.append(True)
        if len(connections) == 2:
            now[0] += timedelta(seconds=3)
            raise sqlite3.OperationalError("SECRET_SENTINEL")
        return connect()

    advisor._connect = unavailable_settlement
    result = advisor.predict(example(), ForecastConfig(), "dataset")
    assert result.status == "error" and result.reason == "ledger_settlement_unavailable"
    assert result.generated_at == NOW + timedelta(seconds=3)
    assert result.estimated_cost_usd == WORST_COST and result.input_tokens is None
    assert "SECRET" not in result.model_dump_json() and len(calls) == 1
    with sqlite3.connect(tmp_path / "jev.sqlite3") as connection:
        assert connection.execute("SELECT input_tokens, result_json FROM jev_attempts").fetchone() == (None, None)
