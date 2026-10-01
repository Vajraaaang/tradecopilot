from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest


def snapshot(now: datetime) -> dict:
    return {
        "ready": True,
        "complete": False,
        "error": None,
        "meta": {
            "mode": "live",
            "symbol": "TEST",
            "state": "BUY",
            "quote_time": now.isoformat(),
            "quote_age": 0.0,
            "event_time_et": now.isoformat(),
            "position_status": "FLAT",
        },
        "quote": {"last": 10.0, "bid": 9.99, "ask": 10.01, "total_volume": 123456},
        "indicators": {"vwap": 9.8, "ema9_1m": 9.9},
        "pattern": {"retracement_percent": 25.0},
        "plan": {"trigger": 10.0, "stop": 9.8, "reward_risk": 3.0, "maximum_shares": 100},
        "quality": {"catalyst": "Verified earnings", "catalyst_verified": True, "float_shares": 1000000},
        "level2": {"persistent_seller": "absent", "tape_available": True},
        "day_stop": {"locked": False, "realized": -25},
        "missing": [],
        "positions": [{"account_alias": "SECRET_ACCOUNT", "quantity": 123}],
        "credentials": {"api_key": "SECRET_SENTINEL"},
        "charts": {"future": "DO_NOT_SEND"},
    }


def response(*, action="BUY", confidence=0.8, probabilities=None) -> dict:
    return {
        "model": "jev-1.13.0",
        "answers": {
            "action": {
                "type": "choice",
                "choice": action,
                "probabilities": probabilities or {"BUY": 0.85, "HOLD": 0.02, "SELL": 0.01, "WAIT": 0.12},
                "confidence": confidence,
            }
        },
        "usage": {"input_tokens": 1500, "output_tokens": 32},
    }


def advisor(tmp_path, *, reply=None, clock=None, **kwargs):
    from tradecopilot.jev import JevAdvisor

    calls = []

    def transport(payload, api_key):
        calls.append((copy.deepcopy(payload), api_key))
        if isinstance(reply, Exception):
            raise reply
        return reply if reply is not None else response()

    service = JevAdvisor(
        "test-secret",
        tmp_path / "jev.sqlite3",
        transport=transport,
        clock=clock or (lambda: datetime.now(UTC)),
        **kwargs,
    )
    return service, calls


def test_valid_advice_is_bounded_sanitized_and_does_not_mutate_input(tmp_path):
    now = datetime.now(UTC)
    service, calls = advisor(tmp_path, clock=lambda: now)
    current = snapshot(now)
    original = copy.deepcopy(current)
    result = service.advise(current)
    assert result["status"] == "available"
    assert result["action"] == "BUY"
    assert result["requests_used"] == 1
    assert result["input_tokens"] == 1500
    assert result["estimated_cost_usd"] == pytest.approx(0.000063)
    assert result["source_time"] == now.isoformat()
    assert current == original
    serialized = json.dumps(calls[0][0])
    assert not any(secret in serialized for secret in ("SECRET", "DO_NOT_SEND", "maximum_shares", "realized"))
    assert "test-secret" not in json.dumps(result)
    assert calls[0][0]["model"] == "jev-1.13.0"


@pytest.mark.parametrize("mode", ["replay", "mock"])
def test_simulation_never_spends_credit(tmp_path, mode):
    now = datetime.now(UTC)
    service, calls = advisor(tmp_path)
    current = snapshot(now)
    current["meta"]["mode"] = mode
    assert service.advise(current)["status"] == "blocked"
    assert calls == []
    assert service.metadata()["requests_used"] == 0


@pytest.mark.parametrize("state", ["DATA_STALE", "DATA_INSUFFICIENT", "DAY_STOP", "SELL", "NO_TRADE"])
def test_risk_states_never_call_jev(tmp_path, state):
    now = datetime.now(UTC)
    service, calls = advisor(tmp_path)
    current = snapshot(now)
    current["meta"]["state"] = state
    assert service.advise(current)["status"] == "blocked"
    assert calls == []


def test_freshness_uses_wall_clock_and_rejects_future_timestamps(tmp_path):
    now = datetime.now(UTC)
    service, calls = advisor(tmp_path, clock=lambda: now)
    for age in (3, -10):
        current = snapshot(now - timedelta(seconds=age))
        current["meta"]["quote_age"] = 0
        assert service.advise(current)["status"] == "blocked"
    assert calls == []


@pytest.mark.parametrize(
    "change",
    [
        {"ready": False},
        {"error": "provider stopped"},
        {"complete": True},
        {"day_stop": {"locked": True}},
        {"missing": ["no bars"]},
    ],
)
def test_unavailable_inputs_do_not_call_provider(tmp_path, change):
    service, calls = advisor(tmp_path)
    current = {**snapshot(datetime.now(UTC)), **change}
    assert service.advise(current)["status"] == "blocked"
    assert calls == []


def test_duplicate_requests_and_restart_use_cached_result(tmp_path):
    now = datetime.now(UTC)
    first, calls = advisor(tmp_path, clock=lambda: now)
    current = snapshot(now)
    first.advise(current)
    second, other_calls = advisor(tmp_path, clock=lambda: now)
    result = second.advise(current)
    assert result["cached"] is True
    assert result["requests_used"] == 1
    assert len(calls) == 1 and not other_calls


def test_concurrent_instances_share_attempt_limit(tmp_path):
    now = datetime.now(UTC)
    first, calls = advisor(tmp_path, clock=lambda: now, request_limit=1)
    second, other_calls = advisor(tmp_path, clock=lambda: now, request_limit=1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda svc: svc.advise(snapshot(now)), (first, second)))
    assert len(calls) + len(other_calls) == 1
    assert all(item["requests_used"] == 1 for item in results)


def test_failures_are_counted_without_retries_and_persist_across_restart(tmp_path):
    now = datetime.now(UTC)
    first, calls = advisor(tmp_path, reply=TimeoutError("SECRET_SENTINEL"), clock=lambda: now, request_limit=1)
    result = first.advise(snapshot(now))
    assert result["status"] == "unavailable" and result["action"] is None
    assert "SECRET" not in json.dumps(result)
    second, other_calls = advisor(tmp_path, clock=lambda: now + timedelta(seconds=60), request_limit=1)
    assert second.advise(snapshot(now + timedelta(seconds=60)))["status"] == "budget_exhausted"
    assert len(calls) == 1 and not other_calls


def test_cooldown_survives_restart_and_symbol_changes(tmp_path):
    now = datetime.now(UTC)
    first, calls = advisor(tmp_path, clock=lambda: now)
    first.advise(snapshot(now))
    later = now + timedelta(seconds=5)
    second, other_calls = advisor(tmp_path, clock=lambda: later)
    current = snapshot(later)
    current["meta"]["symbol"] = "OTHER"
    assert second.advise(current)["status"] == "cooldown"
    assert len(calls) == 1 and not other_calls


@pytest.mark.parametrize(
    "probabilities",
    [
        {"BUY": 0.8},
        {"BUY": 2, "HOLD": -1, "SELL": 0, "WAIT": 0},
        {"BUY": float("nan"), "HOLD": 0, "SELL": 0, "WAIT": 1},
        {"BUY": 0.1, "HOLD": 0.1, "SELL": 0.1, "WAIT": 0.1},
        {"BUY": 0.1, "HOLD": 0.1, "SELL": 0.1, "WAIT": 0.7},
    ],
)
def test_malformed_or_inconsistent_probabilities_fail_closed(tmp_path, probabilities):
    service, calls = advisor(tmp_path, reply=response(probabilities=probabilities))
    result = service.advise(snapshot(datetime.now(UTC)))
    assert result["status"] == "unavailable"
    assert result["action"] is None
    assert len(calls) == 1


def test_low_confidence_abstains_instead_of_claiming_hold(tmp_path):
    service, _ = advisor(tmp_path, reply=response(confidence=0.2))
    result = service.advise(snapshot(datetime.now(UTC)))
    assert result["status"] == "uncertain" and result["action"] is None
    assert result["probabilities"]["BUY"] == 0.85


def test_jev_cannot_override_entry_guard_or_invent_a_position(tmp_path):
    service, _ = advisor(tmp_path)
    current = snapshot(datetime.now(UTC))
    current["meta"]["state"] = "ARMED"
    result = service.advise(current)
    assert result["status"] == "blocked" and result["action"] is None


def test_oversized_context_does_not_spend_credit(tmp_path):
    service, calls = advisor(tmp_path)
    current = snapshot(datetime.now(UTC))
    current["quality"]["catalyst"] = "a" * 30000
    assert service.advise(current)["status"] == "blocked"
    assert not calls


def test_check_is_explicit_and_uses_same_budget(tmp_path):
    service, calls = advisor(tmp_path, request_limit=1)
    result = service.check()
    assert result["status"] == "available"
    assert result["action"] is None
    assert "synthetic" in result["message"].lower()
    assert service.check()["status"] == "budget_exhausted"
    assert len(calls) == 1


@pytest.mark.parametrize("status", [200, 401, 429, 500])
def test_http_transport_uses_official_endpoint_without_retry_or_secret_errors(monkeypatch, status):
    from tradecopilot.jev import _post

    events = []

    class Reply:
        def __init__(self):
            self.status = status

        def read(self, limit):
            return json.dumps(response() if status == 200 else {"error": "SECRET_SENTINEL"}).encode()[:limit]

    class Connection:
        def __init__(self, host, timeout):
            events.append((host, timeout))

        def request(self, method, path, body, headers):
            events.append((method, path, json.loads(body), headers))

        def getresponse(self):
            return Reply()

        def close(self):
            events.append("closed")

    monkeypatch.setattr("tradecopilot.jev.http.client.HTTPSConnection", Connection)
    if status == 200:
        assert _post({"state": "test"}, "SECRET_SENTINEL") == response()
    else:
        with pytest.raises(ConnectionError) as failure:
            _post({"state": "test"}, "SECRET_SENTINEL")
        assert "SECRET" not in str(failure.value)
    assert events[0] == ("api.typesafe.ai", 5)
    assert events[1][:2] == ("POST", "/v1/systemone")
    assert events[1][3]["Authorization"] == "Bearer SECRET_SENTINEL"
    assert len(events) == 3 and events[-1] == "closed"


def test_slow_response_expires_without_an_action(tmp_path):
    from tradecopilot.jev import JevAdvisor

    now = [datetime.now(UTC)]
    current = snapshot(now[0])

    def slow(payload, key):
        now[0] += timedelta(seconds=31)
        return response()

    service = JevAdvisor("test", tmp_path / "jev.sqlite3", transport=slow, clock=lambda: now[0])
    result = service.advise(current)
    assert result["status"] == "unavailable" and result["action"] is None
    assert result["input_tokens"] == 1500


@pytest.mark.parametrize("position,choice", [("FLAT", "HOLD"), ("FLAT", "SELL"), ("OPEN", "BUY"), ("OPEN", "WAIT")])
def test_position_semantics_cannot_be_overridden(tmp_path, position, choice):
    probabilities = {label: float(label == choice) for label in ("BUY", "HOLD", "SELL", "WAIT")}
    service, _ = advisor(tmp_path, reply=response(action=choice, probabilities=probabilities))
    current = snapshot(datetime.now(UTC))
    current["meta"]["position_status"] = position
    result = service.advise(current)
    assert result["status"] == "blocked" and result["action"] is None


@pytest.mark.parametrize("section", ["meta", "quote", "day_stop", "indicators", "plan", "pattern", "quality", "level2"])
def test_missing_or_malformed_required_sections_are_blocked_before_spending(tmp_path, section):
    service, calls = advisor(tmp_path)
    current = snapshot(datetime.now(UTC))
    current[section] = None
    assert service.advise(current)["status"] == "blocked"
    current.pop(section)
    assert service.advise(current)["status"] == "blocked"
    assert not calls


def test_incomplete_buy_plan_cannot_be_sent_to_provider(tmp_path):
    service, calls = advisor(tmp_path)
    current = snapshot(datetime.now(UTC))
    current["plan"] = {}
    assert service.advise(current)["status"] == "blocked"
    assert not calls


def test_wire_encoding_is_the_same_encoding_used_for_input_cap(monkeypatch):
    from tradecopilot.jev import _post

    payload = {"state": {"text": "a b", "price": 10}, "model": "jev-1.13.0"}

    class Connection:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, method, path, *, body, headers):
            assert body == json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

        def getresponse(self):
            return self

        status = 200

        def read(self, limit):
            return json.dumps(response()).encode()

        def close(self):
            pass

    monkeypatch.setattr("tradecopilot.jev.http.client.HTTPSConnection", Connection)
    assert _post(payload, "test") == response()
