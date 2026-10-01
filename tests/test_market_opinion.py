from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta

import pytest


def snapshot(now, *, age=0):
    source = now - timedelta(seconds=age)
    return {
        "ready": True,
        "complete": False,
        "error": None,
        "meta": {
            "symbol": "AAPL",
            "mode": "live",
            "price_only": True,
            "data_provider": "finnhub",
            "state": "DATA_INSUFFICIENT",
            "position_status": "UNKNOWN",
            "quote_time": source.isoformat(),
            "quote_age": age,
        },
        "quote": {"last": 101, "bid": None, "ask": None, "total_volume": None},
        "market_context": {
            "previous_close": 99,
            "session_open": 100,
            "session_high": 102,
            "session_low": 98,
            "price_history": [],
        },
        "missing": ["bid/ask", "minute OHLCV", "account context"],
        "day_stop": {"locked": True},
        "positions": [{"account_alias": "DO_NOT_SEND", "quantity": 25}],
        "credentials": {"api_key": "SECRET"},
    }


def response(action="BUY"):
    return {
        "model": "jev-1.13.0",
        "answers": {
            "action": {
                "type": "choice",
                "choice": action,
                "confidence": 0.9,
                "probabilities": {key: (0.94 if key == action else 0.02) for key in ("BUY", "HOLD", "SELL", "WAIT")},
            }
        },
        "usage": {"input_tokens": 900, "output_tokens": 32},
    }


def test_market_opinion_is_independent_of_broker_and_strategy_blocks(tmp_path):
    from tradecopilot.jev import JevAdvisor

    now = datetime.now(UTC)
    current = snapshot(now)
    original = copy.deepcopy(current)
    calls = []

    def transport(payload, key):
        calls.append(payload)
        return response()

    advisor = JevAdvisor("test", tmp_path / "ledger.sqlite3", transport=transport, clock=lambda: now)
    result = advisor.advise(current)
    assert result["status"] == "available" and result["action"] == "BUY"
    assert result["assessment_mode"] == "market_opinion"
    assert result["context_kind"] == "session_summary"
    assert result["limitations"] and result["source_time"] == current["meta"]["quote_time"]
    assert current == original
    assert len(calls) == 1
    assert "DO_NOT_SEND" not in str(calls) and "SECRET" not in str(calls)
    assert "positions" not in calls[0]["state"] and "day_stop" not in calls[0]["state"]


def test_last_session_opinion_preserves_old_asof_without_claiming_live_prediction(tmp_path):
    from tradecopilot.jev import JevAdvisor

    now = datetime.now(UTC)
    advisor = JevAdvisor(
        "test", tmp_path / "ledger.sqlite3", transport=lambda p, k: response("HOLD"), clock=lambda: now
    )
    current = snapshot(now, age=12 * 3600)
    result = advisor.advise(current)
    assert result["status"] == "available" and result["action"] == "HOLD"
    assert result["context_kind"] == "session_summary"
    assert result["source_time"] == current["meta"]["quote_time"]
    assert datetime.fromisoformat(result["expires_at"]) > now
    assert "session" in result["message"].lower()


def test_one_price_is_not_enough_and_costs_nothing(tmp_path):
    from tradecopilot.jev import JevAdvisor

    now = datetime.now(UTC)
    current = snapshot(now)
    current["market_context"] = {"price_history": []}
    calls = []
    advisor = JevAdvisor("test", tmp_path / "ledger.sqlite3", transport=lambda p, k: calls.append(p), clock=lambda: now)
    assert advisor.advise(current)["status"] == "blocked"
    assert not calls and advisor.metadata()["requests_used"] == 0


@pytest.mark.parametrize("change", [{"mode": "replay"}, {"mode": "mock"}, {"quote_age": 999999}])
def test_invalid_source_still_blocks_without_spending(tmp_path, change):
    from tradecopilot.jev import JevAdvisor

    now = datetime.now(UTC)
    current = snapshot(now)
    current["meta"].update(change)
    calls = []
    advisor = JevAdvisor("test", tmp_path / "ledger.sqlite3", transport=lambda p, k: calls.append(p), clock=lambda: now)
    assert advisor.advise(current)["status"] == "blocked"
    assert not calls


def test_recent_history_can_support_opinion_without_session_fields(tmp_path):
    from tradecopilot.jev import JevAdvisor

    now = datetime.now(UTC)
    current = snapshot(now)
    current["market_context"] = {
        "price_history": [
            {"timestamp": (now - timedelta(seconds=75 - 15 * i)).isoformat(), "price": 100 + i / 5} for i in range(6)
        ]
    }
    advisor = JevAdvisor(
        "test", tmp_path / "ledger.sqlite3", transport=lambda p, k: response("SELL"), clock=lambda: now
    )
    result = advisor.advise(current)
    assert result["status"] == "available" and result["action"] == "SELL"
    assert result["context_kind"] == "recent_prices"


@pytest.mark.parametrize("bad", ["duplicate", "future", "other_symbol", "not_finite", "unsorted"])
def test_invalid_history_never_supplies_fake_context(tmp_path, bad):
    from tradecopilot.jev import JevAdvisor

    now = datetime.now(UTC)
    current = snapshot(now)
    rows = [{"timestamp": (now - timedelta(seconds=75 - 15 * i)).isoformat(), "price": 100 + i / 5} for i in range(6)]
    if bad == "duplicate":
        rows = [rows[-1]] * 6
    elif bad == "future":
        rows[-1]["timestamp"] = (now + timedelta(seconds=10)).isoformat()
    elif bad == "other_symbol":
        rows[-1]["symbol"] = "MSFT"
    elif bad == "not_finite":
        rows[-1]["price"] = float("nan")
    else:
        rows.reverse()
    current["market_context"] = {"price_history": rows}
    calls = []
    advisor = JevAdvisor("test", tmp_path / "ledger.sqlite3", transport=lambda p, k: calls.append(p), clock=lambda: now)
    assert advisor.advise(current)["status"] == "blocked"
    assert not calls


def test_explicit_wait_is_reported_as_uncertainty_and_shares_existing_budget(tmp_path):
    from tradecopilot.jev import JevAdvisor

    now = datetime.now(UTC)
    advisor = JevAdvisor(
        "test", tmp_path / "ledger.sqlite3", transport=lambda p, k: response("WAIT"), clock=lambda: now, request_limit=1
    )
    result = advisor.advise(snapshot(now))
    assert result["status"] == "uncertain" and result["action"] is None
    assert result["probabilities"]["WAIT"] == 0.94
    assert advisor.check()["status"] == "budget_exhausted"
