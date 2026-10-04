"""Synthetic API contracts only; these tests never contact a provider."""
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from tradecopilot.forecast.bars import HistoricalBar, write_bar_dataset
from tradecopilot.forecast.contracts import content_hash
from tradecopilot.forecast.sessions import session_bounds
from tradecopilot.rl.data import prepare_data

SYMBOLS = ["AAPL", "AMZN", "MSFT", "NFLX", "TSLA"]


def test_cache_api_exists():
    assert importlib.util.find_spec("tradecopilot.forecast.jev_rl") is not None, "paid cache boundary missing"


def synthetic(tmp_path: Path, future_price: int = 100):
    reg = {"fixture": True, "forecast_profile": "MARKET_ONLY", "symbols": SYMBOLS,
           "source_range": {"start": "2025-11-03", "end_exclusive": "2025-11-07", "provider": "Alpaca",
                            "feed": "sip", "adjustment": "raw"},
           "splits": {"train": ["2025-11-04"] * 2, "tune": ["2025-11-05"] * 2,
                      "test": ["2025-11-06"] * 2}, "environment": {"warmup_minutes": 60},
           "jev": {"model": "jev-1.13.0", "prompt_version": "masked-normalized-ohlcv-3h-v1",
                   "as_of_minutes_after_open": 61, "horizons": ["15m", "60m", "close"],
                   "flat_threshold_bps": 10, "simulated_latency_seconds": 60, "abstention_threshold": 0.6},
           "api_budget": {"max_attempts": 80, "max_conservative_cost_usd": 0.25,
                          "max_requests_per_run": 10, "max_cost_per_run_usd": 0.05,
                          "global_limit": 100, "cooldown_seconds": 30, "initial_global_attempts": 15,
                          "retry_provider_failures": False, "abort_on_invalid_response": True}}
    reg["registration_id"] = content_hash(reg)
    tmp_path.mkdir(parents=True, exist_ok=True)
    rp = tmp_path / "registration.json"
    rp.write_text(json.dumps(reg))
    bars = []
    for day in range(3, 7):
        bounds = session_bounds(date(2025, 11, day))
        assert bounds
        for symbol in SYMBOLS:
            for minute in range(65):
                start = bounds[0] + timedelta(minutes=minute)
                price = Decimal(future_price if minute >= 61 else 100) + Decimal(minute) / 100
                bars.append(HistoricalBar(symbol=symbol, start_time=start, end_time=start + timedelta(minutes=1),
                                          available_at=start + timedelta(minutes=1), opening=price,
                                          high=price, low=price, close=price, volume=Decimal(100 + minute),
                                          source="alpaca_sip_1min_bar"))
    metadata = {"fixture": True, "provider": "Alpaca", "feed": "sip", "adjustment_policy": "raw",
                "start_date": "2025-11-03", "end_date_exclusive": "2025-11-07", "selected_symbols": SYMBOLS}
    bp, pp = tmp_path / "bars", tmp_path / "prepared"
    write_bar_dataset(bp, bars, metadata)
    prepare_data(bp, rp, pp)
    from tradecopilot.forecast.jev_rl import prepare_requests
    plan = prepare_requests(bp, rp, pp, tmp_path / "requests")
    return plan, json.loads(plan.read_text())


def reply(payload):
    return {"model": "jev-1.13.0", "answers": {key: {"type": "choice", "choice": "UP",
            "probabilities": {"DOWN": 0.1, "FLAT": 0.2, "UP": 0.7}, "confidence": 0.8}
            for key in payload["questions"]}, "usage": {"input_tokens": 123, "output_tokens": 90}}


def test_masked_future_isolation_same_asof_and_timing(tmp_path):
    a, left = synthetic(tmp_path / "a")
    b, right = synthetic(tmp_path / "b", future_price=999)
    assert len(left["batches"]) == 3
    for x, y in zip(left["batches"], right["batches"], strict=True):
        px = json.loads((a.parent / x["payload_path"]).read_text())
        py = json.loads((b.parent / y["payload_path"]).read_text())
        assert px == py
        encoded = json.dumps(px)
        assert len(encoded.encode()) <= 16_000
        assert all(symbol not in encoded for symbol in SYMBOLS)
        assert "2025-" not in encoded and "anchor_price" not in encoded and "outcome" not in encoded
        assert len(px["questions"]) == 15
        assert len(px["state"]["stocks"]) == 5
        assert all(len(stock["recent_bars"]) == 15 for stock in px["state"]["stocks"].values())
        assert len({r["as_of"] for r in x["inputs"]}) == 1
        for key, q in px["questions"].items():
            instructions = q["instructions"]
            assert instructions["question_key"] == key
            stock_key, horizon = key.split("_")
            minutes = int(horizon[:-1]) if horizon != "close" else px["state"]["minutes_until_close"]
            assert instructions["target_minutes_after_asof"] == minutes
            assert f"state.stocks.{stock_key}" in instructions["task"]
            assert f"{minutes} minutes after as-of" in instructions["task"]
            assert "minute-bar close proxy at target minute-end relative to as-of completed bar close" \
                in instructions["task"]
            assert "state.stocks[stock_key]" not in instructions["task"]
            assert all("minute-bar close proxy" in value and "as-of completed bar close" in value
                       for value in q["criteria"].values())
        for r in x["inputs"]:
            assert r["replay_available_at"] == r["as_of"] + 60
            if r["horizon_key"] != "close":
                assert r["target_time"] - r["as_of"] == int(r["horizon_key"][:-1]) * 60


@pytest.mark.parametrize("mutation", ["sum", "argmax", "nan", "keys", "model", "confidence"])
def test_invalid_answers_rejected(mutation):
    from tradecopilot.forecast.jev_rl import _validate_answers
    raw = reply({"questions": {"S0_15m": {}}})
    answer = raw["answers"]["S0_15m"]
    if mutation == "sum":
        answer["probabilities"]["UP"] = 0.5
    elif mutation == "argmax":
        answer["choice"] = "DOWN"
    elif mutation == "nan":
        answer["probabilities"]["UP"] = float("nan")
    elif mutation == "keys":
        raw["answers"]["extra"] = answer
    elif mutation == "model":
        raw["model"] = "jev-latest"
    else:
        answer["confidence"] = 1.1
    with pytest.raises(ValueError):
        _validate_answers(raw, {"S0_15m"})


def test_collection_real_times_seal_and_corruption(tmp_path, monkeypatch):
    import tradecopilot.forecast.jev_rl as module
    plan, _ = synthetic(tmp_path)
    calls = []
    def transport(payload, key):
        calls.append(payload)
        assert key == "synthetic-secret"
        return reply(payload)
    monkeypatch.setattr(module, "_post", transport)
    monkeypatch.setattr(module, "_wait_cooldown", lambda seconds: None)
    # Synthetic time advances across attempts; replay timestamps never use it.
    times = iter(datetime(2026, 10, 4, tzinfo=UTC) + timedelta(seconds=i * 31) for i in range(30))
    monkeypatch.setattr(module, "_now", lambda: next(times))
    path = module.collect_cache(plan, tmp_path / "cache", api_key="synthetic-secret", ledger_path=tmp_path / "ledger")
    cache = module.load_cache(path)
    assert len(calls) == 3 and len(cache["records"]) == 45
    assert cache["usage"]["attempts"] == 3 and cache["usage"]["input_tokens"] == 369
    assert all(datetime.fromisoformat(r["generated_at"]).year == 2026 for r in cache["records"])
    artifact = Path(cache["inventory"][-1]["path"])
    artifact.write_bytes(artifact.read_bytes() + b" ")
    with pytest.raises(ValueError, match="artifact"):
        module.load_cache(path)


@pytest.mark.parametrize("mode,tokens", [("invalid", 123), ("unknown", 64_000), ("failure", 64_000)])
def test_failure_stops_no_retry_conservative_settlement(tmp_path, monkeypatch, mode, tokens):
    import sqlite3

    import tradecopilot.forecast.jev_rl as module
    plan, _ = synthetic(tmp_path)
    calls = []
    def transport(payload, key):
        calls.append(1)
        if mode == "failure":
            raise RuntimeError("private-secret MUST NOT BE LOGGED")
        raw = reply(payload)
        raw["answers"] = {}
        if mode == "unknown":
            raw["usage"]["input_tokens"] = True
        return raw
    monkeypatch.setattr(module, "_post", transport)
    ledger = tmp_path / "ledger"
    out = tmp_path / "cache"
    with pytest.raises(ValueError, match="stopped"):
        module.collect_cache(plan, out, api_key="synthetic-secret", ledger_path=ledger)
    assert len(calls) == 1 and not (out / "cache.json").exists()
    partial = (out / "partial.json").read_text()
    assert "private-secret" not in partial and "synthetic-secret" not in partial
    with sqlite3.connect(ledger) as c:
        assert c.execute("SELECT input_tokens FROM jev_attempts").fetchone()[0] == tokens
    with pytest.raises(ValueError):
        module.collect_cache(plan, out, api_key="synthetic-secret", ledger_path=ledger)
    assert len(calls) == 1


def test_atomic_ledger_unique_batch_global_and_experiment_caps(tmp_path):
    from tradecopilot.forecast.jev_rl import _BatchLedger
    ledger = _BatchLedger("synthetic-secret", tmp_path / "ledger", "registration", "plan")
    def reserve(i):
        return ledger.reserve(i, "{}", datetime(2026, 10, 4, tzinfo=UTC))[0]
    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(reserve, [0] * 8))
    assert sum(value is not None for value in values) == 1
    with ledger._connect() as c:
        c.execute("UPDATE jev_attempts SET requested_at = 0")
    assert reserve(0) is None  # Unique even after cooldown expires.
    for i in range(1, 80):
        attempt, reason, _ = ledger.reserve(i, "{}", datetime(2026, 10, 4, tzinfo=UTC) + timedelta(seconds=i * 31))
        assert attempt is not None and reason is None
    assert ledger.reserve(80, "{}", datetime(2026, 10, 5, tzinfo=UTC))[1] == "experiment_request_limit"
    with pytest.raises(ValueError, match="immutable"):
        _BatchLedger("synthetic-secret", tmp_path / "ledger", "registration", "different-plan")
    with ledger._connect() as c:
        for i in range(20):
            c.execute("INSERT INTO jev_attempts(requested_at,fingerprint,request_json) VALUES(0,?, '{}')", (str(i),))
    other = _BatchLedger("synthetic-secret", tmp_path / "ledger", "other-registration", "other-plan")
    assert other.reserve(0, "{}", datetime(2026, 10, 5, tzinfo=UTC))[1] == "global_request_limit"


def test_provider_over_limit_usage_is_preserved_and_aborts(tmp_path, monkeypatch):
    import sqlite3

    import tradecopilot.forecast.jev_rl as module
    plan, _ = synthetic(tmp_path)
    def transport(payload, key):
        raw = reply(payload)
        raw["usage"]["input_tokens"] = 70_000
        return raw
    monkeypatch.setattr(module, "_post", transport)
    with pytest.raises(ValueError, match="stopped"):
        module.collect_cache(plan, tmp_path / "cache", api_key="synthetic-key", ledger_path=tmp_path / "ledger")
    with sqlite3.connect(tmp_path / "ledger") as c:
        assert c.execute("SELECT input_tokens FROM jev_attempts").fetchone()[0] == 70_000


def test_symlink_private_artifact_is_rejected(tmp_path):
    import tradecopilot.forecast.jev_rl as module
    plan, _ = synthetic(tmp_path)
    target = tmp_path / "requests" / "payload-000.json"
    original = target.read_bytes()
    target.unlink()
    alternate = tmp_path / "alternate.json"
    alternate.write_bytes(original)
    target.symlink_to(alternate)
    with pytest.raises(ValueError, match=r"artifact|symlink"):
        module._load_plan(plan)


def test_experiment_cost_cap_across_run_ids_and_immutable_limits(tmp_path):
    from tradecopilot.forecast.jev_rl import _BatchLedger
    ledger = _BatchLedger("synthetic-key", tmp_path / "ledger", "reg", "plan")
    attempt, _, _ = ledger.reserve(0, "{}", datetime(2026, 10, 4, tzinfo=UTC))
    assert attempt is not None
    ledger.settle(attempt, 5_900_000, {"status": "synthetic-cost-cap-fixture"})
    assert ledger.reserve(10, "{}", datetime(2026, 10, 5, tzinfo=UTC))[1] == "experiment_cost_limit"
    with ledger._connect() as c:
        c.execute("UPDATE forecast_run_limits SET max_requests=9 WHERE run_id='jev-rl:reg:1'")
    with pytest.raises(ValueError, match="immutable"):
        _BatchLedger("synthetic-key", tmp_path / "ledger", "reg", "plan")


def test_payload_edit_during_collection_cannot_change_frozen_later_request(tmp_path, monkeypatch):
    import tradecopilot.forecast.jev_rl as module
    plan, value = synthetic(tmp_path)
    original_second = json.loads((plan.parent / value["batches"][1]["payload_path"]).read_text())
    calls = []
    def transport(payload, key):
        calls.append(payload)
        if len(calls) == 1:
            target = plan.parent / value["batches"][1]["payload_path"]
            altered = json.loads(target.read_text())
            altered["state"]["private_unregistered_future"] = 999
            target.write_text(json.dumps(altered))
        return reply(payload)
    monkeypatch.setattr(module, "_post", transport)
    monkeypatch.setattr(module, "_wait_cooldown", lambda seconds: None)
    times = iter(datetime(2026, 10, 4, tzinfo=UTC) + timedelta(seconds=i * 31) for i in range(30))
    monkeypatch.setattr(module, "_now", lambda: next(times))
    with pytest.raises(ValueError):  # Complete publication fails against the edited artifact inventory.
        module.collect_cache(plan, tmp_path / "cache", api_key="synthetic-key", ledger_path=tmp_path / "ledger")
    assert calls[1] == original_second


def amendment(tmp_path, plan_path, attempted=2):
    plan = json.loads(plan_path.read_text())
    value = {"schema_version": "jev-rl-acquisition-amendment-v1", "registration_id": plan["registration_id"],
             "plan_id": plan["plan_id"], "existing_attempted_batch_indices": list(range(attempted)),
             "remaining_unattempted_batch_indices": list(range(attempted, len(plan["batches"]))),
             "retry_provider_failures": False, "failure_policy": "preserve_unavailable_and_continue_unattempted",
             "api_budget": plan["registration"]["api_budget"], "fixture": True}
    value["amendment_id"] = content_hash(value)
    path = tmp_path / "amendment.json"
    path.write_text(json.dumps(value))
    return path


def interrupted(tmp_path, monkeypatch):
    import tradecopilot.forecast.jev_rl as module
    plan, _ = synthetic(tmp_path)
    calls = []
    def transport(payload, key):
        calls.append(payload)
        raw = reply(payload)
        if len(calls) == 2:
            raw["answers"] = {}
        return raw
    monkeypatch.setattr(module, "_post", transport)
    times = iter(datetime(2026, 10, 4, tzinfo=UTC) + timedelta(seconds=i * 31) for i in range(100))
    monkeypatch.setattr(module, "_now", lambda: next(times))
    monkeypatch.setattr(module, "_wait_cooldown", lambda seconds: None)
    out, ledger = tmp_path / "cache", tmp_path / "ledger"
    with pytest.raises(ValueError, match="stopped"):
        module.collect_cache(plan, out, api_key="synthetic-key", ledger_path=ledger)
    return plan, out, ledger, calls


def test_explicit_continuation_preserves_failed_forecasts_and_never_reposts(tmp_path, monkeypatch):
    import tradecopilot.forecast.jev_rl as module
    plan, out, ledger, calls = interrupted(tmp_path, monkeypatch)
    prior = {p.name: p.read_bytes() for p in out.glob("response-*.json")}
    original_partial = (out / "partial.json").read_bytes()
    ap = amendment(tmp_path, plan)
    def future_failure(payload, key):
        calls.append(payload)
        raise RuntimeError("private-provider-secret")
    monkeypatch.setattr(module, "_post", future_failure)
    path = module.continue_cache(plan, out, ap, api_key="synthetic-key", ledger_path=ledger)
    cache = module.load_cache(path)
    assert len(calls) == 3 and len(cache["records"]) == 45
    assert cache["batch_counts"] == {"valid": 1, "unavailable": 2}
    assert cache["usage"]["attempts"] == 3 and cache["usage"]["input_tokens"] == 123 * 2 + 64000
    assert (out / "partial-before-continuation.json").read_bytes() == original_partial
    for filename, blob in prior.items():
        assert (out / filename).read_bytes() == blob
    failed = cache["records"][15:]
    assert all(r["status"] == "unavailable" and r["model_confidence"] == 0
               and r["probabilities"] == {"DOWN": 0, "FLAT": 0, "UP": 0} for r in failed)
    assert all(r["input_id"] and r["anchor_price"] and r["replay_available_at"] == r["as_of"] + 60 for r in failed)
    assert all(datetime.fromisoformat(r["generated_at"]).year == 2026 for r in failed)
    assert "private-provider-secret" not in path.read_text()
    with pytest.raises(ValueError):
        module.continue_cache(plan, out, ap, api_key="synthetic-key", ledger_path=ledger)
    assert len(calls) == 3


@pytest.mark.parametrize("corruption", ["response", "ledger", "amendment", "plan"])
def test_continuation_rejects_wrong_evidence_before_new_provider_call(tmp_path, monkeypatch, corruption):
    import sqlite3

    import tradecopilot.forecast.jev_rl as module
    plan, out, ledger, calls = interrupted(tmp_path, monkeypatch)
    ap = amendment(tmp_path, plan)
    if corruption == "response":
        rp = out / "response-000.json"
        value = json.loads(rp.read_text())
        value["input_tokens"] += 1
        rp.write_text(json.dumps(value))
    elif corruption == "ledger":
        with sqlite3.connect(ledger) as c:
            c.execute("UPDATE jev_attempts SET request_json='{}' WHERE id=1")
    elif corruption == "amendment":
        value = json.loads(ap.read_text())
        value["remaining_unattempted_batch_indices"] = [1, 2]
        value["amendment_id"] = content_hash({k: v for k, v in value.items() if k != "amendment_id"})
        ap.write_text(json.dumps(value))
    else:
        value = json.loads(plan.read_text())
        value["model"] = "jev-latest"
        plan.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        module.continue_cache(plan, out, ap, api_key="synthetic-key", ledger_path=ledger)
    assert len(calls) == 2


def test_continuation_global_cap_and_lock_prevent_new_calls(tmp_path, monkeypatch):
    import fcntl
    import sqlite3

    import tradecopilot.forecast.jev_rl as module
    plan, out, ledger, calls = interrupted(tmp_path, monkeypatch)
    ap = amendment(tmp_path, plan)
    lock = (out / ".continuation.lock").open("a+b")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="active"):
            module.continue_cache(plan, out, ap, api_key="synthetic-key", ledger_path=ledger)
    finally:
        lock.close()
    with sqlite3.connect(ledger) as c:
        for i in range(98):
            c.execute("INSERT INTO jev_attempts(requested_at,fingerprint,request_json) VALUES(0,?, '{}')", (str(i),))
    with pytest.raises(ValueError, match="budget"):
        module.continue_cache(plan, out, ap, api_key="synthetic-key", ledger_path=ledger)
    assert len(calls) == 2 and not (out / "cache.json").exists()


def test_continued_cache_rechecks_ledger_read_only_and_unavailable_seal(tmp_path, monkeypatch):
    import sqlite3

    import tradecopilot.forecast.jev_rl as module
    plan, out, ledger, calls = interrupted(tmp_path, monkeypatch)
    ap = amendment(tmp_path, plan)
    path = module.continue_cache(plan, out, ap, api_key="synthetic-key", ledger_path=ledger)
    assert len(calls) == 3
    # Loading must never initialize clients or migrate/mutate the ledger.
    monkeypatch.setattr(module, "_BatchLedger", lambda *args, **kwargs: pytest.fail("loader constructed client"))
    before = ledger.read_bytes()
    assert module.load_cache(path)["batch_counts"] == {"valid": 2, "unavailable": 1}
    assert ledger.read_bytes() == before
    with sqlite3.connect(ledger) as c:
        c.execute("UPDATE jev_attempts SET input_tokens=input_tokens+1 WHERE id=2")
    with pytest.raises(ValueError):
        module.load_cache(path)
    assert len(calls) == 3


def test_unknown_reserved_without_response_is_never_retried(tmp_path, monkeypatch):
    import sqlite3

    import tradecopilot.forecast.jev_rl as module
    plan, out, ledger, calls = interrupted(tmp_path, monkeypatch)
    ap = amendment(tmp_path, plan)
    with sqlite3.connect(ledger) as c:
        c.execute("UPDATE jev_attempts SET result_json=NULL WHERE id=2")
    with pytest.raises(ValueError):
        module.continue_cache(plan, out, ap, api_key="synthetic-key", ledger_path=ledger)
    assert len(calls) == 2
