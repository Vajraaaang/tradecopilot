"""Sealed same-as-of Jev batches for explicitly hypothetical retrospective RL research.

Preparation does no inference. Collection makes at most one attempt per frozen batch,
through the shared global ledger; partial failures never become a complete cache.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import sqlite3
import time
from collections import defaultdict
from contextlib import closing
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from tradecopilot.forecast.bars import HistoricalBar, load_bar_dataset
from tradecopilot.forecast.contracts import LABELS, content_hash
from tradecopilot.forecast.jev import JevForecaster
from tradecopilot.forecast.sessions import session_bounds
from tradecopilot.jev import INPUT_TOKEN_PRICE_USD, MAX_INPUT_TOKENS, MAX_REQUEST_BYTES, _post, default_ledger_path
from tradecopilot.rl.data import _dates, _source_check, load_prepared

MODEL = "jev-1.13.0"
PROMPT_VERSION = "masked-normalized-ohlcv-3h-v1"
AVAILABILITY_MODE = "hypothetical_as_of_plus_60_seconds"
_BUDGET = {"max_attempts": 80, "max_conservative_cost_usd": 0.25, "max_requests_per_run": 10,
           "max_cost_per_run_usd": 0.05, "global_limit": 100, "cooldown_seconds": 30,
           "initial_global_attempts": 15, "retry_provider_failures": False, "abort_on_invalid_response": True}
_NY = ZoneInfo("America/New_York")


def _now() -> datetime:
    return datetime.now(UTC)


def _wait_cooldown(seconds: float) -> None:
    time.sleep(seconds)


def _encoded(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _read(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_bytes())
    if not isinstance(result, dict):
        raise ValueError("JSON object required")
    return result


def _seal(value: dict[str, Any], key: str) -> None:
    if value.get(key) != content_hash({k: v for k, v in value.items() if k != key}):
        raise ValueError(f"invalid {key} seal")


def _write(path: Path, value: dict[str, Any]) -> None:
    with path.open("xb") as stream:
        stream.write(_encoded(value))
    path.chmod(0o600)


def _safe_path(path: Path, root: Path | None = None) -> Path:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("invalid private artifact path")
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("private artifact symlink rejected")
    resolved = path.resolve()
    if root is not None and not resolved.is_relative_to(root.resolve()):
        raise ValueError("private artifact outside bundle")
    return resolved


def _artifact(path: Path) -> dict[str, Any]:
    path = _safe_path(path.absolute())
    blob = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}


def _inventory(items: list[dict[str, Any]], root: Path) -> None:
    if not items or len({item["path"] for item in items}) != len(items):
        raise ValueError("invalid artifact inventory")
    for item in items:
        path = Path(item["path"])
        if _artifact(_safe_path(path, root)) != item:
            raise ValueError("private artifact integrity mismatch")


def _registration(reg: dict[str, Any]) -> list[date]:
    _seal(reg, "registration_id")
    jev = reg["jev"]
    if (reg["api_budget"] != _BUDGET or jev["model"] != MODEL or jev["prompt_version"] != PROMPT_VERSION
            or jev["horizons"] != ["15m", "60m", "close"] or jev["as_of_minutes_after_open"] != 61
            or jev["simulated_latency_seconds"] != 60 or jev["flat_threshold_bps"] != 10
            or jev["abstention_threshold"] != 0.6):
        raise ValueError("unsupported immutable registration")
    dates = _dates(reg)
    days = sorted(set().union(*dates.values()))
    if not reg.get("fixture") and [len(dates[role]) for role in ("train", "tune", "test")] != [60, 10, 10]:
        raise ValueError("registration requires 60/10/10 sessions")
    return days


def _stock_context(rows: list[HistoricalBar], asof: datetime) -> tuple[dict[str, Any], list[HistoricalBar]]:
    eligible = sorted((b for b in rows if b.end_time <= asof and b.available_at <= asof),
                      key=lambda b: b.start_time)
    if len(eligible) != 61 or eligible[-1].end_time != asof:
        raise ValueError("complete causal first 61 minutes required")
    if any(b.start_time != eligible[0].start_time + timedelta(minutes=i) for i, b in enumerate(eligible)):
        raise ValueError("missing causal bar minute")
    anchor = eligible[-1].close
    volume_mean = sum(b.volume for b in eligible) / Decimal(len(eligible))
    def bps(value: Decimal) -> float:
        return round(float(Decimal(10000) * (value / anchor - 1)), 3)
    recent = [[int((b.end_time - asof).total_seconds() / 60), bps(b.opening), bps(b.high), bps(b.low),
               bps(b.close), round(float(b.volume / volume_mean), 4) if volume_mean else 0.0]
              for b in eligible[-15:]]
    returns = {f"return_{minutes}m_bps": round(float(Decimal(10000)
               * (anchor / eligible[-1 - minutes].close - 1)), 3) for minutes in (1, 5, 15, 30, 60)}
    context = {"summary": returns | {"open_to_anchor_bps": round(float(Decimal(10000)
               * (anchor / eligible[0].opening - 1)), 3), "session_high_bps": bps(max(b.high for b in eligible)),
               "session_low_bps": bps(min(b.low for b in eligible)),
               "volume_last15_vs_first46": round(float((sum((b.volume for b in eligible[-15:]), Decimal(0)) / 15)
               / (sum((b.volume for b in eligible[:-15]), Decimal(0)) / 46)), 4)
               if sum(b.volume for b in eligible[:-15]) else 0.0}, "recent_bars": recent}
    return context, eligible


def prepare_requests(bars_dir: Path, registration_path: Path, prepared_dir: Path, output_dir: Path) -> Path:
    """Freeze all masked payloads and private source mappings before any paid attempt."""
    if output_dir.exists():
        raise ValueError("request output is immutable")
    bundle_root = registration_path.absolute().parent
    for original_path in (registration_path, bars_dir, prepared_dir, output_dir):
        _safe_path(original_path.absolute(), bundle_root)
    reg = _read(registration_path)
    days = _registration(reg)
    _, prepared = load_prepared(prepared_dir, "train")
    # Loading train verifies all arrays and their inventory without releasing TEST rewards.
    if prepared["registration_id"] != reg["registration_id"]:
        raise ValueError("prepared registration mismatch")
    bars, source = load_bar_dataset(bars_dir)
    _source_check(bars, source, reg)
    if source["data_id"] != prepared["source_data_id"]:
        raise ValueError("prepared source mismatch")
    groups: dict[tuple[str, date], list[HistoricalBar]] = defaultdict(list)
    for bar in bars:
        groups[bar.symbol, bar.start_time.astimezone(_NY).date()].append(bar)
    output_dir.mkdir(parents=True, mode=0o700)
    batches: list[dict[str, Any]] = []
    for index, day in enumerate(days):
        bounds = session_bounds(day)
        assert bounds is not None
        asof = bounds[0] + timedelta(minutes=61)
        state: dict[str, Any] = {"minutes_after_open": 61, "minutes_until_close": int((bounds[1]-asof)
                                .total_seconds() / 60), "stocks": {},
                                "recent_bar_columns": ["minutes_relative_to_asof", "open_bps", "high_bps",
                                                       "low_bps", "close_bps", "volume_vs_past_mean"]}
        questions: dict[str, Any] = {}
        inputs = []
        for masked_index, symbol in enumerate(reg["symbols"]):
            masked = f"S{masked_index}"
            context, observed = _stock_context(groups[symbol, day], asof)
            state["stocks"][masked] = context
            for horizon in ("15m", "60m", "close"):
                minutes = int(horizon[:-1]) if horizon != "close" else state["minutes_until_close"]
                target = asof + timedelta(minutes=minutes)
                key = f"{masked}_{horizon}"
                questions[key] = {"type": "choice", "instructions": {"question_key": key,
                    "stock_key": masked, "target_minutes_after_asof": minutes,
                    "task": "Forecast the minute-bar close proxy at target minute-end relative to as-of "
                    f"completed bar close, {minutes} minutes after as-of. Use only state.stocks.{masked} "
                    "at this shared as-of; normalized bps anchor is zero. Return DOWN/FLAT/UP probabilities. "
                    "No invented observations/news."},
                    "criteria": {"DOWN": "Target minute-bar close proxy is below as-of completed bar close "
                                 "by more than 10 bps.",
                                 "FLAT": "Target minute-bar close proxy is within +/-10 bps of as-of "
                                 "completed bar close inclusive.",
                                 "UP": "Target minute-bar close proxy is above as-of completed bar close "
                                 "by more than 10 bps."}}
                record = {"question_key": key, "symbol": symbol, "session_date": day.isoformat(),
                          "as_of": int(asof.timestamp()), "target_time": int(target.timestamp()),
                          "horizon_key": horizon, "anchor_price": str(observed[-1].close),
                          "replay_available_at": int(asof.timestamp()) + 60,
                          "source_bar_ids": [b.bar_id for b in observed], "config_id": prepared["config_id"]}
                record["input_id"] = content_hash(record)
                inputs.append(record)
        payload = {"model": MODEL, "state": state, "questions": questions}
        encoded = _encoded(payload)
        if len(encoded) > MAX_REQUEST_BYTES:
            raise ValueError("request exceeds 16000 byte cap")
        filename = f"payload-{index:03d}.json"
        _write(output_dir / filename, payload)
        batches.append({"batch_index": index, "run_id": f"jev-rl:{reg['registration_id']}:{index // 10}",
                        "payload_path": filename, "payload_id": content_hash(payload), "inputs": inputs})
    original = [registration_path, bars_dir / "manifest.json", bars_dir / "bars.jsonl",
                prepared_dir / "manifest.json", prepared_dir / "episodes.npz"]
    plan: dict[str, Any] = {"schema_version": "jev-rl-request-plan-v1", "bundle_root": str(bundle_root.resolve()),
        "registration": reg,
        "registration_id": reg["registration_id"], "source_data_id": source["data_id"],
        "base_prepared_data_id": prepared["data_id"], "model": MODEL, "prompt_version": PROMPT_VERSION,
        "availability_mode": AVAILABILITY_MODE, "batches": batches,
        "inventory": [_artifact(p) for p in original] + [_artifact(output_dir / b["payload_path"]) for b in batches]}
    plan["plan_id"] = content_hash(plan)
    path = output_dir / "request-plan.json"
    _write(path, plan)
    _load_plan(path)
    return path


def _load_plan(path: Path) -> dict[str, Any]:
    plan = _read(path)
    _seal(plan, "plan_id")
    root = _safe_path(Path(plan["bundle_root"]))
    _safe_path(path.absolute(), root)
    _inventory(plan["inventory"], root)
    days = _registration(plan["registration"])
    if (plan["schema_version"] != "jev-rl-request-plan-v1" or plan["model"] != MODEL
            or plan["prompt_version"] != PROMPT_VERSION or plan["availability_mode"] != AVAILABILITY_MODE
            or plan["registration_id"] != plan["registration"]["registration_id"]
            or len(plan["batches"]) != len(days)):
        raise ValueError("invalid request plan")
    for i, batch in enumerate(plan["batches"]):
        if (batch["batch_index"] != i or batch["payload_path"] != f"payload-{i:03d}.json"
                or batch["run_id"] != f"jev-rl:{plan['registration_id']}:{i // 10}"):
            raise ValueError("invalid batch mapping")
        payload = _read(path.parent / batch["payload_path"])
        if content_hash(payload) != batch["payload_id"] or len(_encoded(payload)) > MAX_REQUEST_BYTES:
            raise ValueError("invalid payload seal")
        bounds = session_bounds(days[i])
        assert bounds is not None
        asof = int(bounds[0].timestamp()) + 61 * 60
        expected = {f"S{s}_{h}" for s in range(5) for h in ("15m", "60m", "close")}
        if set(payload["questions"]) != expected or {r["question_key"] for r in batch["inputs"]} != expected:
            raise ValueError("invalid batch questions")
        for record in batch["inputs"]:
            _seal(record, "input_id")
            key = record["question_key"]
            masked, horizon = key.split("_")
            target = int(bounds[1].timestamp()) if horizon == "close" else asof + int(horizon[:-1]) * 60
            if (record["symbol"] != plan["registration"]["symbols"][int(masked[1:])]
                    or record["session_date"] != days[i].isoformat() or record["as_of"] != asof
                    or record["target_time"] != target or record["horizon_key"] != horizon
                    or record["replay_available_at"] != asof + 60 or len(record["source_bar_ids"]) != 61):
                raise ValueError("invalid input timing")
    return plan


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _reported_tokens(raw: dict[str, Any]) -> int | None:
    value = raw.get("usage", {})
    tokens = value.get("input_tokens") if isinstance(value, dict) else None
    return tokens if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0 else None


def _validate_answers(raw: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    if raw.get("model") != MODEL or not isinstance(raw.get("answers"), dict) or set(raw["answers"]) != keys:
        raise ValueError("invalid model or answer keys")
    result = {}
    for key in sorted(keys):
        a = raw["answers"][key]
        if not isinstance(a, dict) or set(a) != {"type", "choice", "probabilities", "confidence"}:
            raise ValueError("invalid answer fields")
        p, confidence, choice = a["probabilities"], a["confidence"], a["choice"]
        if (a["type"] != "choice" or not isinstance(p, dict) or set(p) != set(LABELS) or choice not in LABELS
                or not all(_finite(v) and 0 <= v <= 1 for v in p.values())
                or not math.isclose(sum(p.values()), 1, rel_tol=0, abs_tol=1e-6)
                or p[choice] != max(p.values()) or not _finite(confidence) or not 0 <= confidence <= 1):
            raise ValueError("invalid probability distribution")
        result[key] = {"type": "choice", "choice": choice, "probabilities": p, "confidence": confidence}
    return result


class _BatchLedger(JevForecaster):
    """Experiment and existing per-run/global reservations share one IMMEDIATE transaction."""
    def __init__(self, api_key: str, path: Path | None, registration_id: str, plan_id: str) -> None:
        self._experiment = registration_id
        self._plan_id = plan_id
        super().__init__(api_key, path, run_id=f"jev-rl:{registration_id}:0")
        with closing(self._connect()) as c:
            c.execute("BEGIN IMMEDIATE")
            try:
                c.execute("CREATE TABLE IF NOT EXISTS jev_rl_experiments ("
                          "experiment_id TEXT PRIMARY KEY,plan_id TEXT NOT NULL,max_attempts INTEGER NOT NULL,"
                          "max_cost_usd REAL NOT NULL)")
                c.execute("CREATE TABLE IF NOT EXISTS jev_rl_batches (experiment_id TEXT NOT NULL,"
                          "batch_index INTEGER NOT NULL,attempt_id INTEGER NOT NULL UNIQUE,"
                          "PRIMARY KEY(experiment_id,batch_index))")
                limits = c.execute("SELECT plan_id,max_attempts,max_cost_usd FROM jev_rl_experiments "
                                   "WHERE experiment_id=?", (registration_id,)).fetchone()
                if limits is not None and limits != (plan_id, 80, 0.25):
                    raise ValueError("experiment limits and request plan are immutable")
                c.execute("INSERT OR IGNORE INTO jev_rl_experiments VALUES(?,?,80,0.25)", (registration_id, plan_id))
                for group in range(8):
                    run_id = f"jev-rl:{registration_id}:{group}"
                    run = c.execute("SELECT max_requests,max_cost_usd FROM forecast_run_limits WHERE run_id=?",
                                    (run_id,)).fetchone()
                    if run is not None and run != (10, 0.05):
                        raise ValueError("forecast run limits are immutable")
                    c.execute("INSERT OR IGNORE INTO forecast_run_limits VALUES(?,10,0.05)", (run_id,))
                c.commit()
            finally:
                if c.in_transaction:
                    c.rollback()

    def reserve(self, index: int, request: str, now: datetime) -> tuple[int | None, str | None, float]:
        with closing(self._connect()) as c:
            c.execute("BEGIN IMMEDIATE")
            try:
                global_count, last = c.execute("SELECT COUNT(*),MAX(requested_at) FROM jev_attempts").fetchone()
                if global_count >= 100:
                    return None, "global_request_limit", 0
                count, tokens = c.execute("SELECT COUNT(*),COALESCE(SUM(COALESCE(a.input_tokens,64000)),0) "
                    "FROM jev_rl_batches b JOIN jev_attempts a ON b.attempt_id=a.id WHERE b.experiment_id=?",
                    (self._experiment,)).fetchone()
                if count >= 80 or not 0 <= index < 80:
                    return None, "experiment_request_limit", 0
                if Decimal(tokens + 64000) * Decimal(str(INPUT_TOKEN_PRICE_USD)) > Decimal("0.25"):
                    return None, "experiment_cost_limit", 0
                if c.execute("SELECT 1 FROM jev_rl_batches WHERE experiment_id=? AND batch_index=?",
                             (self._experiment, index)).fetchone():
                    return None, "batch_already_reserved_no_retry", 0
                run_id = f"jev-rl:{self._experiment}:{index // 10}"
                run_count, run_tokens = c.execute("SELECT COUNT(*),COALESCE(SUM(COALESCE(input_tokens,64000)),0) "
                                                 "FROM jev_attempts WHERE forecast_run_id=?", (run_id,)).fetchone()
                if run_count >= 10:
                    return None, "run_request_limit", 0
                if Decimal(run_tokens + 64000) * Decimal(str(INPUT_TOKEN_PRICE_USD)) > Decimal("0.05"):
                    return None, "run_cost_limit", 0
                delay = 30 - (now.timestamp() - last) if last is not None else 0
                if delay > 0:
                    return None, "cooldown", delay
                cursor = c.execute("INSERT INTO jev_attempts(requested_at,fingerprint,request_json,forecast_run_id) "
                    "VALUES(?,?,?,?)", (now.timestamp(), content_hash({"plan_id": self._plan_id, "batch": index}),
                                        request, run_id))
                attempt = cursor.lastrowid
                assert attempt is not None
                c.execute("INSERT INTO jev_rl_batches VALUES(?,?,?)", (self._experiment, index, attempt))
                c.commit()
                return attempt, None, 0
            finally:
                if c.in_transaction:
                    c.rollback()

    def settle(self, attempt: int, tokens: int, evidence: dict[str, Any]) -> None:
        with closing(self._connect()) as c:
            c.execute("UPDATE jev_attempts SET input_tokens=?,result_json=? WHERE id=?",
                      (tokens, _encoded(evidence).decode(), attempt))


def _records(batch: dict[str, Any], response: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for item in batch["inputs"]:
        unavailable = response["answers"] is None
        answer: dict[str, Any] = ({"probabilities": {label: 0.0 for label in LABELS}, "confidence": 0.0}
                  if unavailable else response["answers"][item["question_key"]])
        record = {k: item[k] for k in ("symbol", "session_date", "as_of", "target_time", "horizon_key",
                                       "anchor_price", "replay_available_at", "input_id")}
        record.update(generated_at=response["completed_at"], probabilities=answer["probabilities"],
                      model_confidence=float(answer["confidence"]),
                      status="unavailable" if unavailable else
                      "ok" if min(max(answer["probabilities"].values()), answer["confidence"]) >= 0.6 else "abstained",
                      request_id=response["request_id"], payload_id=batch["payload_id"])
        result.append(record)
    return result


def collect_cache(plan_path: Path, output_dir: Path, *, api_key: str, ledger_path: Path | None = None) -> Path:
    """Execute a sealed plan once. Provider failure stops with private, sanitized partial evidence."""
    plan = _load_plan(plan_path)  # Must precede even ledger/credential client initialization.
    _safe_path(output_dir.absolute(), Path(plan["bundle_root"]))
    if output_dir.exists():
        raise ValueError("collection output is immutable; no retry")
    payloads = [_read(plan_path.parent / batch["payload_path"]) for batch in plan["batches"]]
    if any(content_hash(payload) != batch["payload_id"]
           for batch, payload in zip(plan["batches"], payloads, strict=True)):
        raise ValueError("payload changed after plan validation")
    ledger = _BatchLedger(api_key, ledger_path, plan["registration_id"], plan["plan_id"])
    with closing(ledger._connect()) as c:
        count = c.execute("SELECT COUNT(*) FROM jev_attempts").fetchone()[0]
        previous = c.execute("SELECT COUNT(*) FROM jev_rl_batches WHERE experiment_id=?",
                             (plan["registration_id"],)).fetchone()[0]
    if previous or (not plan["registration"].get("fixture") and count < 15):
        raise ValueError("existing attempts or missing initial shared ledger; no retry")
    output_dir.mkdir(parents=True, mode=0o700)
    records = []
    total_tokens = 0
    attempted = 0
    responses = []
    for batch, payload in zip(plan["batches"], payloads, strict=True):
        while True:
            attempt, reason, delay = ledger.reserve(batch["batch_index"], _encoded(payload).decode(), _now())
            if reason != "cooldown":
                break
            _wait_cooldown(delay)  # No paid attempt happened; safe to wait and recheck atomically.
        if reason is not None:
            _write(output_dir / "stopped.json", {"reason": reason, "attempts": attempted})
            raise ValueError("Jev collection stopped by durable budget or duplicate guard")
        assert attempt is not None
        attempted += 1
        requested_at = _now().isoformat()
        tokens = MAX_INPUT_TOKENS
        answers = None
        failure = "provider_unavailable"
        try:
            raw = _post(payload, api_key)
            reported = _reported_tokens(raw)
            tokens = MAX_INPUT_TOKENS if reported is None else reported
            failure = "invalid_response"
            if reported is None or reported > MAX_INPUT_TOKENS:
                raise ValueError("invalid provider usage")
            answers = _validate_answers(raw, set(payload["questions"]))
        except Exception:
            # Never persist provider error bodies, auth headers, or arbitrary returned text.
            pass
        completed_at = _now().isoformat()
        evidence = {"model": MODEL, "request_id": str(attempt), "payload_id": batch["payload_id"],
                    "requested_at": requested_at, "completed_at": completed_at, "input_tokens": tokens,
                    "status": "validated" if answers is not None else failure, "answers": answers}
        ledger.settle(attempt, tokens, evidence)
        response_path = output_dir / f"response-{batch['batch_index']:03d}.json"
        _write(response_path, evidence)
        responses.append(_artifact(response_path))
        total_tokens += tokens
        if answers is not None:
            records.extend(_records(batch, evidence))
        partial = {"schema_version": "jev-rl-partial-v1", "plan_id": plan["plan_id"], "attempts": attempted,
                   "input_tokens": total_tokens,
                   "conservative_estimated_cost_usd": total_tokens * INPUT_TOKEN_PRICE_USD,
                   "records": records, "responses": responses, "status": evidence["status"]}
        temporary = output_dir / ".partial.tmp"
        _write(temporary, partial)
        os.replace(temporary, output_dir / "partial.json")
        if answers is None:
            raise ValueError("Jev collection stopped on invalid response or provider failure; no retry")
    cache = {"schema_version": "jev-rl-cache-v1", "registration_id": plan["registration_id"],
             "source_data_id": plan["source_data_id"], "base_prepared_data_id": plan["base_prepared_data_id"],
             "model": MODEL, "prompt_version": PROMPT_VERSION, "availability_mode": AVAILABILITY_MODE,
             "records": records, "usage": {"attempts": attempted, "input_tokens": total_tokens,
             "conservative_estimated_cost_usd": total_tokens * INPUT_TOKEN_PRICE_USD},
             "plan_path": str(plan_path.resolve()), "inventory": [*plan["inventory"], _artifact(plan_path),
                                                                   *responses, _artifact(output_dir / "partial.json")]}
    cache["cache_id"] = content_hash(cache)
    path = output_dir / "cache.json"
    temporary_cache = output_dir / ".cache.tmp"
    _write(temporary_cache, cache)
    load_cache(temporary_cache)
    os.replace(temporary_cache, path)
    return path


def _response(response: dict[str, Any], batch: dict[str, Any], *, allow_failure: bool) -> None:
    if set(response) != {"model", "request_id", "payload_id", "requested_at", "completed_at",
                         "input_tokens", "status", "answers"}:
        raise ValueError("invalid private response fields")
    tokens = response["input_tokens"]
    if (response["model"] != MODEL or response["payload_id"] != batch["payload_id"]
            or not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0
            or not isinstance(response["request_id"], str) or not response["request_id"].isdigit()):
        raise ValueError("invalid private response")
    requested, completed = (datetime.fromisoformat(response[k]) for k in ("requested_at", "completed_at"))
    if requested.utcoffset() != timedelta(0) or completed.utcoffset() != timedelta(0) or completed < requested:
        raise ValueError("invalid actual generation time")
    if response["status"] == "validated":
        if tokens > MAX_INPUT_TOKENS:
            raise ValueError("invalid validated usage")
        _validate_answers(response, {r["question_key"] for r in batch["inputs"]})
    elif (not allow_failure or response["status"] not in {"invalid_response", "provider_unavailable"}
          or response["answers"] is not None):
        raise ValueError("invalid failed response")


def _amendment(path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    _safe_path(path.absolute(), Path(plan["bundle_root"]))
    amendment = _read(path)
    _seal(amendment, "amendment_id")
    attempted = amendment["existing_attempted_batch_indices"]
    remaining = amendment["remaining_unattempted_batch_indices"]
    total = len(plan["batches"])
    if (amendment["schema_version"] != "jev-rl-acquisition-amendment-v1"
            or amendment["registration_id"] != plan["registration_id"] or amendment["plan_id"] != plan["plan_id"]
            or amendment["retry_provider_failures"] is not False
            or amendment["failure_policy"] != "preserve_unavailable_and_continue_unattempted"
            or amendment["api_budget"] != _BUDGET or not isinstance(attempted, list) or not attempted
            or any(not isinstance(i, int) or isinstance(i, bool) for i in attempted)
            or attempted != list(range(len(attempted)))
            or remaining != list(range(len(attempted), total))):
        raise ValueError("invalid acquisition amendment")
    return amendment


def _existing_responses(plan: dict[str, Any], output_dir: Path, ledger_path: Path
                        ) -> list[dict[str, Any]]:
    """Reconcile saved evidence exactly against durable, previously reserved provider attempts."""
    if not plan["registration"].get("fixture") and ledger_path.resolve() != default_ledger_path():
        raise ValueError("continued acquisition requires the existing shared default ledger")
    if not ledger_path.exists():
        raise ValueError("durable acquisition ledger missing")
    with closing(sqlite3.connect(f"{ledger_path.resolve().as_uri()}?mode=ro", uri=True)) as c:
        experiment = c.execute("SELECT plan_id,max_attempts,max_cost_usd FROM jev_rl_experiments "
                               "WHERE experiment_id=?", (plan["registration_id"],)).fetchone()
        if experiment != (plan["plan_id"], 80, 0.25):
            raise ValueError("durable experiment mismatch")
        rows = c.execute("SELECT b.batch_index,a.id,a.requested_at,a.fingerprint,a.request_json,"
                         "a.forecast_run_id,a.input_tokens,a.result_json FROM jev_rl_batches b "
                         "JOIN jev_attempts a ON b.attempt_id=a.id WHERE b.experiment_id=? ORDER BY b.batch_index",
                         (plan["registration_id"],)).fetchall()
        if [row[0] for row in rows] != list(range(len(rows))) or len(rows) > len(plan["batches"]):
            raise ValueError("durable attempted batch coverage mismatch")
        for group in range(8):
            limits = c.execute("SELECT max_requests,max_cost_usd FROM forecast_run_limits WHERE run_id=?",
                               (f"jev-rl:{plan['registration_id']}:{group}",)).fetchone()
            if limits != (10, 0.05):
                raise ValueError("durable immutable run limits mismatch")
    responses = []
    for index, attempt, reserved, fingerprint, request, run_id, tokens, result in rows:
        batch = plan["batches"][index]
        response_path = output_dir / f"response-{index:03d}.json"
        _safe_path(response_path.absolute(), Path(plan["bundle_root"]))
        response = _read(response_path)
        _response(response, batch, allow_failure=True)
        # Payload IDs hash canonical JSON and ledger requests must themselves reproduce that payload.
        try:
            request_payload = json.loads(request)
        except (ValueError, TypeError) as error:
            raise ValueError("invalid durable request") from error
        if (str(attempt) != response["request_id"] or tokens != response["input_tokens"]
                or result is None or json.loads(result) != response or run_id != batch["run_id"]
                or fingerprint != content_hash({"plan_id": plan["plan_id"], "batch": index})
                or content_hash(request_payload) != batch["payload_id"] or request != _encoded(request_payload).decode()
                or reserved > datetime.fromisoformat(response["requested_at"]).timestamp()):
            raise ValueError("saved response differs from durable attempt")
        responses.append(response)
    expected_paths = {f"response-{i:03d}.json" for i in range(len(rows))}
    if {p.name for p in output_dir.glob("response-*.json")} != expected_paths:
        raise ValueError("saved response and durable attempt coverage mismatch")
    return responses


def _attempt_batch(ledger: _BatchLedger, batch: dict[str, Any], payload: dict[str, Any], api_key: str
                   ) -> dict[str, Any]:
    while True:
        attempt, reason, delay = ledger.reserve(batch["batch_index"], _encoded(payload).decode(), _now())
        if reason != "cooldown":
            break
        _wait_cooldown(delay)
    if reason is not None:
        raise ValueError("Jev continuation stopped by durable budget or duplicate guard")
    assert attempt is not None
    requested = _now().isoformat()
    tokens = MAX_INPUT_TOKENS
    answers = None
    status = "provider_unavailable"
    try:
        raw = _post(payload, api_key)
        reported = _reported_tokens(raw)
        tokens = MAX_INPUT_TOKENS if reported is None else reported
        status = "invalid_response"
        if reported is None or reported > MAX_INPUT_TOKENS:
            raise ValueError("invalid usage")
        answers = _validate_answers(raw, set(payload["questions"]))
    except Exception:
        pass  # Never retain arbitrary provider text or exception messages.
    response = {"model": MODEL, "request_id": str(attempt), "payload_id": batch["payload_id"],
                "requested_at": requested, "completed_at": _now().isoformat(), "input_tokens": tokens,
                "status": "validated" if answers is not None else status, "answers": answers}
    ledger.settle(attempt, tokens, response)
    return response


def _continued_partial(plan: dict[str, Any], output_dir: Path, responses: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = sum(response["input_tokens"] for response in responses)
    value = {"schema_version": "jev-rl-partial-v1", "plan_id": plan["plan_id"], "attempts": len(responses),
             "input_tokens": tokens, "conservative_estimated_cost_usd": tokens * INPUT_TOKEN_PRICE_USD,
             "records": [record for batch, response in zip(plan["batches"], responses, strict=False)
                         for record in _records(batch, response)],
             "responses": [_artifact(output_dir / f"response-{i:03d}.json") for i in range(len(responses))],
             "status": responses[-1]["status"]}
    temporary = output_dir / ".partial.tmp"
    _write(temporary, value)
    os.replace(temporary, output_dir / "partial.json")
    return value


def continue_cache(plan_path: Path, existing_output_dir: Path, amendment_path: Path, *, api_key: str,
                   ledger_path: Path | None = None) -> Path:
    """Explicitly continue only unattempted batches; failed cases remain unavailable, never retried."""
    plan = _load_plan(plan_path)
    amendment = _amendment(amendment_path, plan)
    output_dir = _safe_path(existing_output_dir.absolute(), Path(plan["bundle_root"]))
    if not output_dir.is_dir() or (output_dir / "cache.json").exists():
        raise ValueError("continuation requires existing incomplete collection")
    lock_path = _safe_path(output_dir / ".continuation.lock", Path(plan["bundle_root"]))
    with lock_path.open("a+b") as lock:
        lock_path.chmod(0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another continuation is active") from error
        try:
            return _continue_locked(plan_path, plan, output_dir, amendment_path, amendment, api_key,
                                    (ledger_path or default_ledger_path()).resolve())
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _continue_locked(plan_path: Path, plan: dict[str, Any], output_dir: Path, amendment_path: Path,
                     amendment: dict[str, Any], api_key: str, ledger_path: Path) -> Path:
    if (output_dir / "cache.json").exists():
        raise ValueError("collection already sealed")
    responses = _existing_responses(plan, output_dir, ledger_path)
    if any(response["input_tokens"] > MAX_INPUT_TOKENS for response in responses):
        raise ValueError("provider usage exceeds reservation; further calls stopped")
    initially_attempted = len(amendment["existing_attempted_batch_indices"])
    if len(responses) < initially_attempted:
        raise ValueError("amendment original attempted coverage mismatch")
    payloads = [_read(plan_path.parent / batch["payload_path"]) for batch in plan["batches"]]
    if any(content_hash(payload) != batch["payload_id"]
           for batch, payload in zip(plan["batches"], payloads, strict=True)):
        raise ValueError("payload changed after plan validation")
    # Preserve the exact original partial, including its honest incomplete coverage.
    original = output_dir / "partial-before-continuation.json"
    if not original.exists():
        blob = (output_dir / "partial.json").read_bytes()
        with original.open("xb") as stream:
            stream.write(blob)
        original.chmod(0o600)
    else:
        _safe_path(original, Path(plan["bundle_root"]))
    ledger = _BatchLedger(api_key, ledger_path, plan["registration_id"], plan["plan_id"])
    _continued_partial(plan, output_dir, responses)
    for batch, payload in zip(plan["batches"][len(responses):], payloads[len(responses):], strict=True):
        response = _attempt_batch(ledger, batch, payload, api_key)
        _write(output_dir / f"response-{batch['batch_index']:03d}.json", response)
        responses.append(response)
        _continued_partial(plan, output_dir, responses)
        if response["input_tokens"] > MAX_INPUT_TOKENS:
            raise ValueError("continuation stopped: provider usage exceeds reservation")
    records = [record for batch, response in zip(plan["batches"], responses, strict=True)
               for record in _records(batch, response)]
    tokens = sum(response["input_tokens"] for response in responses)
    counts = {"valid": sum(r["status"] == "validated" for r in responses),
              "unavailable": sum(r["status"] != "validated" for r in responses)}
    cache = {"schema_version": "jev-rl-cache-v1", "registration_id": plan["registration_id"],
             "source_data_id": plan["source_data_id"], "base_prepared_data_id": plan["base_prepared_data_id"],
             "model": MODEL, "prompt_version": PROMPT_VERSION, "availability_mode": AVAILABILITY_MODE,
             "records": records, "usage": {"attempts": len(responses), "input_tokens": tokens,
             "conservative_estimated_cost_usd": tokens * INPUT_TOKEN_PRICE_USD},
             "plan_path": str(plan_path.resolve()), "amendment_path": str(amendment_path.resolve()),
             "acquisition_amendment": amendment, "amendment_id": amendment["amendment_id"],
             "ledger_path": str(ledger_path), "batch_counts": counts,
             "inventory": [*plan["inventory"], _artifact(plan_path), _artifact(amendment_path), _artifact(original),
                           *[_artifact(output_dir / f"response-{i:03d}.json") for i in range(len(responses))],
                           _artifact(output_dir / "partial.json")]}
    cache["cache_id"] = content_hash(cache)
    temporary = output_dir / ".cache.tmp"
    if temporary.exists():
        raise ValueError("existing unpublished cache requires review")
    _write(temporary, cache)
    load_cache(temporary)
    path = output_dir / "cache.json"
    os.replace(temporary, path)
    return path


def load_cache(path: Path) -> dict[str, Any]:
    """Verify original artifacts, every response, and amended acquisition's durable ledger evidence."""
    try:
        cache = _read(path)
        _seal(cache, "cache_id")
        plan = _load_plan(Path(cache["plan_path"]))
        _safe_path(path.absolute(), Path(plan["bundle_root"]))
        _inventory(cache["inventory"], Path(plan["bundle_root"]))
        amended = "acquisition_amendment" in cache
        amendment = _amendment(Path(cache["amendment_path"]), plan) if amended else None
        if amended:
            assert amendment is not None
            if (cache["acquisition_amendment"] != amendment or cache["amendment_id"] != amendment["amendment_id"]):
                raise ValueError("acquisition amendment snapshot mismatch")
        responses = (_existing_responses(plan, path.parent, Path(cache["ledger_path"])) if amended
                     else [_read(path.parent / f"response-{b['batch_index']:03d}.json") for b in plan["batches"]])
        if len(responses) != len(plan["batches"]):
            raise ValueError("incomplete response coverage")
        expected_records = []
        tokens = 0
        for batch, response in zip(plan["batches"], responses, strict=True):
            _response(response, batch, allow_failure=amended)
            tokens += response["input_tokens"]
            expected_records.extend(_records(batch, response))
        expected_usage = {"attempts": len(plan["batches"]), "input_tokens": tokens,
                          "conservative_estimated_cost_usd": tokens * INPUT_TOKEN_PRICE_USD}
        if (cache["schema_version"] != "jev-rl-cache-v1" or cache["records"] != expected_records
                or cache["usage"] != expected_usage or len({r["request_id"] for r in expected_records})
                != len(plan["batches"]) or any(cache[k] != plan[k] for k in ("registration_id", "source_data_id",
                "base_prepared_data_id", "model", "prompt_version", "availability_mode"))):
            raise ValueError("cache mapping or usage mismatch")
        required_paths = {item["path"] for item in plan["inventory"]} | {str(Path(cache["plan_path"]).resolve())}
        required_paths.update(str((path.parent / f"response-{b['batch_index']:03d}.json").resolve())
                              for b in plan["batches"])
        required_paths.add(str((path.parent / "partial.json").resolve()))
        if amended:
            counts = {"valid": sum(r["status"] == "validated" for r in responses),
                      "unavailable": sum(r["status"] != "validated" for r in responses)}
            if cache["batch_counts"] != counts:
                raise ValueError("invalid acquisition batch counts")
            required_paths.update({str(Path(cache["amendment_path"]).resolve()),
                                   str((path.parent / "partial-before-continuation.json").resolve())})
        if {item["path"] for item in cache["inventory"]} != required_paths:
            raise ValueError("incomplete private artifact inventory")
        return cache
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
        raise ValueError("invalid Jev cache or private artifact") from error
