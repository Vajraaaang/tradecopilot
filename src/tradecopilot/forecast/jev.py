"""Fixed-horizon Jev forecasts with the existing shared trial ledger and durable run caps."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Sequence
from contextlib import closing
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from tradecopilot.forecast.contracts import (
    FEATURE_NAMES,
    LABELS,
    ForecastConfig,
    ForecastExample,
    ForecastPrediction,
    content_hash,
    utc,
)
from tradecopilot.jev import (
    COOLDOWN_SECONDS,
    INPUT_TOKEN_PRICE_USD,
    JEV_MODEL,
    MAX_INPUT_TOKENS,
    MAX_REQUEST_BYTES,
    JevAdvisor,
)

FORECAST_PROMPT_VERSION = "forward-last-price-v1"
_WORST_COST = MAX_INPUT_TOKENS * INPUT_TOKEN_PRICE_USD
_FIXTURE_REASON = "offline_contract_fixture_not_model_performance"


def _input_problem(example: ForecastExample, config: ForecastConfig) -> str | None:
    if example.config_id != config.config_id:
        return "config_mismatch"
    if example.symbol not in config.symbols:
        return "symbol_outside_config"
    if example.target_time - example.as_of != timedelta(minutes=config.horizon_minutes):
        return "target_horizon_mismatch"
    if set(example.features) != set(FEATURE_NAMES) or not all(_finite(v) for v in example.features.values()):
        return "invalid_features"
    if not 0 <= example.features["source_age_seconds"] <= config.max_source_age_seconds:
        return "stale_source"
    return None


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _payload(example: ForecastExample, config: ForecastConfig) -> dict[str, Any]:
    band = f"{config.flat_threshold_bps:g}"
    return {
        "model": JEV_MODEL,
        "state": {
            "symbol": example.symbol,
            "as_of": example.as_of.isoformat(),
            "anchor_price": str(example.anchor_price),
            "target_time": example.target_time.isoformat(),
            "features": {name: example.features[name] for name in FEATURE_NAMES},
        },
        "questions": {
            "direction": {
                "type": "choice",
                "instructions": (
                    f"Forecast the stock's last price exactly {config.horizon_minutes} minutes after as_of, "
                    "at target_time, using only these causal observed features and anchor last price. "
                    "Return probabilities for DOWN, FLAT, and UP. Define return_bps as "
                    "10000 * (last price at target_time / anchor_price - 1). "
                    f"FLAT includes both boundaries of the +/-{band} basis point interval. "
                    "Do not invent observations, news, or outcomes. This is a fixed-horizon price forecast, "
                    "not a trading instruction or probability of profit."
                ),
                "criteria": {
                    "DOWN": f"The last price return at target_time is strictly below -{band} basis points.",
                    "FLAT": (
                        f"The last price return at target_time is between -{band} and +{band} basis points inclusive."
                    ),
                    "UP": f"The last price return at target_time is strictly above +{band} basis points.",
                },
            },
        },
    }


def _validate_response(raw: dict[str, Any]) -> tuple[dict[str, float], float, int, dict[str, Any]]:
    if raw["model"] != JEV_MODEL:
        raise ValueError("Unexpected model version")
    answer = raw["answers"]["direction"]
    probabilities = answer["probabilities"]
    confidence = answer["confidence"]
    choice = answer["choice"]
    tokens = raw["usage"]["input_tokens"]
    if (
        answer["type"] != "choice"
        or not isinstance(probabilities, dict)
        or set(probabilities) != set(LABELS)
        or choice not in LABELS
        or not all(_finite(value) and 0 <= value <= 1 for value in probabilities.values())
        or not math.isclose(sum(probabilities.values()), 1, rel_tol=0, abs_tol=1e-6)
        or probabilities[choice] != max(probabilities.values())
        or not _finite(confidence)
        or not 0 <= confidence <= 1
        or isinstance(tokens, bool)
        or not isinstance(tokens, int)
        or not 0 <= tokens <= MAX_INPUT_TOKENS
    ):
        raise ValueError("Invalid forecast distribution or usage")
    # Persist only validated response fields; arbitrary provider text may contain private data.
    clean_response = {
        "model": JEV_MODEL,
        "answers": {
            "direction": {
                "type": "choice",
                "choice": choice,
                "probabilities": probabilities,
                "confidence": confidence,
            }
        },
        "usage": {"input_tokens": tokens},
    }
    return probabilities, confidence, tokens, clean_response


class JevForecaster(JevAdvisor):
    def __init__(
        self,
        api_key: str,
        ledger_path: Path | None = None,
        *,
        transport: Callable[[dict[str, Any], str], dict[str, Any]] | None = None,
        clock: Callable[[], datetime] | None = None,
        max_requests: int = 10,
        max_cost_usd: float = 0.05,
        run_id: str = "forecast-pilot-v1",
    ) -> None:
        if isinstance(max_requests, bool) or not isinstance(max_requests, int) or not 1 <= max_requests <= 10:
            raise ValueError("Forecast runs allow 1-10 requests.")
        if not _finite(max_cost_usd) or not 0 < max_cost_usd <= 0.05:
            raise ValueError("Forecast runs allow a positive budget up to $0.05.")
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", run_id):
            raise ValueError("Forecast run ID must be a short identifier.")
        self._run_id = run_id
        super().__init__(api_key, ledger_path, transport=transport, clock=clock)
        with closing(self._connect()) as connection:
            # Guard the schema migration too: concurrent constructors must not race on ALTER TABLE.
            connection.execute("BEGIN IMMEDIATE")
            try:
                columns = {row[1] for row in connection.execute("PRAGMA table_info(jev_attempts)")}
                if "forecast_run_id" not in columns:
                    connection.execute("ALTER TABLE jev_attempts ADD COLUMN forecast_run_id TEXT")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS forecast_run_limits ("
                    "run_id TEXT PRIMARY KEY, max_requests INTEGER NOT NULL, max_cost_usd REAL NOT NULL)"
                )
                limits = connection.execute(
                    "SELECT max_requests, max_cost_usd FROM forecast_run_limits WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                if limits is not None and (max_requests > limits[0] or max_cost_usd > limits[1]):
                    raise ValueError("Cannot expand persisted forecast run limits; use the original or smaller caps.")
                connection.execute(
                    "INSERT INTO forecast_run_limits(run_id, max_requests, max_cost_usd) VALUES (?, ?, ?) "
                    "ON CONFLICT(run_id) DO UPDATE SET max_requests = excluded.max_requests, "
                    "max_cost_usd = excluded.max_cost_usd",
                    (run_id, max_requests, max_cost_usd),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def _reserve_forecast(
        self,
        fingerprint: str,
        request: str,
        fresh_until: datetime | None = None,
        target_time: datetime | None = None,
    ) -> tuple[int | None, dict[str, Any] | None, str | None]:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                now = utc(self._clock()).timestamp()
                if target_time is not None and now >= target_time.timestamp():
                    return None, None, "target_already_elapsed"
                if fresh_until is not None and now > fresh_until.timestamp():
                    return None, None, "stale_source"
                row = connection.execute(
                    "SELECT result_json FROM jev_attempts WHERE fingerprint = ? AND requested_at >= ? "
                    "AND requested_at <= ? AND result_json IS NOT NULL ORDER BY id DESC LIMIT 1",
                    (fingerprint, now - COOLDOWN_SECONDS, now),
                ).fetchone()
                if row:
                    cached = json.loads(row[0])
                    if cached["prediction"]["status"] in {"ok", "abstained"}:
                        return None, dict(cached["prediction"]), None
                count, last = connection.execute("SELECT COUNT(*), MAX(requested_at) FROM jev_attempts").fetchone()
                if count >= self._limit:
                    return None, None, "global_request_limit"
                limits = connection.execute(
                    "SELECT max_requests, max_cost_usd FROM forecast_run_limits WHERE run_id = ?",
                    (self._run_id,),
                ).fetchone()
                attempts, tokens = connection.execute(
                    "SELECT COUNT(*), COALESCE(SUM(COALESCE(input_tokens, ?)), 0) FROM jev_attempts "
                    "WHERE forecast_run_id = ?",
                    (MAX_INPUT_TOKENS, self._run_id),
                ).fetchone()
                if attempts >= limits[0]:
                    return None, None, "run_request_limit"
                cost_reserved = Decimal(tokens + MAX_INPUT_TOKENS) * Decimal(str(INPUT_TOKEN_PRICE_USD))
                if cost_reserved > Decimal(str(limits[1])):
                    return None, None, "run_cost_limit"
                if last is not None and now - last < COOLDOWN_SECONDS:
                    return None, None, "cooldown"
                cursor = connection.execute(
                    "INSERT INTO jev_attempts(requested_at, fingerprint, request_json, forecast_run_id) "
                    "VALUES (?, ?, ?, ?)",
                    (now, fingerprint, request, self._run_id),
                )
                connection.commit()
                return cursor.lastrowid, None, None
            finally:
                # Read-only early returns release their lock; failed inserts are never partially committed.
                if connection.in_transaction:
                    connection.rollback()

    def predict(
        self,
        example: ForecastExample,
        config: ForecastConfig,
        dataset_id: str,
        *,
        prospective: bool = True,
    ) -> ForecastPrediction:
        now = utc(self._clock())
        try:
            example_id = example.example_id
        except (ValueError, TypeError):
            example_id = "invalid-example"
        base: dict[str, Any] = {
            "example_id": example_id,
            "dataset_id": dataset_id,
            "model_id": JEV_MODEL,
            "model_version": JEV_MODEL,
            "prompt_version": FORECAST_PROMPT_VERSION,
            "execution": "live_api",
            "generated_at": now,
        }

        def failure(reason: str, **metadata: Any) -> ForecastPrediction:
            return ForecastPrediction(
                **(
                    base
                    | {
                        "status": "error",
                        "reason": reason,
                        "input_tokens": 0,
                        "estimated_cost_usd": 0,
                    }
                    | metadata
                )
            )

        problem = _input_problem(example, config)
        if problem:
            return failure(problem)
        if prospective:
            if example.provenance == "historical":
                return failure("historical_input_requires_retrospective_mode")
            age = (now - example.as_of).total_seconds()
            if age < 0:
                return failure("future_as_of")
            if now >= example.target_time:
                return failure("target_already_elapsed")
            if age > config.max_source_age_seconds:
                return failure("stale_as_of")
            if age + example.features["source_age_seconds"] > config.max_source_age_seconds:
                return failure("stale_source")
        try:
            payload = _payload(example, config)
            request = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
            if len(request.encode()) > MAX_REQUEST_BYTES:
                return failure("request_too_large")
            # Context includes local evaluation provenance but none of it is sent to the provider.
            fingerprint = hashlib.sha256(
                (
                    FORECAST_PROMPT_VERSION
                    + request
                    + content_hash(
                        {
                            "example_id": example_id,
                            "config_id": config.config_id,
                            "dataset_id": dataset_id,
                            "prospective": prospective,
                        }
                    )
                ).encode()
            ).hexdigest()
            fresh_until = (
                example.as_of
                + timedelta(seconds=config.max_source_age_seconds - example.features["source_age_seconds"])
                if prospective
                else None
            )
            request_id, cached, block = self._reserve_forecast(
                fingerprint, request, fresh_until, example.target_time if prospective else None,
            )
            if cached is not None:
                prediction = ForecastPrediction.model_validate(cached)
                return prediction.model_copy(update={"cached": True})
            if block is not None:
                return failure(block)
        except Exception:
            return failure("ledger_or_request_unavailable")
        assert request_id is not None
        started = time.perf_counter()
        tokens: int | None = None
        clean_response: dict[str, Any] | None = None
        try:
            raw = self._transport(payload, self._api_key)
            probabilities, confidence, tokens, clean_response = _validate_response(raw)
            completed_at = utc(self._clock())
            if prospective and completed_at >= example.target_time:
                prediction = failure("response_after_target", generated_at=completed_at)
            else:
                abstained = min(max(probabilities.values()), confidence) < config.abstention_threshold
                reason = (
                    "retrospective_api_inference"
                    if not prospective
                    else "below_confidence_threshold"
                    if abstained
                    else None
                )
                prediction = ForecastPrediction(
                    **(
                        base
                        | {
                            "generated_at": completed_at,
                            "status": "abstained" if abstained else "ok",
                            "probabilities": probabilities,
                            "model_confidence": confidence,
                            "reason": reason,
                        }
                    )
                )
        except Exception:
            # Provider exceptions and bodies can contain private headers; never persist or return their text.
            prediction = failure("jev_unavailable_or_invalid_response", generated_at=utc(self._clock()))
        latency = (time.perf_counter() - started) * 1000
        prediction = prediction.model_copy(
            update={
                "input_tokens": tokens,
                "latency_ms": latency,
                "request_id": str(request_id),
                "estimated_cost_usd": (tokens * INPUT_TOKEN_PRICE_USD if tokens is not None else _WORST_COST),
            }
        )
        try:
            with closing(self._connect()) as connection:
                connection.execute(
                    "UPDATE jev_attempts SET input_tokens = ?, result_json = ? WHERE id = ?",
                    (
                        tokens,
                        json.dumps(
                            {"prediction": prediction.model_dump(mode="json"), "response": clean_response},
                            allow_nan=False,
                        ),
                        request_id,
                    ),
                )
        except Exception:
            return failure(
                "ledger_settlement_unavailable",
                generated_at=utc(self._clock()),
                input_tokens=None,
                request_id=str(request_id),
                latency_ms=latency,
                estimated_cost_usd=_WORST_COST,
            )
        return prediction


def fixture_predictions(
    examples: Sequence[ForecastExample],
    dataset_id: str,
    config: ForecastConfig,
) -> list[ForecastPrediction]:
    """Offline schema demonstration. Its deterministic heuristic is not Jev model-performance evidence."""
    predictions = []
    for example in examples:
        problem = _input_problem(example, config)
        movement = example.features["return_5m_bps"]
        winner = (
            "UP"
            if movement > config.flat_threshold_bps
            else ("DOWN" if movement < -config.flat_threshold_bps else "FLAT")
        )
        predictions.append(
            ForecastPrediction(
                example_id=example.example_id,
                dataset_id=dataset_id,
                model_id="jev-contract-fixture",
                model_version="deterministic-fixture-v1",
                prompt_version=FORECAST_PROMPT_VERSION,
                execution="fixture",
                generated_at=datetime.now(UTC),
                status="error" if problem else "abstained" if config.abstention_threshold > 0.7 else "ok",
                reason=problem or _FIXTURE_REASON,
                probabilities=None if problem else {label: 0.7 if label == winner else 0.15 for label in LABELS},
                model_confidence=None if problem else 0.7,
                estimated_cost_usd=0,
                input_tokens=0,
                latency_ms=0,
            )
        )
    return predictions
