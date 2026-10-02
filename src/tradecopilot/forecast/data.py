"""Append-only local observations, inference records, and operational counters."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from tradecopilot.forecast.contracts import ForecastPrediction, Observation, utc

_SENSITIVE = re.compile(
    r"secret|password|credential|authorization|api.?key|token|exception|traceback|body|message", re.I,
)
_CODE = re.compile(r"[A-Za-z0-9_.:-]{1,120}")


def _event_payload(payload: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in payload.items():
        if _SENSITIVE.search(key) or not _CODE.fullmatch(key):
            raise ValueError("operational events accept counters and safe codes only")
        if (
            value is None or isinstance(value, bool | int)
            or (isinstance(value, float) and math.isfinite(value))
            or (isinstance(value, str) and _CODE.fullmatch(value))
        ):
            result[key] = value
        else:
            raise ValueError("operational events accept counters and safe codes only")
    return result


class ForecastStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, timeout=30)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA busy_timeout=30000")
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS forecast_observations (
                    observation_id TEXT PRIMARY KEY,
                    receipt_timestamp TEXT NOT NULL,
                    provider_timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS forecast_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recorded_at TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS forecast_predictions (
                    prediction_id TEXT PRIMARY KEY,
                    generated_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS forecast_poll_gate (
                    gate_id INTEGER PRIMARY KEY CHECK (gate_id = 1),
                    next_slot REAL NOT NULL
                );
                """
            )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def reserve_collection_slot(self, clock: Callable[[], datetime], spacing_seconds: float = 1.5) -> float:
        """Claim a dispatch slot or return a wait; callers recheck after sleeping, including late wakes."""
        if not math.isfinite(spacing_seconds) or spacing_seconds < 1.5:
            raise ValueError("request spacing must be at least 1.5 seconds")
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            # The writer lock can block: sample dispatch time after acquisition, never before it.
            at = utc(clock()).timestamp()
            row = self._connection.execute("SELECT next_slot FROM forecast_poll_gate WHERE gate_id = 1").fetchone()
            wait = max(0.0, row[0] - at) if row else 0.0
            if wait == 0:
                self._connection.execute(
                    "INSERT INTO forecast_poll_gate VALUES (1, ?) ON CONFLICT(gate_id) DO UPDATE SET next_slot = ?",
                    (at + spacing_seconds, at + spacing_seconds),
                )
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise
        return float(wait)

    def ingest(self, observations: Sequence[Observation]) -> int:
        # Validate the entire batch before opening its transaction, including model_copy inputs.
        validated = [Observation.model_validate(row.model_dump(mode="json")) for row in observations]
        rows = [
            (row.observation_id, row.receipt_timestamp.isoformat(), row.provider_timestamp.isoformat(),
             row.symbol, row.model_dump_json())
            for row in validated
        ]
        before = self._connection.total_changes
        with self._connection:
            self._connection.executemany(
                "INSERT OR IGNORE INTO forecast_observations VALUES (?, ?, ?, ?, ?)", rows,
            )
        return self._connection.total_changes - before

    def observations(self) -> list[Observation]:
        rows = self._connection.execute(
            "SELECT payload FROM forecast_observations "
            "ORDER BY receipt_timestamp, provider_timestamp, symbol, observation_id"
        )
        return [Observation.model_validate_json(row[0]) for row in rows]

    def record_event(self, kind: str, payload: Mapping[str, object]) -> None:
        if not _CODE.fullmatch(kind):
            raise ValueError("operational event kind must be a safe code")
        encoded = json.dumps(_event_payload(payload), sort_keys=True, allow_nan=False)
        with self._connection:
            self._connection.execute(
                "INSERT INTO forecast_events (recorded_at, kind, payload) VALUES (?, ?, ?)",
                (datetime.now(UTC).isoformat(), kind, encoded),
            )

    def events(self) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT event_id, recorded_at, kind, payload FROM forecast_events ORDER BY event_id"
        )
        return [
            {"event_id": row[0], "recorded_at": row[1], "kind": row[2], "payload": json.loads(row[3])}
            for row in rows
        ]

    def save_prediction(self, prediction: ForecastPrediction) -> None:
        prediction = ForecastPrediction.model_validate(prediction.model_dump(mode="json"))
        with self._connection:
            self._connection.execute(
                "INSERT OR IGNORE INTO forecast_predictions VALUES (?, ?, ?)",
                (prediction.prediction_id, prediction.generated_at.isoformat(), prediction.model_dump_json()),
            )

    def predictions(self) -> list[ForecastPrediction]:
        rows = self._connection.execute(
            "SELECT payload FROM forecast_predictions ORDER BY generated_at, prediction_id"
        )
        return [ForecastPrediction.model_validate_json(row[0]) for row in rows]
