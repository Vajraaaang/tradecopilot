from __future__ import annotations

import json
import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from tradecopilot.models import (
    ExperimentCandidate,
    JournalObservation,
    PositionChange,
    StateTransition,
    StrategyVersion,
)
from tradecopilot.redaction import redact

EASTERN = ZoneInfo("America/New_York")


class Journal:
    def __init__(self, path: Path, retention_rows: int = 10_000) -> None:
        self.path = path
        self.retention_rows = retention_rows
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                occurred_at TEXT NOT NULL,
                trading_date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                prior_state TEXT,
                new_state TEXT NOT NULL,
                strategy_version TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_transitions_date
                ON transitions(trading_date, occurred_at);

            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                occurred_at TEXT NOT NULL,
                trading_date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                mode TEXT NOT NULL,
                state TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_observations_date
                ON observations(trading_date, occurred_at);

            CREATE TABLE IF NOT EXISTS session_locks (
                trading_date TEXT PRIMARY KEY,
                locked_at TEXT NOT NULL,
                reason TEXT NOT NULL,
                mode TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS position_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                occurred_at TEXT NOT NULL,
                trading_date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                account_alias TEXT,
                event_kind TEXT NOT NULL,
                prior_quantity TEXT NOT NULL,
                new_quantity TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_position_events_date
                ON position_events(trading_date, occurred_at);

            CREATE TABLE IF NOT EXISTS strategy_versions (
                version TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                deployment_status TEXT NOT NULL,
                execution_mode TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS experiments (
                candidate_id TEXT PRIMARY KEY,
                base_version TEXT NOT NULL,
                candidate_version TEXT NOT NULL,
                changed_parameter TEXT NOT NULL,
                human_approval_status TEXT NOT NULL,
                research_result TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            """
        )
        self._connection.commit()

    def record_transition(self, transition: StateTransition) -> None:
        payload = _safe_json(transition)
        self._connection.execute(
            """
            INSERT INTO transitions (
                occurred_at, trading_date, symbol, prior_state, new_state,
                strategy_version, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                transition.provider_timestamp.isoformat(),
                transition.provider_timestamp.astimezone(EASTERN).date().isoformat(),
                transition.symbol,
                transition.prior_state.value if transition.prior_state else None,
                transition.new_state.value,
                transition.strategy_version,
                payload,
            ),
        )

    def record_observation(self, observation: JournalObservation) -> None:
        self._connection.execute(
            """
            INSERT INTO observations (
                occurred_at, trading_date, symbol, mode, state, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                observation.provider_timestamp.isoformat(),
                observation.trading_date.isoformat(),
                observation.symbol,
                observation.mode.value,
                observation.decision.state.value,
                _safe_json(observation),
            ),
        )
        self._prune_observations()

    def record_position_change(self, change: PositionChange) -> None:
        self._connection.execute(
            """
            INSERT INTO position_events (
                occurred_at, trading_date, symbol, account_alias, event_kind,
                prior_quantity, new_quantity, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                change.provider_timestamp.isoformat(),
                change.provider_timestamp.astimezone(EASTERN).date().isoformat(),
                change.symbol,
                change.account_alias,
                change.kind.value,
                str(change.prior_quantity),
                str(change.new_quantity),
                _safe_json(change),
            ),
        )

    def _prune_observations(self) -> None:
        self._connection.execute(
            """
            DELETE FROM observations
            WHERE id NOT IN (
                SELECT id FROM observations ORDER BY id DESC LIMIT ?
            )
            """,
            (self.retention_rows,),
        )

    def lock_session(self, trading_date: date, reason: str, mode: str) -> None:
        self._connection.execute(
            """
            INSERT OR IGNORE INTO session_locks (trading_date, locked_at, reason, mode)
            VALUES (?, ?, ?, ?)
            """,
            (trading_date.isoformat(), datetime.now(UTC).isoformat(), reason, mode),
        )
        self._connection.commit()

    def is_session_locked(self, trading_date: date) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM session_locks WHERE trading_date = ?", (trading_date.isoformat(),)
        ).fetchone()
        return row is not None

    def administrative_reset(self, trading_date: date, *, replay_or_test: bool) -> None:
        if not replay_or_test:
            raise PermissionError("live session locks never auto-reset")
        self._connection.execute("DELETE FROM session_locks WHERE trading_date = ?", (trading_date.isoformat(),))
        self._connection.commit()

    def save_strategy(self, strategy: StrategyVersion) -> None:
        self._connection.execute(
            """
            INSERT OR REPLACE INTO strategy_versions (
                version, strategy_id, deployment_status, execution_mode, payload_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                strategy.version,
                strategy.strategy_id,
                strategy.deployment_status,
                strategy.execution_mode,
                _safe_json(strategy),
            ),
        )

    def save_experiment(self, candidate: ExperimentCandidate) -> None:
        self._connection.execute(
            """
            INSERT OR REPLACE INTO experiments (
                candidate_id, base_version, candidate_version, changed_parameter,
                human_approval_status, research_result, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate.candidate_id,
                candidate.base_version,
                candidate.candidate_version,
                candidate.changed_parameter,
                candidate.human_approval_status,
                candidate.research_result,
                _safe_json(candidate),
            ),
        )

    def get_experiment(self, candidate_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT payload_json FROM experiments WHERE candidate_id = ?", (candidate_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def report(self, trading_date: date) -> dict[str, Any]:
        states = self._connection.execute(
            """
            SELECT new_state, COUNT(*) FROM transitions
            WHERE trading_date = ? GROUP BY new_state ORDER BY new_state
            """,
            (trading_date.isoformat(),),
        ).fetchall()
        lock = self._connection.execute(
            "SELECT locked_at, reason FROM session_locks WHERE trading_date = ?",
            (trading_date.isoformat(),),
        ).fetchone()
        position_events = self._connection.execute(
            """
            SELECT event_kind, COUNT(*) FROM position_events
            WHERE trading_date = ? GROUP BY event_kind ORDER BY event_kind
            """,
            (trading_date.isoformat(),),
        ).fetchall()
        return {
            "date": trading_date.isoformat(),
            "transitions": {state: count for state, count in states},
            "session_lock": {"locked_at": lock[0], "reason": lock[1]} if lock else None,
            "position_events": {kind: count for kind, count in position_events},
        }

    def flush(self) -> None:
        self._connection.commit()

    def close(self) -> None:
        self.flush()
        self._connection.close()

    def __enter__(self) -> Journal:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()


def _safe_json(model: BaseModel) -> str:
    raw = model.model_dump(mode="json")
    return json.dumps(redact(raw), sort_keys=True, separators=(",", ":"))
