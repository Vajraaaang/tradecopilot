"""Immutable local Kronos reports; no provider or inference calls during display."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from zoneinfo import ZoneInfo

from tradecopilot.forecast.bars import HistoricalBar
from tradecopilot.forecast.contracts import content_hash, utc
from tradecopilot.forecast.sessions import session_bounds

SCHEMA = "kronos-paper-report-v1"
_MAX_BYTES = 256 * 1024 * 1024
_MINUTE = timedelta(minutes=1)
_NEW_YORK = ZoneInfo("America/New_York")


def _json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def validate_prospective_grid(future_times: Any) -> tuple[datetime, ...]:
    """Require the fixed 15 consecutive UTC minute ends in one regular XNYS session."""
    try:
        if not isinstance(future_times, (list, tuple)) or len(future_times) != 15:
            raise ValueError
        future = tuple(datetime.fromisoformat(t) for t in future_times)
        bounds = session_bounds(future[0].astimezone(_NEW_YORK).date())
        if bounds is None or any(
            t.tzinfo is None or t.utcoffset() != timedelta(0) or t.second or t.microsecond
            or not bounds[0] < t <= bounds[1] or (index > 0 and t - future[index - 1] != _MINUTE)
            for index, t in enumerate(future)
        ):
            raise ValueError
        return future
    except (TypeError, ValueError, OverflowError):
        raise ValueError("invalid prospective target grid") from None


def _validate_publication(report: dict[str, Any], completed_at: datetime | None = None) -> None:
    """Bind successful prospective paths to the original sealed bundle's publication."""
    for row in report.get("cases", []):
        if row["group"] != "prospective":
            continue
        future = validate_prospective_grid(row.get("future_times"))
        for forecast in row["forecasts"].values():
            if forecast["status"] != "ok":
                continue
            try:
                published = utc(datetime.fromisoformat(report["published_at"]))
                generated = utc(datetime.fromisoformat(forecast["generated_at"]))
                target = future[0]
            except (KeyError, TypeError, ValueError, IndexError):
                raise ValueError("prospective publication timestamps are required") from None
            if generated > published or published >= target or (completed_at is not None and completed_at >= target):
                raise ValueError("prospective publication must complete before the first target")


def _immutable_forecast(report: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(report)
    for field in ("report_id", "inventory", "bundle_created_at", "parent_report_id", "outcome_source_id"):
        value.pop(field, None)
    for row in value.get("cases", []):
        if row["group"] == "prospective":
            for field in ("actual_close", "outcome_status", "outcomes_received_at"):
                row.pop(field, None)
    return value


def _validate_original(report: dict[str, Any], artifacts: dict[str, bytes], original_path: Path | None) -> None:
    if original_path is None or not original_path.is_absolute():
        raise ValueError("derived reports require a verified original report path")
    original = load_report(original_path)
    if (
        report.get("parent_report_id") != original["report_id"]
        or report.get("published_at") != original["published_at"]
        or _immutable_forecast(report) != _immutable_forecast(original)
    ):
        raise ValueError("derived report must preserve the original forecast, context and publication")
    for name, record in original["inventory"].items():
        if name == "outcome-source.json":
            continue
        body = artifacts.get(name)
        if body is None or len(body) != record["bytes"] or hashlib.sha256(body).hexdigest() != record["sha256"]:
            raise ValueError("derived report must preserve original forecast artifacts")


def write_report(
    directory: Path,
    report: dict[str, Any],
    artifacts: dict[str, bytes],
    *,
    original_report_path: Path | None = None,
) -> Path:
    if not directory.is_absolute():
        raise ValueError("report directory must be absolute")
    if directory.exists():
        raise FileExistsError("report bundles are immutable")
    if report.get("parent_report_id") or original_report_path is not None:
        _validate_original(report, artifacts, original_report_path)
    if any(
        Path(name).name != name or name in ("report.json", ".", "..") or len(body) > _MAX_BYTES
        for name, body in artifacts.items()
    ):
        raise ValueError("invalid report artifact")
    inventory = {
        name: {"sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}
        for name, body in sorted(artifacts.items())
    }
    value = report | {"schema_version": SCHEMA, "inventory": inventory}
    value.pop("report_id", None)
    if len(_json(value)) > _MAX_BYTES:
        raise ValueError("report too large")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".kronos-report-", dir=directory.parent) as temp:
        stage = Path(temp) / "bundle"
        stage.mkdir(mode=0o700)
        for name, body in artifacts.items():
            (stage / name).write_bytes(body)
            (stage / name).chmod(0o600)
        publication = datetime.now(UTC)
        value["bundle_created_at"] = publication.isoformat()
        if not value.get("parent_report_id"):
            value["published_at"] = publication.isoformat()
        _validate_publication(value)
        value["report_id"] = content_hash(value)
        report_bytes = _json(value)
        if len(report_bytes) > _MAX_BYTES:
            raise ValueError("report too large")
        (stage / "report.json").write_bytes(report_bytes)
        (stage / "report.json").chmod(0o600)
        if directory.exists():
            raise FileExistsError("report bundles are immutable")
        if not value.get("parent_report_id"):
            _validate_publication(value, datetime.now(UTC))
        stage.rename(directory)
        # A deadline crossed during the atomic rename must not leave an accepted record.
        try:
            if not value.get("parent_report_id"):
                _validate_publication(value, datetime.now(UTC))
        except ValueError:
            directory.rename(stage)
            raise
    return directory / "report.json"


def _read(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_BYTES:
        raise ValueError("invalid local report artifact")
    with path.open("rb") as stream:
        data = stream.read(_MAX_BYTES + 1)
    if len(data) > _MAX_BYTES:
        raise ValueError("oversized local report artifact")
    return data


def load_report(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(_read(path))
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != SCHEMA
            or value.get("report_id") != content_hash({k: v for k, v in value.items() if k != "report_id"})
            or not isinstance(value.get("inventory"), dict)
        ):
            raise ValueError
        expected = {"report.json"}
        for name, record in value["inventory"].items():
            if Path(name).name != name or name in (".", "..", "report.json"):
                raise ValueError
            data = _read(path.parent / name)
            if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError
            expected.add(name)
        if {p.name for p in path.parent.iterdir()} != expected:
            raise ValueError
        _validate_publication(value)
        return value
    except (ValueError, OSError, TypeError, KeyError):
        raise ValueError("Kronos report integrity verification failed") from None


def attach_outcomes(
    report: dict[str, Any],
    bars: Sequence[HistoricalBar],
    source_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Join subsequently received exact candles without rewriting any forecast."""
    feed = report["connection"]["feed"]
    if feed not in ("iex", "sip") or source_metadata.get("feed") != feed:
        raise ValueError("outcome feed must match the original forecast context")
    _validate_publication(report)
    receipt = utc(datetime.fromisoformat(source_metadata["receipt_at"]))
    lookup = {(b.symbol, b.end_time): b for b in bars}
    if len(lookup) != len(bars):
        raise ValueError("duplicate outcome minute")
    result = copy.deepcopy(report)
    for row in result["cases"]:
        if row["group"] != "prospective":
            continue
        future = tuple(utc(datetime.fromisoformat(t)) for t in row["future_times"])
        if receipt < future[-1]:
            raise ValueError("outcome request was received before the final target")
        outcomes = [lookup.get((row["symbol"], t)) for t in future]
        if any(b is not None and (b.source != f"alpaca_{feed}_1min_bar" or b.available_at > receipt) for b in outcomes):
            raise ValueError("outcome candles must match the original feed and be available at source receipt")
        if any(b is None for b in outcomes):
            row["outcome_status"] = "pending_missing_exact_candles"
            row["actual_close"] = None
            row.pop("outcomes_received_at", None)
        else:
            row["actual_close"] = [float(b.close) for b in outcomes if b is not None]
            row["outcome_status"] = "observed"
            row["outcomes_received_at"] = receipt.isoformat()
    return result
