from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from tradecopilot.models import (
    AccountRiskSnapshot,
    CatalystEvidence,
    DataQuality,
    FloatEvidence,
    Level2Level,
    Level2Snapshot,
    MarketFrame,
    OHLCVBar,
    PositionSnapshot,
    Quote,
    RunMode,
    TimeAndSalesPrint,
)


class ReplayFormatError(ValueError):
    pass


_DATETIME = TypeAdapter(datetime)
_DATE = TypeAdapter(date)


def _dt(value: str) -> datetime:
    parsed = _DATETIME.validate_python(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ReplayFormatError("replay timestamps must be timezone-aware")
    return parsed


def _age(provider_timestamp: datetime, event_time: datetime) -> float:
    age = (event_time - provider_timestamp).total_seconds()
    if age < 0:
        raise ReplayFormatError("replay contains look-ahead data")
    return age


def _stamp(provider_timestamp: datetime, event_time: datetime, source: str) -> dict[str, Any]:
    return {
        "provider_timestamp": provider_timestamp,
        "receipt_timestamp": event_time,
        "age_seconds": _age(provider_timestamp, event_time),
        "source": source,
        "quality": DataQuality.GOOD,
    }


def _bar(raw: Mapping[str, Any], event_time: datetime, timeframe: str, source: str) -> OHLCVBar:
    timestamp = _dt(str(raw["timestamp"]))
    return OHLCVBar(
        **_stamp(timestamp, event_time, source),
        symbol=str(raw["symbol"]).upper(),
        timeframe=timeframe,
        open=Decimal(str(raw["open"])),
        high=Decimal(str(raw["high"])),
        low=Decimal(str(raw["low"])),
        close=Decimal(str(raw["close"])),
        volume=int(raw["volume"]),
        extended_hours=bool(raw.get("extended_hours", False)),
    )


def _quote(raw: Mapping[str, Any], event_time: datetime, source: str) -> Quote:
    timestamp = _dt(str(raw.get("timestamp", event_time.isoformat())))
    quote = Quote(
        **_stamp(timestamp, event_time, source),
        symbol=str(raw["symbol"]).upper(),
        bid=Decimal(str(raw["bid"])),
        ask=Decimal(str(raw["ask"])),
        last=Decimal(str(raw["last"])),
        previous_close=Decimal(str(raw["previous_close"])),
        total_volume=int(raw["total_volume"]),
        current_minute_volume=int(raw.get("current_minute_volume", 0)),
        session_origin_price=Decimal(str(raw["session_origin_price"])),
        average_daily_volume_50d=int(raw["average_daily_volume_50d"]),
        average_cumulative_volume_same_time=(
            int(raw["average_cumulative_volume_same_time"])
            if raw.get("average_cumulative_volume_same_time") is not None
            else None
        ),
    )
    if quote.ask < quote.bid:
        raise ReplayFormatError("quote ask is below bid")
    return quote


def _level2(raw: Mapping[str, Any], event_time: datetime, source: str) -> Level2Snapshot:
    timestamp = _dt(str(raw.get("timestamp", event_time.isoformat())))
    stamped = _stamp(timestamp, event_time, source)

    def levels(side: str) -> tuple[Level2Level, ...]:
        return tuple(
            Level2Level(
                **stamped,
                side=side,
                price=Decimal(str(item["price"])),
                size=int(item["size"]),
            )
            for item in raw.get(f"{side}s", [])
        )

    return Level2Snapshot(
        **stamped,
        symbol=str(raw["symbol"]).upper(),
        bids=levels("bid"),
        asks=levels("ask"),
    )


def _position(raw: Mapping[str, Any], event_time: datetime, source: str) -> PositionSnapshot:
    timestamp = _dt(str(raw.get("timestamp", event_time.isoformat())))
    return PositionSnapshot(
        **_stamp(timestamp, event_time, source),
        symbol=str(raw["symbol"]).upper(),
        quantity=Decimal(str(raw["quantity"])),
        average_entry=(Decimal(str(raw["average_entry"])) if raw.get("average_entry") is not None else None),
        unrealized_pnl=(Decimal(str(raw["unrealized_pnl"])) if raw.get("unrealized_pnl") is not None else None),
        latest_higher_low=(
            Decimal(str(raw["latest_higher_low"])) if raw.get("latest_higher_low") is not None else None
        ),
        mfe=Decimal(str(raw["mfe"])) if raw.get("mfe") is not None else None,
        mae=Decimal(str(raw["mae"])) if raw.get("mae") is not None else None,
    )


def _risk(raw: Mapping[str, Any], event_time: datetime, source: str) -> AccountRiskSnapshot:
    timestamp = _dt(str(raw.get("timestamp", event_time.isoformat())))
    return AccountRiskSnapshot(
        **_stamp(timestamp, event_time, source),
        trading_date=_DATE.validate_python(raw["trading_date"]),
        buying_power=Decimal(str(raw["buying_power"])),
        realized_session_pnl=Decimal(str(raw["realized_session_pnl"])),
        peak_realized_session_pnl=Decimal(str(raw["peak_realized_session_pnl"])),
        unrealized_pnl=Decimal(str(raw.get("unrealized_pnl", 0))),
        consecutive_losses=int(raw["consecutive_losses"]),
        session_locked=bool(raw.get("session_locked", False)),
    )


def _float(raw: Mapping[str, Any], event_time: datetime, source: str) -> FloatEvidence:
    timestamp = _dt(str(raw["timestamp"]))
    return FloatEvidence(
        **_stamp(timestamp, event_time, source),
        shares=int(raw["shares"]),
        verified=bool(raw["verified"]),
        reference=str(raw["reference"]),
    )


def _catalyst(raw: Mapping[str, Any], event_time: datetime, source: str) -> CatalystEvidence:
    timestamp = _dt(str(raw["timestamp"]))
    return CatalystEvidence(
        **_stamp(timestamp, event_time, source),
        description=str(raw["description"]),
        verified=bool(raw["verified"]),
        reference=str(raw["reference"]),
    )


def _tape(rows: list[Mapping[str, Any]], event_time: datetime, source: str) -> tuple[TimeAndSalesPrint, ...]:
    return tuple(
        TimeAndSalesPrint(
            **_stamp(_dt(str(row["timestamp"])), event_time, source),
            symbol=str(row["symbol"]).upper(),
            price=Decimal(str(row["price"])),
            size=int(row["size"]),
            side=str(row["side"]),
        )
        for row in rows
    )


class ReplayProvider:
    """Deterministic JSONL replay with event-time gating and no look-ahead."""

    def __init__(self, path: Path, speed: float = 1.0) -> None:
        if speed < 0:
            raise ValueError("speed cannot be negative")
        self.path = path
        self.speed = speed

    def _rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                try:
                    value = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ReplayFormatError(f"invalid JSON on line {line_number}: {exc}") from exc
                if not isinstance(value, dict):
                    raise ReplayFormatError(f"line {line_number} must be an object")
                rows.append(value)
        return rows

    async def frames(self) -> AsyncIterator[MarketFrame]:
        bars_1m: list[OHLCVBar] = []
        bars_5m: list[OHLCVBar] = []
        level2_history: list[Level2Snapshot] = []
        current_position: PositionSnapshot | None = None
        current_risk: AccountRiskSnapshot | None = None
        current_float: FloatEvidence | None = None
        current_catalyst: CatalystEvidence | None = None
        previous_event_time: datetime | None = None
        source = f"replay:{self.path.name}"

        for raw in self._rows():
            event_time = _dt(str(raw["event_time"]))
            if previous_event_time is not None:
                if event_time < previous_event_time:
                    raise ReplayFormatError("replay events are not chronological")
                if self.speed > 0:
                    delay = (event_time - previous_event_time).total_seconds() / self.speed
                    await asyncio.sleep(max(0, delay))
            previous_event_time = event_time

            for item in raw.get("seed_1m", []):
                bars_1m.append(_bar(item, event_time, "1m", "replay_seed"))
            for item in raw.get("seed_5m", []):
                bars_5m.append(_bar(item, event_time, "5m", "replay_seed"))
            if raw.get("bar_1m") is not None:
                bars_1m.append(_bar(raw["bar_1m"], event_time, "1m", source))
            if raw.get("bar_5m") is not None:
                bars_5m.append(_bar(raw["bar_5m"], event_time, "5m", source))

            bars_1m = _dedupe_bars(bars_1m)
            bars_5m = _dedupe_bars(bars_5m)

            level2 = _level2(raw["level2"], event_time, source) if raw.get("level2") else None
            if level2 is not None:
                signature = (level2.provider_timestamp, level2.bids, level2.asks)
                if (
                    not level2_history
                    or (
                        level2_history[-1].provider_timestamp,
                        level2_history[-1].bids,
                        level2_history[-1].asks,
                    )
                    != signature
                ):
                    level2_history.append(level2)
                level2_history = level2_history[-20:]

            tape_raw = raw.get("time_and_sales")
            tape = _tape(tape_raw, event_time, source) if tape_raw is not None else None
            if raw.get("position") is not None:
                current_position = _position(raw["position"], event_time, source)
            if raw.get("account_risk") is not None:
                current_risk = _risk(raw["account_risk"], event_time, source)
            if raw.get("float_evidence") is not None:
                current_float = _float(raw["float_evidence"], event_time, source)
            if raw.get("catalyst_evidence") is not None:
                current_catalyst = _catalyst(raw["catalyst_evidence"], event_time, source)
            frame_level2 = tuple(_refresh_level2(snapshot, event_time) for snapshot in level2_history)
            frame = MarketFrame(
                event_time=event_time,
                mode=RunMode.REPLAY,
                quote=_quote(raw["quote"], event_time, source) if raw.get("quote") else None,
                bars_1m=tuple(bars_1m),
                bars_5m=tuple(bars_5m),
                level2_history=frame_level2,
                time_and_sales=tape,
                position=_refresh_stamp(current_position, event_time),
                account_risk=_refresh_stamp(current_risk, event_time),
                float_evidence=_refresh_stamp(current_float, event_time),
                catalyst_evidence=_refresh_stamp(current_catalyst, event_time),
                resistance_levels=tuple(Decimal(str(v)) for v in raw.get("resistance_levels", [])),
                gap_percent=(Decimal(str(raw["gap_percent"])) if raw.get("gap_percent") is not None else None),
                market_leader=bool(raw.get("market_leader", False)),
                no_a_quality_candidates=bool(raw.get("no_a_quality_candidates", False)),
                bearish_momentum_environment=bool(raw.get("bearish_momentum_environment", False)),
                tradability_known=bool(raw.get("tradability_known", True)),
                halted=bool(raw.get("halted", False)),
            )
            yield frame


def _dedupe_bars(bars: list[OHLCVBar]) -> list[OHLCVBar]:
    by_key: dict[tuple[str, datetime], OHLCVBar] = {}
    for bar in bars:
        by_key[(bar.timeframe, bar.provider_timestamp)] = bar
    return sorted(by_key.values(), key=lambda bar: bar.provider_timestamp)


def _refresh_stamp(model: Any, event_time: datetime) -> Any:
    if model is None:
        return None
    return model.model_copy(
        update={
            "receipt_timestamp": event_time,
            "age_seconds": _age(model.provider_timestamp, event_time),
        }
    )


def _refresh_level2(snapshot: Level2Snapshot, event_time: datetime) -> Level2Snapshot:
    update = {
        "receipt_timestamp": event_time,
        "age_seconds": _age(snapshot.provider_timestamp, event_time),
    }
    bids = tuple(level.model_copy(update=update) for level in snapshot.bids)
    asks = tuple(level.model_copy(update=update) for level in snapshot.asks)
    return snapshot.model_copy(update={**update, "bids": bids, "asks": asks})
