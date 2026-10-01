"""Regular XNYS sessions; the close is included only as an outcome endpoint."""

from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache
from importlib.metadata import version
from typing import Any

import exchange_calendars  # type: ignore[import-untyped]

from tradecopilot.forecast.contracts import utc

CALENDAR_VERSION = f"exchange-calendars-{version('exchange-calendars')}"


@lru_cache(maxsize=16)
def _calendar(year: int) -> Any:
    return exchange_calendars.get_calendar(
        "XNYS", start=f"{year - 1}-01-01", end=f"{year + 1}-12-31",
    )


@lru_cache(maxsize=4096)
def session_bounds(day: date) -> tuple[datetime, datetime] | None:
    calendar = _calendar(day.year)
    label = day.isoformat()
    if not calendar.is_session(label):
        return None
    return utc(calendar.session_open(label).to_pydatetime()), utc(calendar.session_close(label).to_pydatetime())


def session_for(timestamp: datetime) -> date | None:
    timestamp = utc(timestamp)
    bounds = session_bounds(timestamp.date())
    if bounds is not None and bounds[0] <= timestamp <= bounds[1]:
        return timestamp.date()
    return None
