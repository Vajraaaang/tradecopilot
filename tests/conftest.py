from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from tradecopilot.models import MarketFrame
from tradecopilot.providers.replay import ReplayProvider

ROOT = Path(__file__).resolve().parents[1]


async def collect_frames(path: Path) -> list[MarketFrame]:
    return [frame async for frame in ReplayProvider(path, speed=0).frames()]


@pytest.fixture(scope="session")
def yxt_frames() -> list[MarketFrame]:
    return asyncio.run(collect_frames(ROOT / "examples/yxt_replay.jsonl"))


@pytest.fixture(scope="session")
def day_stop_frame() -> MarketFrame:
    return asyncio.run(collect_frames(ROOT / "examples/day_stop_replay.jsonl"))[0]


class FramesProvider:
    def __init__(self, frames: list[MarketFrame]) -> None:
        self._frames = frames

    async def frames(self) -> AsyncIterator[MarketFrame]:
        for frame in self._frames:
            yield frame
