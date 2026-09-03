from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from tradecopilot.models import MarketFrame


class MockProvider:
    def __init__(self, frames: Sequence[MarketFrame]) -> None:
        self._frames = tuple(frames)

    async def frames(self) -> AsyncIterator[MarketFrame]:
        for frame in self._frames:
            yield frame
