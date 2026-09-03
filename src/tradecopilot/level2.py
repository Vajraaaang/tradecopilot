from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import UTC
from decimal import Decimal
from statistics import median

from tradecopilot.config import StrategyConfig
from tradecopilot.models import (
    DataQuality,
    ExitSignal,
    Level2Snapshot,
    SignalAvailability,
    TimeAndSalesPrint,
)


class Level2Analyzer:
    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    def signals(
        self,
        history: Sequence[Level2Snapshot],
        time_and_sales: Sequence[TimeAndSalesPrint] | None,
    ) -> tuple[ExitSignal, ...]:
        latest = history[-1] if history else None
        persistent = self._persistent_seller(history)
        hidden = self._hidden_seller(time_and_sales, latest)
        red_burst = self._red_tape_burst(time_and_sales, latest)
        return persistent, hidden, red_burst

    def _persistent_seller(self, history: Sequence[Level2Snapshot]) -> ExitSignal:
        if not history:
            return _signal(
                "persistent_large_seller",
                "warning",
                False,
                ("Level 2 unavailable",),
                SignalAvailability.UNAVAILABLE,
            )
        latest_time = history[-1].provider_timestamp
        window_seconds = max(5.0, self.config.minimum_seller_persistence_seconds * 3)
        recent_history = [
            snapshot
            for snapshot in history
            if (latest_time - snapshot.provider_timestamp).total_seconds() <= window_seconds
        ]
        all_sizes = [level.size for snapshot in recent_history for level in snapshot.asks if level.size > 0]
        sorted_sizes = sorted(all_sizes)
        normal_count = max(1, (len(sorted_sizes) * 3) // 4)
        normal_sizes = sorted_sizes[:normal_count]
        typical = Decimal(str(median(normal_sizes))) if normal_sizes else Decimal(0)
        threshold = typical * self.config.seller_size_multiple_of_recent_median
        qualifying_prices: list[Decimal] = []
        for snapshot in recent_history:
            for level in snapshot.asks:
                if typical > 0 and Decimal(level.size) >= threshold:
                    qualifying_prices.append(level.price)
        counts = Counter(qualifying_prices)
        if not counts:
            return _signal(
                "persistent_large_seller",
                "warning",
                False,
                (f"No ask reached {self.config.seller_size_multiple_of_recent_median}x median",),
                SignalAvailability.CLEAR,
                recent_history[-1],
            )
        price, count = counts.most_common(1)[0]
        matching = [
            snapshot
            for snapshot in recent_history
            if any(level.price == price and Decimal(level.size) >= threshold for level in snapshot.asks)
        ]
        elapsed = (matching[-1].provider_timestamp - matching[0].provider_timestamp).total_seconds()
        confirmed = (
            count >= self.config.minimum_seller_persistence_snapshots
            and elapsed >= self.config.minimum_seller_persistence_seconds
        )
        return _signal(
            "persistent_large_seller",
            "exit" if confirmed else "warning",
            confirmed,
            (
                f"ask {price} appeared in {count} snapshots",
                f"persistence {elapsed:.1f}s; threshold {threshold:.0f} shares",
            ),
            SignalAvailability.ACTIVE if confirmed else SignalAvailability.LIMITED,
            recent_history[-1],
        )

    def _hidden_seller(
        self,
        prints: Sequence[TimeAndSalesPrint] | None,
        latest: Level2Snapshot | None,
    ) -> ExitSignal:
        if prints is None:
            return _signal(
                "hidden_seller",
                "warning",
                False,
                ("Time-and-sales provider is not configured",),
                SignalAvailability.UNAVAILABLE,
                latest,
            )
        return _signal(
            "hidden_seller",
            "warning",
            False,
            ("No deterministic iceberg confirmation in available prints",),
            SignalAvailability.LIMITED,
            latest,
        )

    def _red_tape_burst(
        self,
        prints: Sequence[TimeAndSalesPrint] | None,
        latest: Level2Snapshot | None,
    ) -> ExitSignal:
        if prints is None:
            return _signal(
                "red_tape_burst",
                "warning",
                False,
                ("Time-and-sales provider is not configured",),
                SignalAvailability.UNAVAILABLE,
                latest,
            )
        recent = list(prints[-10:])
        classified_sides = {"buy", "green", "sell", "red"}
        if recent and any(item.side.lower() not in classified_sides for item in recent):
            return _signal(
                "red_tape_burst",
                "warning",
                False,
                ("Trade prints are available, but aggressor side is not reliably classified",),
                SignalAvailability.LIMITED,
                latest,
            )
        total = sum(item.size for item in recent)
        red = sum(item.size for item in recent if item.side.lower() in {"sell", "red"})
        confirmed = len(recent) >= 5 and total > 0 and Decimal(red) / Decimal(total) >= Decimal("0.70")
        return _signal(
            "red_tape_burst",
            "exit" if confirmed else "warning",
            confirmed,
            (f"red print share {red}/{total}",),
            SignalAvailability.ACTIVE if confirmed else SignalAvailability.CLEAR,
            latest,
        )


def _signal(
    name: str,
    severity: str,
    confirmed: bool,
    evidence: tuple[str, ...],
    availability: SignalAvailability,
    snapshot: Level2Snapshot | None = None,
) -> ExitSignal:
    from datetime import datetime

    now = datetime.now(UTC)
    return ExitSignal(
        provider_timestamp=snapshot.provider_timestamp if snapshot else now,
        receipt_timestamp=snapshot.receipt_timestamp if snapshot else now,
        age_seconds=snapshot.age_seconds if snapshot else 0,
        source="deterministic_level2_analyzer",
        quality=(DataQuality.GOOD if snapshot else DataQuality.UNAVAILABLE),
        name=name,
        severity=severity,
        confirmed=confirmed,
        evidence=evidence,
        availability=availability,
    )
