from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from tradecopilot.models import MarketFrame, PositionChange, PositionChangeKind, PositionSnapshot


@dataclass
class PositionTracker:
    """Detect manual broker position changes; never submits or modifies an order."""

    _positions: dict[str, PositionSnapshot] = field(default_factory=dict)

    def observe_all(self, frame: MarketFrame) -> tuple[PositionChange, ...]:
        current = {position.symbol.upper(): position for position in frame.positions if position.quantity > 0}
        if frame.position is not None and frame.position.quantity > 0:
            current[frame.position.symbol.upper()] = frame.position
        changes: list[PositionChange] = []
        for symbol in sorted(set(self._positions) | set(current)):
            prior_position = self._positions.get(symbol)
            current_position = current.get(symbol)
            prior_quantity = prior_position.quantity if prior_position is not None else Decimal(0)
            current_quantity = current_position.quantity if current_position is not None else Decimal(0)
            if current_quantity == prior_quantity:
                continue
            if prior_quantity == 0 and current_quantity > 0:
                kind = PositionChangeKind.ENTRY
            elif prior_quantity > 0 and current_quantity == 0:
                kind = PositionChangeKind.EXIT
            elif current_quantity > prior_quantity:
                kind = PositionChangeKind.INCREASE
            else:
                kind = PositionChangeKind.REDUCTION
            evidence = current_position or prior_position
            assert evidence is not None
            provider_timestamp = evidence.provider_timestamp if current_position is not None else frame.event_time
            receipt_timestamp = evidence.receipt_timestamp if current_position is not None else frame.event_time
            changes.append(
                PositionChange(
                    provider_timestamp=provider_timestamp,
                    receipt_timestamp=receipt_timestamp,
                    age_seconds=(receipt_timestamp - provider_timestamp).total_seconds(),
                    source="broker_position_diff",
                    quality=evidence.quality,
                    symbol=symbol,
                    account_alias=evidence.account_alias,
                    kind=kind,
                    prior_quantity=prior_quantity,
                    new_quantity=current_quantity,
                    average_entry=evidence.average_entry,
                )
            )
        self._positions = current
        return tuple(changes)

    def observe(self, frame: MarketFrame) -> PositionChange | None:
        changes = self.observe_all(frame)
        return changes[0] if changes else None
