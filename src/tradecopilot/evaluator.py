from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ReplayOutcome(BaseModel):
    """One chronologically observed signal outcome; no future fields are optional inputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    signal_time: datetime
    session: date
    realized_r: Decimal
    mfe_r: Decimal
    mae_r: Decimal
    spread_dollars: Decimal = Field(ge=0)
    slippage_dollars: Decimal = Field(ge=0)
    false_breakout: bool = False
    halted: bool = False
    failed_fill: bool | None = None
    skipped: bool = False
    time_bucket: str = "unknown"
    price_bucket: str = "unknown"
    float_bucket: str = "unknown"
    gap_bucket: str = "unknown"
    rvol_bucket: str = "unknown"

    @field_validator("signal_time")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("signal_time must be timezone-aware")
        return value


class EvaluationMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    signals: int
    sessions: int
    expectancy_r: Decimal
    profit_factor: Decimal | None
    maximum_drawdown_r: Decimal
    worst_decile_loss_r: Decimal
    false_breakout_rate: Decimal
    average_mfe_r: Decimal
    average_mae_r: Decimal
    average_spread_dollars: Decimal
    average_slippage_dollars: Decimal
    failed_fills_observed: int
    failed_fills_unknown: int
    halts: int
    skipped_signals: int
    stability: dict[str, dict[str, Decimal]]


class WalkForwardWindow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    training_sessions: tuple[date, ...]
    validation_sessions: tuple[date, ...]
    out_of_sample_sessions: tuple[date, ...]
    out_of_sample_metrics: EvaluationMetrics


class ReplayEvaluator:
    """Chronological walk-forward evaluation; input order cannot create look-ahead."""

    def evaluate(self, outcomes: Sequence[ReplayOutcome]) -> EvaluationMetrics:
        ordered = sorted(outcomes, key=lambda outcome: outcome.signal_time)
        return _metrics(ordered)

    def walk_forward(
        self,
        outcomes: Sequence[ReplayOutcome],
        *,
        training_sessions: int,
        validation_sessions: int,
        out_of_sample_sessions: int,
    ) -> tuple[WalkForwardWindow, ...]:
        if min(training_sessions, validation_sessions, out_of_sample_sessions) <= 0:
            raise ValueError("walk-forward session windows must be positive")
        ordered = sorted(outcomes, key=lambda outcome: outcome.signal_time)
        sessions = sorted({outcome.session for outcome in ordered})
        span = training_sessions + validation_sessions + out_of_sample_sessions
        windows: list[WalkForwardWindow] = []
        for start in range(0, len(sessions) - span + 1, out_of_sample_sessions):
            train_end = start + training_sessions
            validation_end = train_end + validation_sessions
            end = validation_end + out_of_sample_sessions
            training = tuple(sessions[start:train_end])
            validation = tuple(sessions[train_end:validation_end])
            out_of_sample = tuple(sessions[validation_end:end])
            observed = [outcome for outcome in ordered if outcome.session in out_of_sample]
            windows.append(
                WalkForwardWindow(
                    training_sessions=training,
                    validation_sessions=validation,
                    out_of_sample_sessions=out_of_sample,
                    out_of_sample_metrics=_metrics(observed),
                )
            )
        return tuple(windows)


def _metrics(outcomes: Sequence[ReplayOutcome]) -> EvaluationMetrics:
    if not outcomes:
        zero = Decimal(0)
        return EvaluationMetrics(
            signals=0,
            sessions=0,
            expectancy_r=zero,
            profit_factor=None,
            maximum_drawdown_r=zero,
            worst_decile_loss_r=zero,
            false_breakout_rate=zero,
            average_mfe_r=zero,
            average_mae_r=zero,
            average_spread_dollars=zero,
            average_slippage_dollars=zero,
            failed_fills_observed=0,
            failed_fills_unknown=0,
            halts=0,
            skipped_signals=0,
            stability={},
        )
    realized = [outcome.realized_r for outcome in outcomes]
    gains = sum((value for value in realized if value > 0), Decimal(0))
    losses = abs(sum((value for value in realized if value < 0), Decimal(0)))
    running = Decimal(0)
    peak = Decimal(0)
    maximum_drawdown = Decimal(0)
    for value in realized:
        running += value
        peak = max(peak, running)
        maximum_drawdown = max(maximum_drawdown, peak - running)
    count = Decimal(len(outcomes))
    worst_count = max(1, (len(realized) + 9) // 10)
    worst = sorted(realized)[:worst_count]
    return EvaluationMetrics(
        signals=len(outcomes),
        sessions=len({outcome.session for outcome in outcomes}),
        expectancy_r=sum(realized, Decimal(0)) / count,
        profit_factor=gains / losses if losses else None,
        maximum_drawdown_r=maximum_drawdown,
        worst_decile_loss_r=sum(worst, Decimal(0)) / Decimal(len(worst)),
        false_breakout_rate=Decimal(sum(outcome.false_breakout for outcome in outcomes)) / count,
        average_mfe_r=sum((outcome.mfe_r for outcome in outcomes), Decimal(0)) / count,
        average_mae_r=sum((outcome.mae_r for outcome in outcomes), Decimal(0)) / count,
        average_spread_dollars=sum((outcome.spread_dollars for outcome in outcomes), Decimal(0)) / count,
        average_slippage_dollars=sum((outcome.slippage_dollars for outcome in outcomes), Decimal(0)) / count,
        failed_fills_observed=sum(outcome.failed_fill is True for outcome in outcomes),
        failed_fills_unknown=sum(outcome.failed_fill is None for outcome in outcomes),
        halts=sum(outcome.halted for outcome in outcomes),
        skipped_signals=sum(outcome.skipped for outcome in outcomes),
        stability=_stability(outcomes),
    )


def _stability(outcomes: Sequence[ReplayOutcome]) -> dict[str, dict[str, Decimal]]:
    dimensions = ("time_bucket", "price_bucket", "float_bucket", "gap_bucket", "rvol_bucket")
    result: dict[str, dict[str, Decimal]] = {}
    for dimension in dimensions:
        buckets: dict[str, list[Decimal]] = {}
        for outcome in outcomes:
            buckets.setdefault(str(getattr(outcome, dimension)), []).append(outcome.realized_r)
        result[dimension] = {
            bucket: sum(values, Decimal(0)) / Decimal(len(values)) for bucket, values in sorted(buckets.items())
        }
    return result
