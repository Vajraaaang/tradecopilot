from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict

from tradecopilot.config import StrategyConfig
from tradecopilot.evaluator import EvaluationMetrics, ReplayEvaluator, ReplayOutcome
from tradecopilot.models import DecisionState, MarketFrame, TradePlan
from tradecopilot.providers.replay import ReplayProvider
from tradecopilot.strategy import DecisionEngine

EASTERN = ZoneInfo("America/New_York")


class NightlyBacktestReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    generated_at: datetime
    strategy_version: str
    replay_files: tuple[str, ...]
    metrics: EvaluationMetrics
    incomplete_signals: int
    data_quality_concerns: tuple[str, ...]
    result: str


class ReplayBacktestRunner:
    """Measure completed signal outcomes in event order; never inspect a future frame."""

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    async def run(self, paths: tuple[Path, ...]) -> NightlyBacktestReport:
        outcomes: list[ReplayOutcome] = []
        incomplete = 0
        concerns: list[str] = []
        for path in sorted(paths):
            path_outcomes, path_incomplete = await self._run_path(path)
            outcomes.extend(path_outcomes)
            incomplete += path_incomplete
        if not paths:
            concerns.append("No replay fixtures were found")
        if incomplete:
            concerns.append(f"{incomplete} BUY signal(s) had no later SELL in the same replay")
        metrics = ReplayEvaluator().evaluate(outcomes)
        if metrics.signals < 100 or metrics.sessions < 20:
            concerns.append("Promotion evidence gate requires at least 100 unseen signals across 20 sessions")
        return NightlyBacktestReport(
            generated_at=datetime.now(UTC),
            strategy_version=self.config.version,
            replay_files=tuple(str(path) for path in sorted(paths)),
            metrics=metrics,
            incomplete_signals=incomplete,
            data_quality_concerns=tuple(concerns),
            result="NO_CHANGE" if not concerns else "MORE_DATA",
        )

    async def _run_path(self, path: Path) -> tuple[list[ReplayOutcome], int]:
        engine = DecisionEngine(self.config)
        active: _ActiveSignal | None = None
        outcomes: list[ReplayOutcome] = []
        async for frame in ReplayProvider(path, speed=0).frames():
            decision = engine.evaluate(frame)
            if decision.state == DecisionState.BUY and decision.trade_plan is not None and active is None:
                rvol = decision.indicator.source_style_rvol if decision.indicator is not None else None
                active = _ActiveSignal.from_frame(frame, decision.trade_plan, rvol)
            if active is None or frame.quote is None:
                continue
            active.observe(frame)
            if decision.state == DecisionState.SELL:
                outcomes.append(active.finish(frame))
                active = None
        return outcomes, int(active is not None)


class _ActiveSignal:
    def __init__(self, frame: MarketFrame, plan: TradePlan, rvol: Decimal | None) -> None:
        assert frame.quote is not None
        self.plan = plan
        self.signal_time = frame.event_time
        self.session = frame.event_time.astimezone(EASTERN).date()
        self.entry = plan.trigger_price
        self.risk = plan.risk_per_share
        self.mfe_r = Decimal(0)
        self.mae_r = Decimal(0)
        self.spread = frame.quote.ask - frame.quote.bid
        self.halted = frame.halted
        self.time_bucket = _time_bucket(frame.event_time)
        self.price_bucket = _price_bucket(self.entry)
        shares = frame.float_evidence.shares if frame.float_evidence else None
        self.float_bucket = _float_bucket(shares)
        self.gap_bucket = _gap_bucket(frame.gap_percent)
        self.rvol_bucket = _rvol_bucket(rvol)

    @classmethod
    def from_frame(cls, frame: MarketFrame, plan: TradePlan, rvol: Decimal | None) -> _ActiveSignal:
        return cls(frame, plan, rvol)

    def observe(self, frame: MarketFrame) -> None:
        if frame.quote is None:
            return
        current_r = (frame.quote.last - self.entry) / self.risk
        self.mfe_r = max(self.mfe_r, current_r)
        self.mae_r = min(self.mae_r, current_r)
        self.halted = self.halted or frame.halted

    def finish(self, frame: MarketFrame) -> ReplayOutcome:
        assert frame.quote is not None
        realized_r = (frame.quote.last - self.entry) / self.risk
        return ReplayOutcome(
            signal_time=self.signal_time,
            session=self.session,
            realized_r=realized_r,
            mfe_r=self.mfe_r,
            mae_r=self.mae_r,
            spread_dollars=self.spread,
            slippage_dollars=self.plan.estimated_slippage,
            false_breakout=frame.quote.last < self.plan.trigger_price,
            halted=self.halted,
            failed_fill=None,
            skipped=False,
            time_bucket=self.time_bucket,
            price_bucket=self.price_bucket,
            float_bucket=self.float_bucket,
            gap_bucket=self.gap_bucket,
            rvol_bucket=self.rvol_bucket,
        )


def write_nightly_report(report: NightlyBacktestReport, output_directory: Path) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    stamp = report.generated_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = output_directory / f"nightly-{stamp}.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(target)
    return target


def report_summary(report: NightlyBacktestReport) -> str:
    return json.dumps(
        {
            "result": report.result,
            "signals": report.metrics.signals,
            "sessions": report.metrics.sessions,
            "expectancy_r": str(report.metrics.expectancy_r),
            "maximum_drawdown_r": str(report.metrics.maximum_drawdown_r),
            "concerns": report.data_quality_concerns,
        },
        separators=(",", ":"),
    )


def _time_bucket(timestamp: datetime) -> str:
    local = timestamp.astimezone(EASTERN)
    return "open" if (local.hour, local.minute) < (10, 0) else "late"


def _price_bucket(price: Decimal) -> str:
    if price < 5:
        return "under_5"
    if price <= 10:
        return "5_to_10"
    if price <= 20:
        return "10_to_20"
    return "over_20"


def _float_bucket(shares: int | None) -> str:
    if shares is None:
        return "unknown"
    if shares < 5_000_000:
        return "under_5m"
    if shares < 10_000_000:
        return "5m_to_10m"
    return "10m_to_20m"


def _gap_bucket(gap: Decimal | None) -> str:
    if gap is None:
        return "unknown"
    if gap < 2:
        return "under_2"
    if gap < 10:
        return "2_to_10"
    return "over_10"


def _rvol_bucket(rvol: Decimal | None) -> str:
    if rvol is None:
        return "unknown"
    if rvol < 5:
        return "under_5"
    if rvol < 10:
        return "5_to_10"
    return "over_10"
