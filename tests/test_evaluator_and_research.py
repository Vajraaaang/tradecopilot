from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from tradecopilot.config import StrategyConfig
from tradecopilot.evaluator import ReplayEvaluator, ReplayOutcome
from tradecopilot.journal import Journal
from tradecopilot.models import DataQuality
from tradecopilot.research import ResearchController


def _outcome(day: int, realized_r: str) -> ReplayOutcome:
    session = date(2026, 7, day)
    return ReplayOutcome(
        signal_time=datetime(2026, 7, day, 13, 30, tzinfo=UTC),
        session=session,
        realized_r=Decimal(realized_r),
        mfe_r=Decimal("1.2"),
        mae_r=Decimal("-0.4"),
        spread_dollars=Decimal("0.02"),
        slippage_dollars=Decimal("0.01"),
        time_bucket="open",
    )


def test_evaluator_uses_chronological_walk_forward_windows() -> None:
    outcomes = [_outcome(day, "1" if day % 2 else "-0.5") for day in range(1, 7)]
    outcomes.reverse()
    windows = ReplayEvaluator().walk_forward(
        outcomes,
        training_sessions=2,
        validation_sessions=1,
        out_of_sample_sessions=1,
    )
    assert windows[0].training_sessions == (date(2026, 7, 1), date(2026, 7, 2))
    assert windows[0].validation_sessions == (date(2026, 7, 3),)
    assert windows[0].out_of_sample_sessions == (date(2026, 7, 4),)
    assert windows[0].out_of_sample_metrics.expectancy_r == Decimal("-0.5")


def test_research_never_promotes_without_human_and_all_gates(tmp_path) -> None:
    config = StrategyConfig()
    with Journal(tmp_path / "journal.sqlite3") as journal:
        controller = ResearchController(journal, config)
        candidate = controller.propose("maximum_spread_percentage", "1.25", "tighter spread", "reduce slippage")
        with pytest.raises(PermissionError):
            controller.promote(candidate, human_confirm=False)
        with pytest.raises(PermissionError):
            controller.promote(candidate, human_confirm=True)


def test_promotion_records_approval_but_does_not_rewrite_configuration(tmp_path) -> None:
    config = StrategyConfig()
    now = datetime.now(UTC)
    with Journal(tmp_path / "journal.sqlite3") as journal:
        controller = ResearchController(journal, config)
        candidate = controller.propose(
            "maximum_spread_percentage", "1.25", "tighter spread", "reduce slippage"
        ).model_copy(
            update={
                "provider_timestamp": now,
                "receipt_timestamp": now,
                "quality": DataQuality.GOOD,
                "training_window": "2026-01-01/2026-03-31",
                "validation_window": "2026-04-01/2026-04-30",
                "out_of_sample_window": "2026-05-01/2026-06-30",
                "data_quality_concerns": (),
                "unseen_signals": 100,
                "out_of_sample_sessions": 20,
                "improved_chronological_windows": 2,
                "shadow_sessions": 10,
                "baseline_metrics": {
                    "expectancy_r": 0.10,
                    "maximum_drawdown_r": 5.0,
                    "worst_decile_loss_r": -1.0,
                },
                "candidate_metrics": {
                    "expectancy_r": 0.16,
                    "maximum_drawdown_r": 5.2,
                    "worst_decile_loss_r": -1.05,
                },
                "regressions": (),
            }
        )
        approved = controller.promote(candidate, human_confirm=True)
    assert approved.human_approval_status == "approved_for_manual_configuration"
    assert config.maximum_spread_percentage == Decimal("1.5")
