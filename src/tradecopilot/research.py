from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from tradecopilot.config import StrategyConfig
from tradecopilot.journal import Journal
from tradecopilot.models import DataQuality, ExperimentCandidate

IMMUTABLE_PARAMETERS = frozenset(
    {
        "strategy_id",
        "version",
        "source_video_id",
        "deployment_status",
        "execution_mode",
        "origins",
        "minimum_source_style_rvol",
        "minimum_percentage_gain",
        "maximum_risk_per_trade_usd",
        "maximum_daily_loss_usd",
        "maximum_consecutive_losses",
        "maximum_float_shares",
        "minimum_pillars",
        "source_price_minimum",
        "source_price_maximum",
        "maximum_retracement_fraction",
        "minimum_reward_risk",
    }
)


class ResearchController:
    def __init__(self, journal: Journal, config: StrategyConfig) -> None:
        self.journal = journal
        self.config = config

    def propose(
        self,
        parameter: str,
        new_value: str,
        hypothesis: str,
        causal_rationale: str,
    ) -> ExperimentCandidate:
        if parameter in IMMUTABLE_PARAMETERS:
            raise ValueError(f"immutable strategy parameter: {parameter}")
        values = self.config.model_dump(mode="json")
        if parameter not in values:
            raise ValueError(f"unknown strategy parameter: {parameter}")
        now = datetime.now(UTC)
        candidate = ExperimentCandidate(
            provider_timestamp=now,
            receipt_timestamp=now,
            age_seconds=0,
            source="after_hours_research",
            quality=DataQuality.INSUFFICIENT,
            candidate_id=str(uuid4()),
            base_version=self.config.version,
            candidate_version=f"{self.config.version}+candidate",
            changed_parameter=parameter,
            old_value=str(values[parameter]),
            new_value=new_value,
            hypothesis=hypothesis,
            causal_rationale=causal_rationale,
            training_window="unset",
            validation_window="unset",
            out_of_sample_window="unset",
            baseline_metrics={},
            candidate_metrics={},
            regressions=(),
            data_quality_concerns=("No walk-forward evaluation has run",),
            overfitting_risks=("Candidate has not been tested out of sample",),
            shadow_test_plan="Run chronological walk-forward evaluation, then 10 shadow sessions",
            rollback_condition="Any material expectancy, drawdown, or tail-loss regression",
            unseen_signals=0,
            out_of_sample_sessions=0,
            improved_chronological_windows=0,
            shadow_sessions=0,
            human_approval_status="not_approved",
            research_result="MORE_DATA",
        )
        self.journal.save_experiment(candidate)
        self.journal.flush()
        return candidate

    def promotion_gate(self, candidate: ExperimentCandidate, *, human_confirm: bool) -> str:
        if not human_confirm:
            return "HUMAN_REVIEW"
        baseline_expectancy = candidate.baseline_metrics.get("expectancy_r")
        candidate_expectancy = candidate.candidate_metrics.get("expectancy_r")
        baseline_drawdown = candidate.baseline_metrics.get("maximum_drawdown_r")
        candidate_drawdown = candidate.candidate_metrics.get("maximum_drawdown_r")
        baseline_tail = candidate.baseline_metrics.get("worst_decile_loss_r")
        candidate_tail = candidate.candidate_metrics.get("worst_decile_loss_r")
        metrics_present = None not in {
            baseline_expectancy,
            candidate_expectancy,
            baseline_drawdown,
            candidate_drawdown,
            baseline_tail,
            candidate_tail,
        }
        expectancy_improved = bool(
            metrics_present
            and candidate_expectancy is not None
            and baseline_expectancy is not None
            and candidate_expectancy >= baseline_expectancy + 0.05
        )
        drawdown_safe = bool(
            metrics_present
            and candidate_drawdown is not None
            and baseline_drawdown is not None
            and candidate_drawdown <= max(baseline_drawdown * 1.10, baseline_drawdown + 0.25)
        )
        tail_safe = bool(
            metrics_present
            and candidate_tail is not None
            and baseline_tail is not None
            and candidate_tail >= min(baseline_tail * 1.10, baseline_tail - 0.25)
        )
        gates = (
            candidate.changed_parameter not in IMMUTABLE_PARAMETERS,
            candidate.quality == DataQuality.GOOD,
            not candidate.data_quality_concerns,
            candidate.training_window != "unset",
            candidate.validation_window != "unset",
            candidate.out_of_sample_window != "unset",
            candidate.unseen_signals >= 100,
            candidate.out_of_sample_sessions >= 20,
            candidate.improved_chronological_windows >= 2,
            candidate.shadow_sessions >= 10,
            not candidate.regressions,
            expectancy_improved,
            drawdown_safe,
            tail_safe,
        )
        return "HUMAN_REVIEW" if all(gates) else "MORE_DATA"

    def promote(self, candidate: ExperimentCandidate, *, human_confirm: bool) -> ExperimentCandidate:
        gate = self.promotion_gate(candidate, human_confirm=human_confirm)
        if gate != "HUMAN_REVIEW" or not human_confirm:
            raise PermissionError(f"candidate cannot be promoted: {gate}")
        approved = candidate.model_copy(
            update={
                "human_approval_status": "approved_for_manual_configuration",
                "research_result": "HUMAN_REVIEW",
            }
        )
        self.journal.save_experiment(approved)
        self.journal.flush()
        return approved
