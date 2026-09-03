# Controlled strategy research

Self-improvement means after-hours, versioned parameter research. It never
means live weight updates, autonomous code rewriting, or self-promotion.

`research propose` creates one candidate with exactly one changed mutable
parameter. It records old/new value, hypothesis, causal rationale, chronological
training/validation/out-of-sample windows, metrics, regressions, data-quality
concerns, overfitting risks, shadow plan, rollback condition, and approval
status. Immutable controls include manual-only execution, risk ceilings,
verified float, four-of-five pillars, 50% retracement, 2R, the strict
first-pullback setup, and the day-stop lock mechanism. The cutoff time itself is
researchable only with user-specific evidence.

`ReplayEvaluator` sorts signals chronologically and creates walk-forward
windows—never random splits. Metrics include spread, slippage, observed/unknown
failed fills, halts, MFE, MAE, expectancy in R, profit factor, maximum drawdown,
worst-decile loss, false-breakout rate, skipped signals, and stability by time,
price, float, gap, and RVOL buckets.

Run the nightly evaluator manually or schedule this exact command with the
operating system after the market closes:

```bash
uv run tradecopilot research nightly \
  --replay-dir examples --output-dir .tradecopilot/research
```

The command reads completed replay files in chronological order, measures only
outcomes that occur after each signal, and writes a timestamped JSON report.
It returns `MORE_DATA` until the minimum evidence gate is met. Scheduling is an
external operational step; the Python process does not silently install a cron
job or promote a candidate.

Promotion approval requires all of:

1. One mutable variable changed.
2. At least 100 unseen signals.
3. At least 20 out-of-sample sessions.
4. Improvement in at least two chronological windows.
5. Meaningful net-expectancy improvement (initial gate: +0.05R).
6. No material drawdown or worst-decile regression.
7. At least 10 shadow sessions.
8. No recorded regressions.
9. Explicit `strategy promote CANDIDATE --human-confirm`.

The command records human approval in the journal but intentionally does not
rewrite the active configuration. Applying a new production configuration and
its rollback remains a separate human-controlled release step. Research status
is one of `NO_CHANGE`, `REJECT`, `MORE_DATA`, `SHADOW_TEST`, or `HUMAN_REVIEW`;
the research job cannot produce an automatic production promotion.
