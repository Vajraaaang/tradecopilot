# Implementation history

This history describes the implemented repository changes. The source commits
and pull requests provide the corresponding code; the validation snapshot
records measured checks without treating a live API check as market accuracy.

## Original analysis desk

Initial commit [`d2a4328`](https://github.com/Vajraaaang/tradecopilot/commit/d2a4328)
introduced the deterministic momentum monitor, replay/mock paths, normalized
provider models, risk and session controls, journal, visual desk, and controlled
research workflow. Current behavior and provider boundaries are documented in
[architecture.md](architecture.md), [strategy.md](strategy.md), and
[safety.md](safety.md).

## Jev adviser and Finnhub integration — PR #1

[PR #1](https://github.com/Vajraaaang/tradecopilot/pull/1) integrated these changes
into `main`:

- [`a1592b0`](https://github.com/Vajraaaang/tradecopilot/commit/a1592b0): optional
  Jev intraday adviser, typed model responses, confidence/probability display,
  pinned model, OS-keychain setup, persistent 100-attempt trial ledger, cooldown,
  request limits and conservative usage estimates.
- [`0a0d124`](https://github.com/Vajraaaang/tradecopilot/commit/0a0d124): Finnhub
  price-only feed, provider/receipt timestamps, secure key setup, symbol selection
  and explicit limits for unavailable bid/ask and minute OHLCV.
- [`991cfee`](https://github.com/Vajraaaang/tradecopilot/commit/991cfee): independent
  Jev market opinions from available price context, so opinion generation does
  not require a brokerage position or a complete deterministic strategy frame.
  BUY/HOLD/SELL are informational leanings; the strategy retains its own gates.

## Forecast and evaluation platform — PR #2

[PR #2](https://github.com/Vajraaaang/tradecopilot/pull/2) adds a separate,
measurable future-price contract and the engineering needed to evaluate it.

[`5f846bb`](https://github.com/Vajraaaang/tradecopilot/commit/5f846bb) implements:

1. Durable observations and sanitized collection telemetry, exact deduplication,
   provider/receipt causality, XNYS calendars, causal price features, fixed per-run
   targets, mature outcome labeling, immutable dataset manifests and a synthetic
   offline generator.
2. Chronological session splits, outcome-time purging, class-prior/momentum/
   logistic baselines, training-only scaling, validation-only temperature
   calibration, proper probability scores, reliability/coverage diagnostics,
   per-symbol results and reproducible JSON artifacts.
3. Typed Jev UP/FLAT/DOWN forecasts, label-free requests, prospective timing,
   original-input/prediction persistence, exact record matching during grading,
   and atomic per-run request/spend caps within the shared adviser ledger.
4. CLI workflows for collection, dataset construction, experiments, paid
   prospective prediction, grading, status and report serving.
5. The saved-result dashboard, integrity-checked HTTP service, unprivileged
   Docker image, Compose configuration and offline CI/container validation.

[`fa6f539`](https://github.com/Vajraaaang/tradecopilot/commit/fa6f539) adds the
post-lock target-expiry guard and regression cases that verify zero paid calls
when a database wait reaches/passes the forecast target. It also installs all
optional dependencies in CI's type-check environment.

The follow-up documentation update adds the forecast architecture/module map,
dashboard screenshot, evidence summary, implementation history and accurate
repository presentation.

## Executed evidence and remaining evaluation

The [October 1 validation snapshot](validation-2026-10-01.md) records 426 passing
tests, lint/types, packaging, browser interactions, a container with external
networking disabled, and a three-request prospective Jev integration pilot.
The pilot cost estimate was $0.000093198 and all three later outcomes were
recorded; two forecasts abstained. The synthetic demonstration and live pilot
have different evidence labels.

A subsequent [real archived-minute study](historical-evaluation.md) imported
19,500 observations and evaluated 18,500 labeled examples with a 3,700-case
chronological CPU holdout. It also made ten budgeted retrospective Jev calls on
a frozen AAPL cohort and published aggregate results/figures with provider
attribution. Historical provenance, source hashes and replay availability are
explicit; historical inputs are rejected from prospective inference.

The study found weak predictive results and zero coverage at the fixed threshold.
Its importer and comparison paths bring the local suite to 455 passing tests.
Jev calibration on independent market data, multi-regime/prospective validation,
transaction-cost-aware trading evaluation, and public production deployment
remain uncompleted. They are not claimed as current accomplishments.

## Kronos + Alpaca paper integration — October 6, 2026

The [Kronos milestone](kronos-paper.md) restores the original candle-path
forecasting direction. It adds secure GET-only paper account/clock access and
explicit IEX/SIP data retrieval, a pinned licensed local Kronos adapter,
individual CPU sample paths, a fixed same-case development study, prospective
publication checks, later exact-candle grading, and a verified read-only viewer.
Independent reviews exposed and repaired publication timing, error-denominator
display, stale-context/control failure retention, and incomplete outcome-source
regrading. The previous Jev, neural and RL results remain unchanged.

No hosted Jev calls, fine-tuning, RL training or broker orders are part of this
milestone. Executed connection/inference/results and validation evidence are
recorded in the Kronos guide and its aggregate result assets.
