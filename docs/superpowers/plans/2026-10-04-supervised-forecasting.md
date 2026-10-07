# Direct supervised forecasting implementation

> Execute with explicit file owners and independent review.

The goal is a measured supervised forecast comparison. PPO remains a downstream
trading-policy experiment; it does not update hosted Jev weights.

## Frozen comparison

The private registration is
`3ac13c643a6e22018273fd6abc2d43cd2a239b68a7ae00ce61ceabd87a26601d`.
Acquire Alpaca SIP/raw bars for AAPL, AMZN, MSFT, NVDA and TSLA from January 2
through June 12, 2025. January 2 supplies prior-session context. Use the following
110 calendar sessions, with separate TRAIN/TUNE/CAL/GATE/TEST blocks of
60/10/10/10/20 sessions. Recorded earlier study periods are retired. Preserve
parent manifests, raw-page hashes and existing acquisition limits when merging
the two contiguous downloads.

Keep the target fixed: exact minute-close return after 15 minutes, with inclusive
±10 bps FLAT. Anchors start 61 minutes after the session opens and repeat every
five minutes. Require 61 contiguous completed past bars and an exact target bar
inside the session. Publish all timestamp exclusions before model comparison.
No interpolation, candidate-specific drops or date replacement is allowed.

The neural input is a 60 × 6 causal sequence plus 115 static fields. The CPU
models use 55 OHLCV features and five symbol indicators. Fit preprocessing only
on TRAIN. Input identities must remain unchanged when outcome values change.

Compare the class prior, logistic C=0.01, logistic C=1 and a registered
31-leaf histogram booster with small TCN32 and LSTM64 classifiers. Neural
families average all three registered seeds (42, 43, 44); an incomplete seed
makes the family ineligible. Each fit has a 20-epoch and 600-second limit.

Choose the family and CPU reference by TUNE log loss. Freeze the models, fit one
CAL temperature, and select a GATE threshold under the existing 80% accuracy,
10% coverage and separate UP/DOWN precision requirements. Release TEST only
when selection is sealed. Report every complete candidate and failed fit, all
class diagnostics, coverage, and uncertainty resampled by whole session date.
Do not retune on TEST or automatically promote a model.

The separate Jev fusion comparison uses only the already paid July–October
cache. Those consumed dates provide exploratory evidence and cannot select the
new held-out model or establish confirmation. This milestone makes no new Jev
calls and places no broker orders.

## Ownership

| Owner | Files and responsibility |
| --- | --- |
| Data worker | `forecast/supervised_data.py`, preparation script and data tests: source merge, causal sequences, TRAIN preprocessing, case catalog and stage access |
| Model worker | `forecast/supervised_models.py`, seed learner and model tests: CPU baselines, TCN/LSTM, bounded training and safe JSON inference |
| Study worker | `forecast/supervised_study.py`, supervisor script and study tests: ensemble selection, calibration, gate, final metrics and artifact seals |
| Fusion worker | `forecast/jev_fusion.py`, fusion script and tests: matched context versus context-plus-Jev exploratory comparison |
| Root | Acquisition, frozen contracts, integration, actual runs, result rendering, documentation, CI and GitHub publication |
| Independent reviewers | Leakage, causality, model contracts, cohort completeness and recomputation of measured results |

## Completion checks

- Verify source boundaries, exact targets, case identities, TRAIN-only fitting,
  future-mutation invariance and stage guards with synthetic fixtures.
- Verify causal TCN padding, independent LSTM state, safe artifact loading,
  probability validity and CPU/native inference parity.
- Review and freeze code, dependencies and data before learning.
- Execute the registered comparison once and preserve target failures.
- Independently recompute predictions, metrics, exclusions and paired intervals.
- Publish aggregate results, charts and reproduction instructions; keep licensed
  bars, feature rows and private predictions local.
- Finish local checks and verify CI on the published commit.
