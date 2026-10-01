# Forecasting workflow

This workflow makes a price prediction testable. It does not execute trades.
The original strategy, backtest, and Jev market-opinion panel have separate
contracts; their BUY/HOLD/SELL judgments are not relabeled as measured forecasts.

## Offline demonstration

```bash
uv sync --frozen
uv run tradecopilot forecast demo --output-dir .tradecopilot/forecast/demo
uv run tradecopilot forecast serve .tradecopilot/forecast/demo/run/report.json --open
```

The generator uses eight synthetic XNYS sessions and five example symbols:
AAPL, MSFT, AMZN, GOOGL, NVDA. It simulates sampled last prices, not actual price
history or reconstructed OHLCV. No API key is loaded. Re-running the demo verifies
and reuses its existing immutable report; use a new directory for a new run.
`--config examples/forecast-config.json` makes the defaults explicit.

The local Jev fixture exercises the probability contract with a deterministic
heuristic. Its scores and zero API cost are not measurements of the Jev model.
The real Jev adapter uses a different execution label, pinned model and prompt,
and the existing shared usage ledger.

## Data and label contract

| Item | Initial setting |
| --- | --- |
| Target | Last-price return at a nominal 15-minute horizon |
| Classes | DOWN below −10 bps; FLAT from −10 through +10 bps; UP above +10 bps |
| Sessions | XNYS regular sessions, including holidays, DST and early closes |
| Input | Five minutes of observed prices; at least six distinct provider timestamps |
| Freshness | Latest source price no more than 30 seconds old at the input anchor |
| History continuity | Gaps no greater than 60 seconds |
| Return endpoints | Latest observation at/before the 1-minute and 5-minute endpoints, within 30 seconds |
| Outcome | First provider observation at/after target, received within target +60 seconds |
| Session boundary | Target/outcome must stay in the anchor's session; closing tick is eligible |
| Abstention | Maximum class probability and Jev's reported confidence must both reach 0.6 |

The outcome is a sampled-price proxy near the nominal target, not a claim that
the provider supplied an exact tick at that instant. Missing, stale, late, and
cross-session data are excluded or remain unscored; no outcome is invented.
The neutral band and confidence threshold are fixed implementation defaults,
not settings optimized on test returns.

Both provider time and receipt time must be at/before the forecast anchor.
Future outcome fields never enter model requests or feature vectors. The seven
features are 1- and 5-minute returns, 5-minute price range and return volatility,
change from previous close, source age, and history point count. There are no
news, order-book, volume, or account features in this price-only workflow.

Raw observations and sanitized operational events are appended to SQLite and
retained across collection restarts. Exact observations are deduplicated without
pruning. Input identities include their configuration and observation IDs and
remain stable when outcomes arrive later. Dataset exports include a manifest
and examples with verified hashes, versions and counts. The raw observation
hash binds the original corpus; raw rows remain in the source SQLite database,
which should be retained alongside the export for a full audit.

## Collect real observations

Store the existing Finnhub key in the OS keychain or set `FINNHUB_API_KEY` in the
server environment. Never put keys in a config JSON, Git, or a command argument.

```bash
uv run tradecopilot auth finnhub
uv run tradecopilot forecast collect --cycles 120 --interval 15 \
  --config examples/forecast-config.json
uv run tradecopilot forecast status
uv run tradecopilot forecast build --output-dir .tradecopilot/forecast/dataset-01
```

Collect only while regular sessions are open. Each invocation is bounded to
1–1600 cycles and can resume the same store on later sessions. Use `--store`
to choose its path. Collection attempts share a persistent pacing gate in that
store: at least 1.5 seconds between dispatches, including retries and restarts.
Do not run separate stores or other clients against the same provider key when
relying on this local rate limit. Transient failures allow at most three total
attempts; permanent response/authentication failures are not retried. Recorded
telemetry includes stale quotes, failures, retries, latency, and data age.

Finnhub supplies sampled last and daily prices, not minute candles or bid/ask.
Some responses can be stale even during market hours; a successful HTTP request
does not make them eligible forecast inputs. Free-tier availability and data
entitlements are controlled by the provider.

## Chronological experiments

```bash
uv run tradecopilot forecast experiment .tradecopilot/forecast/dataset-01 \
  --output-dir .tradecopilot/forecast/experiment-01
```

At least five distinct labeled sessions are required. Sessions are grouped
chronologically into roughly 60% train, 20% validation, and the remaining test
set, with at least three training sessions and one each for validation/test.
Training labels that arrive at/after validation starts and validation labels
that arrive at/after test starts are purged. All symbols from a session stay in
the same split. Five sessions are a software minimum, not a claim of statistical
sufficiency; evaluation across more sessions and market regimes remains needed.

The baselines are a Laplace-smoothed training class prior, a momentum-bucket
frequency model, and class-balanced logistic regression. Scaling uses only the
training period. A second logistic variant selects a temperature from a fixed
grid using validation log loss. The untouched test set reports accuracy, macro
F1, log loss, multiclass Brier score (sum over classes, range 0–2), confusion,
per-class reliability, and per-symbol scores.

All returned probability distributions, including abstentions, contribute to
proper probability scores. Coverage and selective accuracy separately show
which predictions passed the fixed confidence threshold. Missing predictions
and errors remain visible in the eligible denominator. Coverage curves are
descriptive; do not select a threshold from their test results and then quote
those same results as an unbiased benchmark.

Adjacent 15-minute targets overlap and are correlated. The report resamples
whole sessions for a 95% accuracy interval only when at least five test sessions
exist. A narrow interval would still not establish profitability or regime
robustness. This release implements a chronological holdout, not rolling retraining
or a transaction-cost-aware trading backtest.

Each run saves JSON model parameters, splits, all predictions, examples, dataset
metadata, source fingerprint and dependency versions. Artifacts contain data,
not executable pickle objects. Content verification detects accidental changes;
it is not a signature or proof of an external provider's authenticity. Dataset
IDs, model IDs and quality metrics reproduce for the same source/configuration;
wall-clock timestamps, measured latency, and report IDs naturally vary.

## Prospective Jev pilot

Keep the collector running in one terminal. In another, explicitly request one
paid forecast after enough fresh history exists:

```bash
uv run tradecopilot auth jev
uv run tradecopilot forecast predict AAPL --allow-live \
  --run-id pilot-01 --output-dir .tradecopilot/forecast/pilot-01

# After the target time and outcome receipt window, using the same observation store:
uv run tradecopilot forecast grade .tradecopilot/forecast/pilot-01 \
  --output-dir .tradecopilot/forecast/pilot-01-results
uv run tradecopilot forecast serve .tradecopilot/forecast/pilot-01-results/report.json --open
```

The original input is persisted before the request. Jev predicts DOWN/FLAT/UP
using the fixed target definition and causal feature allowlist. Successful
responses must arrive before the target; future anchors, stale inputs, malformed
probabilities and unexpected model versions are rejected. Grading uses the saved
input and later recorded observations. Opening a report never calls a model.

The adapter shares the existing global 100-attempt ledger, 30-second cooldown,
request-size limit and recent-request cache. Each run ID also has persistent
caps of at most 10 attempts and $0.05; `--max-requests` and `--max-cost-usd` can
lower them. The same run ID cannot expand its saved caps. Atomic reservations
cover the maximum 64,000 input tokens before calling; unknown failed attempts
retain that conservative charge. There are no automatic Jev retries. Cost is
an estimate at the configured provider rate, not an account-balance reading.
Preserve the ledger across restarts. This local cap cannot control other clients.

A small pilot demonstrates temporal plumbing and live API behavior. It is
reported separately from the offline benchmark and cannot establish whether Jev
outperforms the baselines. No customer fine-tuning of Jev is claimed. Calibrating
Jev itself requires a separate, sufficiently sized validation cohort; this release
only calibrates the CPU logistic baseline.

## Container and checks

```bash
docker build -t tradecopilot-forecast .
docker run --rm -p 127.0.0.1:8766:8766 tradecopilot-forecast
```

The default container builds the offline demonstration and serves its report as
an unprivileged user. It needs no keys and makes no inference calls. The report
server binds localhost by default; the container explicitly binds `0.0.0.0`
and the example publishes it only on host localhost. `/health` returns ready
only when the complete report bundle passes integrity verification. This is a
local demonstration, not an authenticated public hosting service.

```bash
uv run ruff check src tests
uv run mypy src/tradecopilot
uv run pytest
uv build
```

CI runs the offline checks and demo without secrets. Automated tests cover
causality, calendar edges, late labels, restart/retry behavior, artifact tampering,
calibration splits, budget concurrency and provider failures. Software tests and
synthetic results are distinct from evidence about real predictive performance.

## Accurate resume framing

Describe the implemented engineering work: a versioned forecasting pipeline,
typed Jev integration, budgeted inference, chronological evaluation, calibration,
reproducible artifacts, operational telemetry, containerization, and automated
validation. Do not claim a proprietary trained foundation model, profitable
trading, proven Jev accuracy, production scale, or a public deployment without
corresponding measured evidence.
