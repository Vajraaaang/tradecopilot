# Kronos and Alpaca paper integration implementation plan

> Execute the approved forecasting milestone with scoped ownership, test-driven development and independent review.

Goal: use the existing Alpaca paper credentials for account/data integration, run real local Kronos inference, compare identical historical cases against simple controls, and show forecast paths in a verified local dashboard.

Architecture: an explicit paper account/data client supplies validated completed candles. Pinned upstream Kronos code and pinned local safetensors checkpoints provide optional CPU inference. A bounded study and report viewer record timestamps, lineage, failures and actual metrics. Existing Jev and neural results remain recorded; no trading-order policy is introduced by this milestone.

Scope and defaults:
- Paper account URL is fixed to https://paper-api.alpaca.markets; stock data is fetched separately from https://data.alpaca.markets. Credentials are loaded from the existing Keychain and never serialized or displayed.
- Account, clock and candle operations use GET only. Historical data source/feed is explicit. IEX has been verified for current access; SIP is a separate explicit option.
- First local models: Kronos-mini and Kronos-small with matching tokenizers, pinned revisions and local safetensors. Use CPU first; measure throughput before a larger run. No fine-tuning or Jev calls.
- Model context: 60 completed same-session consecutive minute candles; future timestamps from XNYS, converted into America/New_York wall time for model temporal features while artifacts retain UTC.
- Sampling: preserve individual forecast paths, rather than claiming the upstream averaged path is an uncertainty distribution. Raw sample intervals are uncalibrated.
- Development cohort: AAPL/AMZN/MSFT/NVDA/TSLA, completed October 1–6, 2026 sessions; anchors open+61,+121,+181,+241,+301 minutes; exact +15m target. Freeze sampler/budget and common timestamp eligibility before reading model scores. All exclusions and failures remain counted. This is a development integration benchmark, not fresh confirmation.
- Baselines: last-close persistence and last-five-minute linear momentum, with the same cases/horizon. Report price error in bps, direction with inclusive ±10bp FLAT, sample-interval coverage, failures and runtime. No 80% or profitability claim.
- Market is closed at this start. A next-session forecast, if recorded, is explicitly named overnight plus the first fifteen regular-session candles. It must not be presented as a fifteen-minute wall-clock forecast. Outcomes remain pending until observed.

Task 1 — paper account/data client
Files: src/tradecopilot/forecast/paper.py; tests/test_forecast_paper.py; minimal HistoricalBar IEX source enum extension in forecast/bars.py.
- [ ] Write failing tests for fixed paper/data hosts, GET-only calls, credential redaction, account validation, clock handling, paginated IEX/SIP candles, duplicates/order/interval/session/future-candle filtering, and HTTP failures.
- [ ] Implement a small PaperSnapshot and asynchronous read-only snapshot function using Keychain credentials and bounded httpx calls.
- [ ] Verify synthetic contracts and a root-run real account/clock/candle probe. Account/data proof contains no account ID, balances or credentials.

Task 2 — pinned local Kronos engine
Files: forecast/kronos.py; tests/test_forecast_kronos.py; pinned vendor source/manifest/license; explicit setup script.
- [ ] Pin upstream source/checkpoint revisions and hashes. Inspect before execution; use safetensors only, local files for inference, no automatic provider access in offline tests.
- [ ] Write failing tests for candle/timestamp input contract, matching tokenizer/model/context, deterministic seeded individual paths, missing optional dependencies, finite output shapes, provenance, and no outcome inputs.
- [ ] Implement optional engine and setup command. Run one real inference for each checkpoint; record latency and output diagnostics before freezing development budget.

Task 3 — bounded development study and paper forecast
Files: forecast/kronos_study.py; CLI integration; focused tests.
- [ ] Freeze input-only cases and settings; reuse validated candles; record every planned exclusion.
- [ ] Run both models and controls without outcome fields entering inference. Score after predictions, report all cases/failures, preserve source/model/config identities.
- [ ] Persist a genuinely prospective next-session record separately from historical scores; provide a later outcome-join command without overwriting original forecasts.

Task 4 — report viewer and delivery
Files: forecast/kronos_dashboard.html; narrow server/CLI integration; report/service tests; docs and derived charts.
- [ ] Validate immutable report inventory; serve only exact read-only routes and generic errors.
- [ ] Display paper-account mode/feed, historical vs pending prospective evidence, mean/sample band/actual price paths, all model/control metrics and failures.
- [ ] Browser-check actual results; run focused and full regressions, Ruff/mypy, source review and independent numerical replay.
- [ ] Publish only derived evidence and documentation with reproduction commands; keep licensed rows, checkpoints and credentials local. Verify hosted checks on the published commit.
