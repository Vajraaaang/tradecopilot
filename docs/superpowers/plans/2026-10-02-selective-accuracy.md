# Selective Accuracy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Follow test-first tasks with scoped ownership and review.

**Goal:** Run a reproducible, honest 80% selective-accuracy experiment on expanded historical minute data.

**Architecture:** Reuse immutable HistoricalBar, ForecastExample and OHLCV JSON artifacts. Add one authenticated read-only historical importer, one statistical selection module and one offline experiment runner; preserve old defaults and reports.

**Tech Stack:** Python, httpx, pydantic, NumPy, scikit-learn, exchange-calendars, pytest; existing pinned dependencies.

## Tasks
- [x] Historical importer: create `forecast/alpaca_history.py`, `scripts/download_alpaca_history.py`, `tests/test_forecast_alpaca_history.py`. Test pagination, inclusive/exclusive dates, OHLC validity, duplicate handling, regular-session filtering, bounded pages, source hashes, missing credentials and sanitized HTTP failures before implementing. Return validated bars and metadata; preserve raw responses privately. CLI loads Keychain only and uses fixed `https://data.alpaca.markets/v2/stocks/bars` endpoint, no trading endpoint.
- [x] Model settings: modify `forecast/ohlcv_models.py`, test in existing model tests. Add bounded keyword-only C/L2 candidate settings while retaining identical defaults; export actual settings. Test unsupported parameters and portable parity.
- [x] Statistical primitives: create `forecast/selective.py`, `tests/test_forecast_selective.py`. Test 100-session chronological split, horizon purge, three expanding folds, separate temperature fitting, fixed gate search and fail-closed selection. For boundary invariant assert `max(r.label_observed_at for r in train) < min(r.as_of for r in next_block)`. For no qualifying gate assert `policy['enabled'] is False`; validate probability shape/finite/normalization and row alignment.
- [x] Runner: create `forecast/selective_study.py`, `scripts/run_selective_study.py`, integration tests. Convert validated bars to replay observations with prior-session close proxy, use original dataset builder and OHLCV feature builder, select tuning loss without accessing test labels, persist frozen selection then final scoring. Assertions: disjoint session IDs, no old-study dates, identical candidate cohorts, final selection unaffected by changed test labels, immutable output refusal.
- [x] Evidence/docs: run full tests/Ruff/mypy/package builds; independent spec then code review; run real expanded study only with configured free credentials; publish aggregate report/plots if actually executed. Update README and factual development docs, create stacked PR based on OHLCV branch, attach it and verify hosted CI. No fabricated score when data missing.

Commands: `.venv/bin/pytest tests/test_forecast_alpaca_history.py tests/test_forecast_selective.py tests/test_forecast_selective_study.py tests/test_forecast_ohlcv_models.py`, `.venv/bin/ruff check .`, `.venv/bin/mypy src`, `.venv/bin/pytest`. All checks must pass before publication.

Software implementation and independent reviews complete. 575 tests, Ruff/types
(64 files) and package builds pass. The real importer was attempted and stopped
before requests because Alpaca Keychain credentials are absent. Real-data
confirmation and publication of new performance charts remain pending credentials;
no Jev calls, default changes or 80% claim were made.

October 3 follow-up: approved paper credentials stored and verified in Keychain;
44 raw-page hashes and 200,850 regular bars verified. Real frozen run completed
with 188,700 cases and 18,500 final tests; selected model46.51% vs46.18% control,
no qualifying80% gate. Actual aggregate charts published; independent real-evidence
audit passed. Zero Jev calls/trades and no production promotion. RL remains proposed.
