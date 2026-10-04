# Paid Jev-assisted recurrent policy evaluation

> For agentic workers: use subagent-driven-development with isolated ownership and independent review.

Goal: execute the user-authorized paid Jev comparison, not just a forecast stub or another proposal.

Architecture: reuse the causal simulator, prepared market arrays, recurrent architecture and budget callbacks. One same-as-of request per session supplies five masked stocks × 15m/60m/session-close forecasts. Cache actual generation times separately from explicitly simulated historical availability. CONTEXT_ONLY and JEV_ASSISTED have identical dimensions/metadata; only probabilities/confidence differ. Both profiles train seeds42/43/44, choose checkpoints on TUNE and release common TEST only after both selections freeze.

Scope and approvals: the preceding architecture covers cache-backed Jev policy experiments; the user now explicitly approved API spend. At most80 requests, conservatively $0.25, existing100-global/10-per-run/$0.05/cooldown/no-retry guards retained. New project-uninspected June30–October22,2025 source;60/10/10 chronological dates. Model/vendor training cutoff remains unknown; mask tickers/dates/absolute prices and label all evidence hypothetical retrospective. No brokerage actions or production promotion.

Ownership:
- Forecast cache worker: forecast/jev_rl.py, scripts/collect_jev_rl_cache.py, tests/test_jev_rl_cache.py. Label-free compact past context, typed three-horizon batches, atomic ledger/run-budget reservation, safe sanitized settlement and immutable cache artifacts. No actual paid calls by worker.
- RL worker: rl/jev_ablation.py, scripts/run_jev_rl_ablation.py, tests/test_rl_jev_ablation.py. Validate cache/base registration identities, causal masked augmentation, matched bounded learners, TUNE-only selection, locked TEST, exact scoring/stresses and integrity inventory. No actual learning/scoring by worker.
- Root: frozen registration, source download/market preparation, actual paid acquisition/training/evaluation, aggregate rendering/docs/CI/publication.
- Independent review: request causality, ledger atomicity/spend caps, masked neutral control fairness, temporal expiry, source/checkpoint identity and selection ordering; recompute actual results.

Tasks:
- [ ] Write meaningful failing tests for future-mutation isolation, invalid response/budget settlement, same-as-of batch, cache timing/expiry and locked TEST.
- [ ] Implement the two narrow modules without changing old forecast/NN defaults or adding dependencies; Ruff/strict mypy and focused tests.
- [ ] Review spec and code; freeze source/registration before paid requests and any training.
- [ ] Import/verify bounded SIP/raw source, prepare TRAIN-only scaling and freeze all80 request payloads before first paid call.
- [ ] Collect once under shared durable caps; preserve real request/completion/usage times; stop on provider failure rather than retry or silently backfill.
- [ ] TRAIN-only throughput establishes a common CPU step budget. All six equal-budget runs must complete before TUNE scores.
- [ ] Freeze both profile checkpoints before TEST; compare mean-seed returns, controls, 4/8bps stress, and separate horizon-specific forecast scores against train priors.
- [ ] Independent ledger/source/checkpoint recomputation and actual-spend reconciliation.
- [ ] Publish aggregate-only results/charts and verified code/CI in a stacked GitHub PR; preserve negative outcomes and all limitations.
