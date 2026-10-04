# Jev forecasting and offline RL architecture

**Status: proposed design, October 3, 2026. No RL environment or policy has been implemented or trained.** Four parallel specialists planned the environment, Jev integration, evaluation and platform boundaries. Implementation begins with the bounded foundation below after design review.

## Evidence and objective

The real SIP study retained 200,850 minute bars across 103 dates and produced 188,700 labeled cases. The selected OHLCV logistic forecaster scored 46.51% across 18,500 final-test cases versus 46.18% for the price-only control. No gate met the 80% selective-accuracy criteria. These are measured retrospective forecasting results, not evidence of profitable trading or improved Jev performance. See [the completed experiment](selective-accuracy.md).

There are two independently evaluated objectives:

1. **Forecast lab:** improve calibrated probabilities for the existing 15-minute, ±10-bps DOWN/FLAT/UP task. Preserve accuracy, log loss, Brier, calibration, per-direction precision and coverage scorecards.
2. **Policy lab:** learn target exposure from causal market state and validated forecasts, optimizing simulated net portfolio outcomes under explicit execution costs and risk constraints.

RL trains our policy, not the hosted Jev weights. [TypeSafe states](https://docs.typesafe.ai/models) that customer fine-tuning/LoRA is unavailable; domain customization uses state, typed questions and downstream combination. Its [known numeric/date/context limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13) motivate deterministic arithmetic, temporal filtering and bounded summaries in our own code.

## System boundary

```mermaid
flowchart LR
    D[Verified historical bars] --> F[Causal features]
    F --> C[CPU / Jev forecast lab]
    C --> S[Immutable forecast cache]
    F --> E[Offline Gymnasium environment]
    S --> E
    E --> B[Deterministic policies and PPO]
    B --> V[Chronological evaluation]
    V --> R[Verified aggregate reports]
```

Existing bar stores, calendars, OHLCV calculations, JSON forecast models, hashing and report integrity checks are reused. Provider clients and Keychain handles belong only in data/forecast preparation. An environment receives verified local arrays and metadata; `reset` and `step` cannot access HTTP clients, inference callbacks or broker objects. Cache misses never trigger provider calls. This milestone creates no broker orders and changes no production recommendation defaults.

## Foundation: causal simulator before policy training

One symbol and regular XNYS session form an episode. Every symbol on a date belongs to the same split. Previous-session references are allowed; current-session warmup uses only completed bars. Start with 60 minutes of current-session warmup and explicit missing flags for any still-incomplete window. Start each episode with virtual cash, no position, and no carried policy memory. Initial research is intraday and unlevered; overnight exposure and multiasset shared-capital allocation are later designs.

### Observation contract

A fixed-order Float32 vector contains locally computed OHLCV summaries, missingness bits, normalized ledger state, current allocation/position, `last_committed_target_level`, liquidation-required/rejection state, drawdown, data age, time to session close, and risk-lock state. A label-free as-of adapter reuses existing OHLCV calculations. It must never expose `ForecastExample.target_price`, `label`, `label_observed_at`, future returns or a complete labeled example to a policy.

Train imputation/scaling only on the permitted training block; freeze its feature order, statistics and hash. Optional forecasts supply probability vectors, empirical calibration metadata, age and availability masks. Missing probabilities are masked features with an explicit fallback, not fabricated confidence. Initial MARKET_ONLY replay is valid environment evidence. CPU and Jev profiles are separately identified so a proxy cannot be reported as Jev.

### Action and order contract

`Discrete(3)` selects flat, half-cap or full-cap target exposure. The cap is a separately configured notional limit, not buying power or a stop-loss guarantee. BUY, HOLD and SELL are display translations of actual increases, unchanged holdings and reductions. Shorting and borrowing are prohibited.

A target-level change sizes an order from the ledger at decision time. `last_committed_target_level` is initialized to flat and is part of the observation. A valid risk-projected command updates this latch at commitment, including a later rejected or partially filled order; the canceled remainder never silently retries. Actual holdings, latch, last fill/rejection and risk lock are all observable, so equal holdings with different command histories cannot hide different transitions. Invalid actions do not update it.

Repeating a nonzero committed target is HOLD/no resize, not another purchase or a fee-driven micro-rebalance. FLAT with residual shares always submits another bounded liquidation attempt, and mandatory risk/terminal liquidation continues until flat or unresolved episode end. Risk projection can independently require a reduction. A later continuous-rebalance/retry command is a separately versioned action design. Record requested target, risk-projected command, latch, commitment, fill and rejection reason.

Buys commit a maximum dollar notional using decision-time cash/marks. Reductions commit a bounded share quantity using decision-time marks; FLAT and forced exits commit all shares currently held, not a dollar estimate. At fill, clamp sales to actual held shares and capacity. A ten-share flat order still requests ten shares after either an upward or downward price gap; partial liquidation preserves remaining shares and its unresolved status. The notional cap governs entry/resize commitments and execution-time increase checks. Market movement can create temporary marked exposure above the budget; report it and apply delayed reduction rather than claiming a continuous cap guarantee.

### Timing and fills

Use a predeclared regular decision grid, common to every policy in a scenario. The default cadence is two minutes. Separate `anchor_end_at`, `decision_at`, `execution_at` and `next_observation_at`; never derive readiness from bar end alone. At decision tick `t`, every consumed market/forecast input must have `available_at <= t` under its declared availability mode. A late/missing required current market anchor is a data failure: submit no order, truncate with the last safe marked ledger and explicit unresolved/invalid status. Do not move the decision backward, silently select another profitable episode, or fill on a past open. Optional late forecasts are masked and use the frozen fallback; they do not change the common clock.

With an eligible anchor at `t`, commit the action, wait one full minute, simulate a fill at the open of the bar starting `t+60s`, and return the next observation at `t+120s` only if its required bar/fill-report inputs are available by then. Example: a 09:30–09:31 bar available at 09:31 supports a 09:32 fill and 09:33 next observation. If that anchor arrives at 09:32:30, there is no 09:31 decision or 09:32 fill. An execution bar or next required observation arriving after the scheduled transition produces a declared truncated/unresolved replay, not retroactive fill knowledge. Latency stress scenarios use a separately frozen cadence of latency plus one complete bar. A later event-clock/order-queue design requires its own timing specification.

The simulator may use the future execution bar internally to produce the fill/reward, but the policy cannot observe it before acting. `simulated_fill_at` and `fill_report_available_at` remain separate. Opening fills are explicitly hypothetical OHLCV replay assumptions, not observed executable bid/ask or queue priority.

Use adverse synthetic spread/slippage and explicit fees. Initial capacity uses a declared fraction of volume already known at decision time; this is a synthetic bound, not opening liquidity. Partial fills obey cash/capacity/quantity precision. Cancel unfilled remainder in the foundation rather than introducing hidden queued orders. Missing/zero-volume execution bars produce no invented fill; there is no price interpolation. Existing holdings still bear price movement during the delay.

### Ledger, reward and episode end

Use an exact Decimal ledger with declared currency/quantity rounding. Cash, shares, average cost, realized/unrealized P&L, execution drag and explicit fees must reconcile. Negative cash or shares is an invariant failure. Marks may use the last available price with a stale flag, but stale marks never establish liquidity or silently resolve an episode.

Foundation reward is additive net equity change in initial-equity basis points:

`reward = 10,000 × (equity_next − equity_previous) / initial_equity`.

Fill prices already contain spread/slippage and cash already deducts fees, so costs are not subtracted again. Total reward reconciles to final net return. Do not clip reward or add a changing drawdown/turnover penalty in this first reward version. Report drawdown, exposure and turnover separately; hard action/risk constraints operate independently.

A daily loss threshold locks increases and initiates delayed flattening. Gaps and missing liquidity can overshoot it; it is not a guaranteed maximum loss. Commit forced liquidation before the last executable window and charge closing costs once. Unfillable residual positions remain in the ledger and mark the episode unresolved. They cannot be dropped from denominators or counted as successful flat liquidation. Session completion is termination; an external time cutoff/data failure is truncation, following [Gymnasium's API](https://gymnasium.farama.org/api/env/).

### Proposed research defaults

These are assistant-selected fixture/research defaults, not user trading instructions or empirically optimized parameters. All are frozen in the environment manifest and configurable before a run.

| Parameter | Foundation proposal |
| --- | --- |
| Virtual initial equity | $10,000 per independent symbol episode |
| Notional cap | $1,000; actions target $0/$500/$1,000 |
| Daily loss-lock threshold | $100 from initial equity; delayed exits can overshoot |
| Quantity quantum | 0.000001 share, rounded down |
| Currency accounting | cents, deterministic declared rounding |
| Base synthetic spread/slippage | 2 bps per side; explicit fee zero |
| Cost sensitivity | 0 bps diagnostic; 5/10/20 bps per-side stress |
| Latency | one full minute; two-/five-minute stress |
| Synthetic capacity | 1% of decision-time known minute volume; 0.1%/5% stress |
| Randomness | fixed seed 42 for replay; all policy-training seeds later registered |

Independent symbol-episode capital must be stated in aggregate reports. It is not a shared five-asset portfolio; compare policies using identical capital/caps and equally weighted date-level outcomes.

## Jev forecasting workstream

Add `JevContextV2` while preserving current v1 defaults. Compute ratios, bps changes and labels in code. Supply roughly 15 normalized recent bars and compact 30/60-minute/volume/regime summaries, bucketed where appropriate for numeric robustness. Market/sector/news fields are admitted only after genuinely timestamped sources are acquired; missing fields stay unavailable.

Keep the existing direction Choice, tied to an explicit `target_definition` (minute-close proxy versus last-trade quote), horizon and boundary rule. Add one separately labeled adverse-move question only with a frozen observable price-event definition. Its result is a risk feature, not an executable stop-fill forecast. A proposed pilot uses a 20-bps future-low event; this is a design parameter, not an established optimum. Define the event as a strictly lower-than-threshold minimum over complete bars starting at/after as-of and ending by target time; equality is NO. Missing/late path coverage means unavailable outcome, not NO. Outcome records and their true/replay-assumed `label_observed_at` stay separate from label-free context. Noul responses use `{type: "noul", noul: number}` with finite `noul` in [0,1]; request criteria keys are `"true"`/`"false"`. There is no Noul `confidence` or Choice probability map.

Retrieve initially three nearest **earlier completed** analogues. Normalization/index vintages belong to the preceding training block. Require labels to mature before the query as-of and exclude the query/overlapping target windows. Never retrieve by future outcome. Retrospective ticker/date masking is a controlled ablation that mitigates recognizable-history exposure; it does not prove the provider never trained on historical outcomes.

An immutable forecast artifact separates `logical_as_of`, input availability, model/calibration fitting cutoffs, actual inference start/completion, simulated availability mode, prompt/model/context versions, provider confidence, empirical calibration and cost. Preserve actual `generated_at`; a separately named `replay_effective_available_at = logical_as_of + declared_simulated_latency` belongs only to an explicitly hypothetical profile with its assumption hash. Prospective mode requires actual completion/receipt before the decision. The teacher verifier validates fitting IDs/hashes against the split, not merely timestamp claims; unknown vendor pretraining cutoffs remain unknown. Provider confidence is [distribution concentration](https://docs.typesafe.ai/confidence), not measured success probability on our market task.

CPU features for RL training require expanding chronological out-of-fold predictions. Fit each forecaster on earlier matured labels, calibrate on a separate earlier block, then produce the subsequent episode-block cache. Exclude insufficient-history warmup blocks. The current final model fitted on the entire development period cannot generate causal early-development RL states. Historical Jev calls made today stay tagged retrospective; they cannot be relabeled as forecasts actually delivered historically.

First prepare complete offline CPU/market-only coverage. Sparse Jev pilot coverage is disclosed by symbol/date/regime, with fixed missingness masks and fallback; episodes cannot be selected for favorable Jev outcomes. A regularized CPU+Jev combiner needs adequate chronological teacher labels and out-of-fold stacking inputs. A tiny feasibility pilot is insufficient for learned fusion, 80% claims or distillation.

Existing durable protections remain: shared 100-attempt limit (15 observed attempts so far), ten requests/$0.05 per forecast run, request-size limits and failed-attempt accounting. Planning and environment development make zero Jev calls. A later reviewed pilot proposal is 20 representative predeclared cases, v1/v2 pairs plus ten masking/option-order checks, capped at 50 attempts and $0.15 aggregate across existing ten-call runs. It would remain within the observed remaining attempt budget; actual remaining limits must be rechecked atomically before starting. No quota expansion or pilot execution is authorized by this planning document.

## Evaluation and promotion gates

All existing April–September data is now inspected and usable for development only. Register the consumed September 1–30 dates explicitly; the older blacklist covered only September 16–30. Fresh confirmation requires a date manifest frozen before training/selection. A prospective candidate is the next 30 regular sessions beginning October 5, 2026, verified against XNYS; data collection/evaluation timing is a future step, not a scheduled task or executed benchmark. Before any policy training, persist a pretraining registration with exact dates/symbols, eligibility/missing-data rules, policies/seeds, costs, comparison selection and success criteria. Future bar hashes do not exist yet; bind them later in a separate acquisition artifact without editing the registration. Checkpoint/selection hashes must be frozen before confirmation outcomes are inspected. If implementation/tuning continues into that window or outcomes become inspected before selection is frozen, retire the affected window as development and register a new future block. Do not claim the proposed October window is already protected or active.

Group all symbols by date. Purge outcome/holding windows crossing split boundaries, including execution latency. Randomize training episodes only inside the training dates. Do not random-split adjacent minutes or let normalization/retrieval/calibration use evaluation outcomes. Foundation common controls are cash, capped intraday buy-and-hold and one frozen causal market-feature rule. Add the CPU forecast rule only after eligible caches exist, then compare PPO CPU with that rule; add Jev-based controls/PPO only when suitable caches exist. Every policy uses identical clocks, capital, execution, costs and risk projection.

Report forecast accuracy/proper scores/coverage independently from policy net returns, daily returns, drawdown, exposure, turnover, trades, rejections, terminal costs and unresolved episodes. Cash/all-flat Sharpe is N/A. Show all fixed training seeds and use paired date-level uncertainty intervals with symbols together; add contiguous-date sensitivity for serial dependence. Stress cost assumptions. Training reward is not the final policy success metric.

Readiness gates are separate:

1. **Environment ready:** accounting, causal timing, API compatibility, deterministic replay and future-mutation tests pass.
2. **Research executed:** every registered policy/seed/scenario completes with verified ledgers and honest coverage, including unsuccessful policies.
3. **Measured policy improvement:** every registered scored episode must be valid and fully resolved. Any invalid/unresolved episode makes that run ineligible for an improvement/promotion claim, with all outcomes retained in diagnostics; stale marked residuals are not liquidated net returns. Otherwise frozen criteria require positive net return, a positive paired lower 95% interval against cash and the strongest predeclared fixed rule, drawdown within the registered limit, and positive incremental return under doubled base costs. These are proposed research gates; a small sample may be inconclusive.
4. **Forecast goal:** the existing 80% accuracy/10% coverage/count/per-direction criterion remains independently unmet unless a new measured forecast experiment passes it.
5. **Operational deployment:** separate prospective paper execution would need actual receipt/fill reconciliation and risk-control validation, plus explicit authorization. No such execution is part of this plan.

## Platform and implementation boundaries

Use optional `sim` dependencies for Gymnasium only in the foundation; optional `rl` adds SB3/torch later. Default app and Docker keep minimal dependencies. Baseline pytest conditionally skips optional simulation tests when Gymnasium is absent; a dedicated `sim` CI job runs them and its checker. Optional imports are lazy and receive narrow explicit typing treatment, while dedicated simulation/RL jobs type-check those modules with installed stubs/dependencies. A minimal-install import/CLI smoke catches accidental optional imports. Replace baseline CI's current `--all-extras` before adding torch, selecting existing extras explicitly and a separate simulation/RL job. [SB3 recommends separate evaluation environments and multiple runs](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html).

The pure execution/accounting core stays independent of Gymnasium. A small Gym adapter owns `reset(seed, options)` and `step(action)`. Run bundles hash source, dependency lock, episode/split/feature/forecast/action/execution/reward schemas, normalization, seeds, budgets, checkpoints, full ledgers and aggregates. Mutable checkpoint staging stays separate from immutable reports. Checkpoint hashes establish integrity, not safe deserialization: only explicitly trusted locally produced policy artifacts may load, and report serving never loads policies.

## Registered policy-capacity comparison

Local hardware was read-only verified: Apple M3 Max, 38,654,705,664 bytes of RAM (36 GiB), 14 logical CPUs. Memory capacity supports considering the following candidates; training throughput and out-of-sample quality are not established.

| Candidate | Actor/critic configuration | Purpose |
| --- | --- | --- |
| PPO reference | Separate 64→64 heads | Small-network control |
| PPO capacity control | Separate 256→256 heads | Larger feedforward comparison |
| Recurrent PPO | Separate one-layer 256-unit actor and critic LSTMs, then 256→128 heads | Test temporal memory |
| Two-layer recurrent | Same hidden size, two layers | Deferred until development evidence justifies another registered comparison |

The recurrent proposal uses `MlpLstmPolicy`, `lstm_hidden_size=256`, `n_lstm_layers=1`, `shared_lstm=False`, `enable_critic_lstm=True`, and separate `pi`/`vf` heads `[256,128]`, supported by [SB3-Contrib](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html). Parameter counts differ between these candidates, so a result cannot isolate memory from capacity without an additional parameter-matched ablation.

Benchmark CPU first and MPS only if the pinned runtime reports support and all operators/parity checks pass. Use identical registered training inputs, include environment/optimizer time and device-transfer overhead, and separate warmup from steady-state measurements. Hardware availability does not establish faster training; [SB3's PPO guidance](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) particularly recommends CPU for non-CNN policies. The [PyTorch MPS backend](https://docs.pytorch.org/docs/2.14/notes/mps.html) is a capability to measure, not a quality advantage.

All three initial candidates share observations, dates, execution, reward, seeds and proposed optimizer settings: one environment, 128 rollout steps, batch size128, ten epochs, learning rate0.0003, gamma0.99, GAE0.95, clipping0.2. These are pretraining research proposals, not tuned values. A short throughput test on training/synthetic data sets a feasible common step budget before outcome-quality inspection, bounded by an initial100,000-step proposal and600-second hard cap per seed. Report actual steps, optimizer updates, elapsed time, memory, device and stopped runs. A time-stopped recurrent run is not labeled an equal-step comparison. Predeclare seeds42/43/44 and preserve every outcome.

Recurrent inference passes returned LSTM state and `episode_start` masks explicitly. Reset memory for every terminal/truncated episode and symbol/date switch; vectorized resets affect only ended lanes. Begin with zero recurrent state at the first eligible decision—no privileged extra warmup. Evaluation is deterministic with frozen training normalization; no training/tuning hidden state enters confirmation. Test uninterrupted versus carried-state replay, boundary isolation, vector lane independence, padding/reset masks, future mutation and checkpoint/normalization restoration.

CPU-first training follows simulator and forecast-lineage validation. No live provider calls, unbounded seed search, GPU reservation, arbitrary uploaded policy loading or broker client enters training. Distillation, multiasset books, paper execution and RL prompt selection require later measured justification and separately scoped designs.

[Phased implementation plan](superpowers/plans/2026-10-03-rl-system.md).
