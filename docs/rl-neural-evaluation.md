# Offline neural policy evaluation

The user requested real evaluation of the network design. This implements a bounded comparison of small feedforward PPO, larger feedforward PPO and RecurrentPPO, rather than selecting a network from its size alone. Results are trading-policy outcomes in an OHLCV replay simulator; they are not Jev forecast accuracy. Hosted Jev weights are not trained.

## Registered protocol

Registration `5bb6992bed6be525687269c4e0b12c0e962676647cb12b3fd4a79ff41c1de183` was frozen before source acquisition or quality scoring. The previously inspected April–September dates are excluded. Historical SIP/raw minute bars cover AAPL, AMZN, MSFT, NFLX and TSLA. Independent source audit reconciled 413,141 downloaded rows to 195,150 regular-session rows and 217,991 exclusions, verified all 42 acquisition pages and every retained record, and found zero missing regular-session minutes. Availability is assumed at bar end in this retrospective replay; actual acquisition times are retained separately. November 3 is a reference session; November 4–January 30 has 60 training dates, February 2–27 has 19 tuning dates, and March 2–30 has 21 test dates. Dates belong wholly to one split, including all five symbols. Test data is inaccessible through the prepared-data loader until the tuning selection seal exists.

| Candidate | Actor / critic | Parameters |
| --- | --- | ---: |
| PPO 64 | Separate 64→64 networks | 24,964 |
| PPO 256 | Separate 256→256 networks | 198,148 |
| Recurrent PPO 256 | Separate one-layer 256-unit LSTMs, then 256→128 | 986,372 |

All candidates receive the same 127-dimensional observation: 55 market features, their 55 missingness masks, five symbol indicators and 12 ledger/timing fields. Imputation and normalization fit training data only. There are no Jev forecasts or probability features in this MARKET_ONLY experiment.

The architectures have different parameter counts, so this measures the complete candidates rather than isolating the causal effect of recurrence at matched capacity. Two-layer LSTMs and broader optimizer searches were not tested.

Each architecture trains seeds 42, 43 and 44 for 92,160 steps (7,200 optimizer epochs) on CPU with the same PPO settings: one environment, rollout/batch 128, ten epochs, learning rate 0.0003, gamma 0.99, GAE 0.95 and clipping 0.2. Each seed is limited to 600 seconds with a process supervisor. Incomplete or unequal-budget runs stop selection instead of receiving a quality ranking. Recurrent state starts at zero, carries within a replay, and resets across symbol/session boundaries.

CPU and MPS throughput probes used only training episodes and discarded their weights. CPU achieved approximately 2,892 / 2,609 / 192 steps per second across the three candidates, versus 241 / 322 / 68 on MPS. Both produced matching deterministic actions and recurrent state on the fixed parity probe. CPU won every throughput comparison on the user's M3 Max / 36 GiB machine. This is a measured local benchmark, not a universal hardware recommendation.

Architecture selection uses average February date-level net return over all three seeds. The checkpoint is the best February seed within the winning architecture. Both choices are hashed before March test scoring. All three seeds of the selected architecture are scored on March alongside cash, capped intraday hold and a fixed five-minute momentum rule. The strongest control is chosen on February, not March. Costs of 4 and 8 bps per side stress the primary checkpoint and that control; base cost is 2 bps.

## Execution and acceptance

Each independent symbol/session starts with $10,000 virtual cash and a $1,000 notional budget. Actions select flat, half or full exposure; repeated nonzero targets hold rather than repurchase. After 60 minutes of past-only warmup, decisions use a common two-minute grid and a full-minute delay before a hypothetical opening fill. Capacity is bounded by 1% of volume known at decision time. Exact Decimal accounting includes adverse fill costs, fee/quantity conventions, partial fills, gap-safe share liquidation and a $100 loss lock. Missing/late mandatory inputs truncate; unresolved positions block improvement claims. These fill/capacity assumptions are synthetic, not verified executable bid/ask liquidity.

A successful comparison requires all scored episodes valid and resolved, positive primary test net return, positive paired date-bootstrap lower bounds against cash and the February-selected control, drawdown within the registered limit, and positive incremental return at doubled costs. The simulator's delayed loss lock is not a guaranteed loss bound. All outcomes remain in private ledgers; invalid episodes cannot be silently discarded.

Date-level returns equally average five independently funded symbol episodes and reset capital daily. They do not represent one continuous multiasset brokerage account. Reports preserve turnover, costs, trades and seed variation; no annualized operational profit or 80% forecast-accuracy claim is implied.

## Reproduction

Install the optional neural stack with `uv sync --frozen --group dev --extra rl --extra plots`. The default application and Docker image do not install PyTorch. Prepare verified SIP/raw bars with `scripts/prepare_rl_data.py`, then run `scripts/run_rl_capacity_study.py` with absolute prepared-data, registration, budget and fresh output paths. Raw bars, arrays, individual ledgers and trusted local checkpoints remain private. Registration and budget identities, data/normalizer hashes, package/lock/script digests, versions and checkpoint checksums are verified before tuning and again before locked scoring. `load_report` verifies every saved artifact.

No provider clients, Keychain handles or broker tools enter the rollout path. This experiment spends zero Jev credit and submits no broker orders. A later Jev-feature comparison requires eligible out-of-fold forecast caches and a fresh registered confirmation window.
