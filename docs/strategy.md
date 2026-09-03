# Ross first-pullback strategy

## Version

```yaml
strategy_id: ross_first_pullback
version: 1.0.0-candidate
source_video_id: xGIa8Vg0PWM
deployment_status: shadow
execution_mode: manual_only
entry_mode: strict
```

The engine requires verified float below 20 million shares and at least four
of five pillars: source-style RVOL >= 5x, gain >= 10%, verified catalyst (or a
strict market-leader exception), source-band price context, and verified float.
The price pillar accepts a stock that originated between $2 and $20 even if it
later moves above $20. Market capitalization is never treated as float.

`source_style_rvol` is current session volume divided by 50-day average full-day
volume and controls the pillar. `time_adjusted_rvol` is stored separately only
when same-time history exists.

## Setup and trigger

The pluggable `ImpulseDetector` scans a bounded recent window for a clear
multi-bar move. `PullbackDetector` requires two to four trailing red bars,
calculates `(impulse_high - pullback_low) / (impulse_high - impulse_low)`, and
accepts exactly 50% but rejects anything greater. Average red pullback volume
must be lower than average green impulse volume. Debug fields include exact bar
indexes and timestamps classified as impulse and pullback.

A setup must remain above VWAP and EMA9, avoid a clear five-minute downtrend,
have acceptable spread/liquidity, preserve the pullback low, and offer at least
2R before resistance. The strict trigger is the immediately preceding pullback
candle's high. `ARMED` occurs below it; `BUY` requires a newly observed intrabar
cross from below with fresh quote/book data and acceptable breakout volume. A
candle close is not required. Anticipatory entries are disabled.

The stop is one configured buffer below pullback low. Risk per share equals
entry minus stop plus estimated slippage; shares are floored so maximum dollar
risk cannot be exceeded. Stops cannot widen and averaging down is rejected.

## Hold and exit

`HOLD` requires explicit evidence that stop, latest meaningful higher low,
thesis, progress, spread/liquidity, and confirmed exit state remain intact.
Touching 1R, 2R, a half-dollar, a whole-dollar, or a target does not itself sell.

`EXIT_WARNING` covers an initial topping wick plus slowing progress.
`SELL` covers structural stop/higher-low break, unreclaimed false breakout,
strong-volume VWAP loss, no follow-through, sharp bid disappearance/spread
expansion, unsafe tradability, or confirmed repeated seller/tape evidence.
One flashing order cannot confirm persistence. Hidden seller and red-tape burst
remain unavailable without real time-and-sales.

After `SELL`, a flat position plus a fresh independent impulse/pullback, new
structural low, new strict trigger, new 2R opportunity, and fresh/liquid data are
required for `REENTRY_WATCH`. A rebound alone is not enough.

## Threshold provenance

Source-derived fields:

- 5x source-style RVOL, 10% minimum gain, 30% preferred gain.
- Verified float below 20 million and at least four of five pillars.
- Approximate $2-$20 source band; preferred $5-$10 origin.
- Maximum 50% retracement, above VWAP/EMA9, first-candle-new-high trigger,
  stop below pullback low, and minimum 2R opportunity.
- 10:00 a.m. America/New_York personal cutoff, kept as an initial shadow rule.

Implementation safety defaults, not claims from the source:

- $25 trade risk, $75 daily loss, three consecutive losses.
- 1.5% maximum spread; 1.5 ATR maximum extension from structural support.
- Two-second quote/book freshness, 90-second one-minute-bar freshness, $0.01
  stop buffer, slippage buffer.
- Eight-bar impulse lookback, three-bar minimum impulse, 6% minimum impulse.
- Two-to-four pullback bars, 0.80 breakout/pullback volume ratio.
- 45% severe upper-wick fraction; three-bar no-follow-through timeout.
- 4x median seller size, three snapshots, two seconds persistence.
- Five-minute opening range and extended-hours VWAP.

These ambiguous implementation defaults are configuration fields, labeled
`implementation_default`, and eligible for one-variable shadow research. They
must not tune themselves or promote themselves.

RSI, MACD, ATR, and EMA crossover are context only and never independently
produce `BUY` or `SELL`.
