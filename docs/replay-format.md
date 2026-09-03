# Replay format

Replay is newline-delimited JSON. Every non-comment line is one event. Events
must be chronological and every timestamp must include a timezone. Provider
timestamps later than `event_time` are rejected as look-ahead.

```json
{
  "event_time": "2026-08-10T13:52:00Z",
  "bar_1m": {
    "timestamp": "2026-08-10T13:51:00Z",
    "symbol": "YXT",
    "open": "9.11",
    "high": "9.15",
    "low": "8.90",
    "close": "8.98",
    "volume": 150000
  },
  "quote": {
    "timestamp": "2026-08-10T13:52:00Z",
    "symbol": "YXT",
    "bid": "9.08",
    "ask": "9.10",
    "last": "9.09",
    "previous_close": "7.00",
    "total_volume": 6330000,
    "current_minute_volume": 110000,
    "session_origin_price": "7.00",
    "average_daily_volume_50d": 1000000
  }
}
```

The first event may contain `seed_1m` and `seed_5m` arrays. Later events append
`bar_1m`/`bar_5m`. Optional fields are `level2`, `time_and_sales`, `position`,
`account_risk`, `float_evidence`, `catalyst_evidence`, `resistance_levels`,
`gap_percent`, `market_leader`, `no_a_quality_candidates`,
`bearish_momentum_environment`, `tradability_known`, and `halted`. Evidence and
risk/position values persist until replaced; their age is recalculated at each
event. Bars and identical snapshots are deduplicated.

`examples/yxt_replay.jsonl` is synthetic behavior, not a representation of
historical fills. It demonstrates impulse, two-candle pullback, half-dollar
support context, strict crossing, continuation, topping-tail warning, repeated
seller confirmation, exit, and a later fresh reentry watch. The later rebound
does not revise the earlier `SELL`. `examples/day_stop_replay.jsonl` separately
demonstrates half-giveback lock persistence.

Playback uses event-time deltas divided by `--speed`; `--speed 0` runs without
wall-clock delay. No replay event can inspect a future line.

For the visual desk:

```bash
uv run tradecopilot app --mode replay \
  --replay examples/yxt_replay.jsonl --speed 10
```

Each normalized frame updates the same deterministic engine and journal used by
the terminal replay. The chart payload contains only bars already present at
that event time. The UI alternates the 1-minute and 5-minute views every eight
seconds; manual interval selection pauses the rotation.
