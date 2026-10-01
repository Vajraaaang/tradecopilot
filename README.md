# tradecopilot

`tradecopilot` is a deterministic terminal monitor and local visual desk for
the Ross first-pullback low-float momentum setup. It is analysis and alerting
software only: it never places, previews/reviews, modifies, replaces, or
cancels an order. Every action in Robinhood remains manual.

The visual desk follows the supplied Robinhood Legend workspace and the
supplied Solana color reference. Its feather and four chart-tool SVGs use exact
path data read from Robinhood's public Legend page; no authenticated Legend
application code or account data is embedded.

## Install

Requirements: [uv](https://docs.astral.sh/uv/) and Node.js 24 LTS. `uv`
installs the project Python runtime. Node 24 is prepared for the dashboard, but
the current static frontend has no npm install step.

```bash
cd /Users/vajraang/tradecopilot
source ~/.nvm/nvm.sh
nvm use 24
uv sync --all-groups --extra openai
cp .env.example .env
```

Replay and deterministic explanations require no broker credentials or OpenAI
key. OpenAI explanations and chat require separately billed API access; a
ChatGPT subscription cannot be linked to the API process.

## Jev intraday adviser

Jev is an optional second opinion in the visual desk. Click **Ask Jev** to
evaluate the current live setup. The panel shows BUY / HOLD / SELL / WAIT
probabilities, confidence, the source timestamp, and usage. WAIT means remaining
flat; low confidence produces **uncertain**, not HOLD. These are model judgments
about the supplied intraday setup, not validated probabilities of profit or
multi-day price forecasts. Existing strategy, position and risk guards remain in
control; Jev cannot place trades or change the strategy's signals.

```bash
# Run from the checkout containing the Jev changes.
TRADECOPILOT_PROJECT="$(git rev-parse --show-toplevel)"

# Enter the key at the hidden prompt; it is stored in the OS keychain.
uv run --project "$TRADECOPILOT_PROJECT" tradecopilot auth jev

# Optional: one small, paid API diagnostic using synthetic data.
uv run --project "$TRADECOPILOT_PROJECT" tradecopilot jev-check

# Requires the existing Alpaca market-data and Robinhood read-only setup.
uv run --project "$TRADECOPILOT_PROJECT" tradecopilot serve \
  --mode live --symbol AAPL --alpaca-feed iex --jev
```

`TYPESAFE_API_KEY` in the server environment takes precedence over the keychain.
The key is never sent to the browser or saved in the repository. Use
`tradecopilot logout jev` to remove the stored key; unset the environment variable
as well if it is configured. Jev needs its own TypeSafe API credit, independent
of OpenAI or ChatGPT.

The trial allows **100 attempted requests total across restarts**, with a
30-second cooldown, a 16,000-byte request limit, caching of identical recent
requests, and zero automatic retries. Failed/timeout attempts count toward the
limit. Merely opening or polling the dashboard spends nothing; replay/mock mode,
stale or missing data, DAY_STOP and confirmed exits cannot trigger Jev requests.
The usage ledger is shared across working directories and `--db` values in the
operating system user's `Library/Application Support/tradecopilot/jev.sqlite3`.
Keep that ledger to preserve the cap; this local cap does not cover usage in
other TypeSafe clients or on other machines.

The model is pinned to `jev-1.13.0`. TypeSafe currently lists **$0.042 per million
input tokens**, with free output tokens. At the model's 64,000-token maximum,
100 requests would be approximately **$0.269** at that price. Normal bounded
requests are much smaller. The displayed cost is an estimate from provider
usage, reserving the full context allowance for failed or unfinished calls;
it is not a live reading of the account balance. Sources:
[models and pricing](https://docs.typesafe.ai/models),
[HTTP API](https://docs.typesafe.ai/api),
[confidence](https://docs.typesafe.ai/confidence).

The initial confidence threshold (0.6) is a conservative implementation default,
not a threshold calibrated against stock returns. Request snapshots, model and
prompt versions, probabilities, usage and timestamps are recorded in the local
ledger for later analysis. The existing strategy backtest does not measure Jev
accuracy. Validate it on timestamped held-out market data with a defined outcome
horizon and trading costs before treating its judgments as predictive evidence.
No Deep Research report is required to connect the API; it can help define that
separate evaluation once the trading horizon and data source are fixed.

## Run

```bash
# Visual replay application. Opens http://127.0.0.1:8765/.
uv run tradecopilot serve --mode replay \
  --replay examples/yxt_replay.jsonl --speed 10

# Terminal simulator: run this before attempting live integration.
uv run tradecopilot replay examples/yxt_replay.jsonl --speed 10 --compact

# Network-free visual mock application.
uv run tradecopilot app --mode mock --replay examples/yxt_replay.jsonl

# Optional structured state-change explanations and GPT chat.
uv sync --extra openai
export OPENAI_API_KEY="your-api-key"
export TRADECOPILOT_EXPLANATION_MODE=openai
export TRADECOPILOT_CHAT_MODE=openai
export TRADECOPILOT_OPENAI_MODEL=gpt-5.6
export TRADECOPILOT_OPENAI_REASONING_EFFORT=high
uv run tradecopilot serve --mode replay \
  --replay examples/yxt_replay.jsonl --speed 10

# Store provider credentials only in the macOS Keychain. The Robinhood command
# opens the official MCP OAuth page; neither command stores secrets in .env.
uv run tradecopilot auth alpaca
uv run tradecopilot auth robinhood

# Safe local checks. The account suffix is not persisted or logged.
uv run tradecopilot doctor --symbol AAPL --account-last4 LAST4 --alpaca-feed sip

# Separate persistent DAY_STOP demonstration.
uv run tradecopilot replay examples/day_stop_replay.jsonl --speed 0 --compact

# Send sanitized state changes to an HTTPS phone/push automation webhook.
export TRADECOPILOT_ALERT_WEBHOOK_URL=https://alerts.example/your-private-endpoint

# Run chronological replay evaluation after market close. This never promotes.
uv run tradecopilot research nightly \
  --replay-dir examples --output-dir .tradecopilot/research

# Rich mock terminal, using the synthetic YXT fixture.
uv run tradecopilot monitor YXT --mode mock

# Live visual desk and terminal monitor. Both are analysis-only and contain no
# review/place/replace/cancel order path.
uv run tradecopilot serve --mode live --symbol AAPL \
  --account-last4 LAST4 --alpaca-feed sip --scan-title "Warrior screen"
uv run tradecopilot monitor AAPL --mode live \
  --account-last4 LAST4 --alpaca-feed sip --risk-usd 25 --daily-loss-usd 75

# Manual evidence is accepted only with sources.
uv run tradecopilot monitor SBFM --mode live \
  --account-last4 LAST4 --alpaca-feed sip \
  --risk-usd 25 --daily-loss-usd 75 \
  --float-shares 2870000 --float-source "verified provider name" \
  --catalyst "brief factual catalyst" \
  --catalyst-source "company release" \
  --catalyst-time "2026-08-10T08:30:00-04:00"

uv run tradecopilot report --date 2026-08-10
uv run tradecopilot research propose \
  --parameter maximum_spread_percentage --value 1.25 \
  --hypothesis "Tighter spreads reduce slippage" \
  --rationale "Compare one variable in chronological windows"
uv run tradecopilot strategy list
uv run tradecopilot strategy compare 1.0.0-candidate CANDIDATE_ID
uv run tradecopilot strategy promote CANDIDATE_ID --human-confirm
```

The replay visibly transitions through:

```text
WATCH -> ARMED -> BUY -> HOLD -> EXIT_WARNING -> SELL -> REENTRY_WATCH
```

`BUY` and `SELL` are analysis states, never broker commands. The app monitors
only while its process is running. Live mode uses an application-owned official
MCP client and fails closed until Alpaca keys and Robinhood OAuth material are
present in the OS keychain. The final local smoke could not authenticate
Robinhood because its browser callback was not completed; replay remains fully
usable. See [docs/safety.md](docs/safety.md).

## Visual desk

The local visual app provides:

- automatically rotating 1-minute and 5-minute candlestick views;
- interactive candle hover/crosshair readouts, click-to-pin inspection, wheel
  zoom centered on the cursor, horizontal trackpad or drag panning, keyboard
  pan/zoom, bottom navigation controls, and double-click reset;
- TradingView-inspired light chart chrome with candle/line chart modes, linear
  or logarithmic price scale, visible grid and axes, full-app fullscreen, and
  drawing undo/clear;
- session VWAP, EMA9, EMA20, volume, RSI, MACD, trigger, and structural-stop
  overlays, with unavailable calculations labeled instead of inferred;
- current/recent normalized positions with deterministic HOLD, warning, exit,
  or reentry feedback;
- a non-mutating Robinhood saved-scan screener enriched with Shibui float and
  50-day daily-volume context;
- a tool-free chat input configured to request `gpt-5.6` with high reasoning;
  model access is account-dependent and was not API-smoke-tested here;
- the same SQLite journal and session-lock behavior as the terminal monitor.

Select `ENABLE ALERTS` in the visual app to grant browser notifications.
`TRADECOPILOT_ALERT_WEBHOOK_URL` sends the same sanitized transition event to a
user-controlled HTTPS webhook suitable for a phone notification service. No
account identifier, position, order, or broker credential is included.

Selecting `1m` or `5m` pauses automatic rotation. Select `AUTO ROTATE OFF` to
resume it. The visual shell uses a moving pale-violet background and one
restrained glass system for navigation, positions, screener, chat, and risk
surfaces. The TradingView-inspired chart remains an opaque white analytical
canvas. Motion is disabled when the operating system requests reduced motion.
The web server binds only to `127.0.0.1`; stop it with Ctrl-C.

Without `OPENAI_API_KEY`, the interface labels GPT `OFFLINE` and disables the
chat field instead of presenting predefined deterministic text as a model
response. The API key stays server-side. The model receives no MCP client,
positions, account data, order data, secrets, or tools. A response that changes
the deterministic state or invents a price is discarded.

The top bar shows the current regular-market status in Pacific time and labels
the replay timestamp separately in Pacific and Eastern time. Replay states are
historical simulation results; they are never presented as current-market
signals.

## Supplemental data and Shibui Finance

Shibui Finance is registered in Codex and supported by the Python provider at
`https://mcp.shibui.finance/mcp`. Its database is daily—not intraday—so
tradecopilot uses `ownership_stats.shares_float`, 50 completed daily-volume
rows, and recent SEC filing metadata. Shibui never supplies current price,
intraday bars, trigger detection, VWAP, spread, or Level 2.

```bash
codex mcp add shibui --url https://mcp.shibui.finance/mcp
export TRADECOPILOT_SHIBUI_MCP_URL=https://mcp.shibui.finance/mcp
uv run tradecopilot doctor
```

Shibui is not a breaking-news or time-and-sales source. A separate sourced JSON
feed boundary is implemented for those fields; see
[`examples/supplemental.example.json`](examples/supplemental.example.json).
Every record needs a source and timezone-aware timestamp. The checked-in file
is synthetic and must not be treated as live evidence.

## Verification

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```

Architecture, data rules, and operations:

- [Architecture](docs/architecture.md)
- [Strategy and threshold provenance](docs/strategy.md)
- [Safety and live authentication](docs/safety.md)
- [Robinhood tool inventory](docs/robinhood-tool-inventory.md)
- [Shibui Finance boundary](docs/shibui.md)
- [Replay format](docs/replay-format.md)
- [Controlled research](docs/self-improvement.md)

## Defaults and limitations

- Strategy `ross_first_pullback`, version `1.0.0-candidate`, shadow,
  `manual_only`.
- Maximum risk per trade: $25; daily loss: $75; consecutive-loss lock: 3.
- Quote and Level 2 maximum age: 2 seconds; latest one-minute bar: 90 seconds.
- Eastern new-entry cutoff: 10:00 a.m.
- Verified float is mandatory; catalyst must be sourced unless the explicit
  market-leader exception passes.
- Alpaca supplies streaming trades, NBBO quotes, minute bars, and raw trade
  prints. SIP is recommended for consolidated coverage; IEX remains supported.
- Robinhood supplies read-only positions/cost basis, P&L, displayed Level 2,
  tradability, saved-scan execution, and informational order history.
- Alpaca trade messages do not provide a reliable aggressor side in this
  integration. Tape is labeled `LIMITED`; hidden-seller and red-burst signals
  cannot be confirmed from those unclassified prints.
- Shibui Finance supplies float, 50-day daily-volume context, and filing
  metadata. It cannot supply current prices, live candles, Level 2, news, or tape.
- A sourced breaking-news provider is not configured. Manual catalyst evidence
  requires description, source, and timezone-aware timestamp.
- All open Robinhood equity positions are listed and reconciled, but full
  VWAP/EMA/stop/exit analysis runs only for the selected ticker. Select another
  holding in the symbol field to retarget the deterministic monitor.
- Robinhood scanner rows are discovery candidates, not strategy signals. A row
  remains `DATA_INSUFFICIENT` until selected and evaluated with fresh live data.
- `logout robinhood` clears local keychain material; remote OAuth revocation is
  not implemented and must be completed in Robinhood if required.
- SQLite raw observation retention defaults to 10,000 rows.
- The optional OpenAI explainer/chat receives sanitized decisions and no tools;
  the deterministic engine remains authoritative. Without the optional extra
  and API key, terminal explanations use deterministic templates while visual
  GPT chat is explicitly disabled.
- The requested `gpt-5.6` model identifier is configurable, not guaranteed by
  this build. If the API account does not expose it, set
  `TRADECOPILOT_OPENAI_MODEL` to an available reasoning model.
