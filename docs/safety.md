# Safety and live authentication

Tradecopilot is analysis-only. `BUY` and `SELL` are deterministic labels shown
to a human; they are not broker instructions. There is no order form, order
transport, Alpaca trading client, or Robinhood mutation method in the runtime.

## Robinhood boundary

The backend owns the complete authenticated MCP session. Application code sees
only `ReadOnlyToolSurface`, which intersects live discovery with this reviewed
allowlist:

```text
get_accounts
get_portfolio
get_realized_pnl
get_pnl_trade_history
search
get_equity_historicals
get_equity_fundamentals
get_financials
get_equity_price_book
get_equity_technical_indicators
get_earnings_results
get_earnings_calendar
get_indexes
get_index_quotes
get_equity_positions
get_equity_tax_lots
get_equity_quotes
get_equity_orders
get_equity_tradability
get_scans
get_scanner_filter_specs
run_scan
```

Unknown tools are denied before transport invocation. The 19 discovered write
or order-workflow methods are explicitly blocked, including every equity and
option review/place/cancel method and every watchlist/scanner mutation. Tests
prove both properties and do not call a blocked tool.

`get_equity_orders` is informational only. The application can use position
and read-only order changes to recognize a trade the user placed manually. It
cannot place, replace, modify, review, or cancel that trade.

## Provider authority

- Alpaca is authoritative for live trades, quotes, one-minute bars, and raw
  time-and-sales prints. Only `alpaca.data` is imported; `alpaca.trading` is not.
- Robinhood is authoritative for positions/cost basis, account P&L, displayed
  Level 2, tradability, saved scan reads, and informational orders.
- Shibui is authoritative only for float, daily ADV50 inputs, ownership
  context, fundamentals, and SEC filing metadata.
- Shibui prices can never overwrite Alpaca live prices or bars.
- A verified catalyst needs a separate sourced feed or the manual CLI fields.

Alpaca trade messages in this implementation do not reliably classify the
aggressor side. The prints are useful as raw tape, but red-tape burst and hidden
seller confirmation remain `LIMITED`, never neutral or confirmed. The strategy
continues with candle, quote, spread, and repeated Level 2 evidence.

## Credentials and OAuth

Run:

```bash
uv run tradecopilot auth alpaca
uv run tradecopilot auth robinhood
```

Alpaca keys and Robinhood OAuth client/token material are stored in the
operating-system keychain. They are not stored in source, `.env`, SQLite,
browser storage, logs, prompts, or fixtures. Robinhood OAuth uses the official
MCP SDK, opens the authorization URL in the desktop browser, and receives the
authorization code through a one-use loopback callback on `127.0.0.1`.

`tradecopilot logout PROVIDER` clears the corresponding local keychain entry.
Remote Robinhood token revocation is not implemented because no documented
revocation endpoint was established; revoke the authorization in Robinhood if
remote revocation is required.

Current local status on 2026-08-10:

- Shibui read-only float/ADV50 enrichment passed a live doctor smoke.
- The Robinhood MCP is registered, but the browser callback was not completed
  during the final smoke, so the application keychain has no confirmed token.
- Alpaca credentials were not supplied, so no live Alpaca request was made.
- Replay/mock mode is fully operational without any provider credential.

## Doctor behavior

`tradecopilot doctor` never initiates OAuth. Without keychain credentials it
reports the live provider as `DEGRADED`. After credentials exist, it performs
only reviewed reads: tool inventory, accounts, quote/OHLCV, Level 2, position,
P&L, Alpaca quote/trade/bars, and Shibui float/daily context. Pass an account
suffix only at runtime:

```bash
uv run tradecopilot doctor --symbol AAPL \
  --account-last4 LAST4 --alpaca-feed sip
```

Full account identifiers remain inside the backend call boundary. The browser,
journal, explanations, and diagnostic messages receive only a nickname or
masked alias.

## Browser and LLM boundary

The visual server binds to `127.0.0.1`. It exposes sanitized snapshot, health,
SSE, symbol selection, and chat routes plus static assets. It has no account,
credential, order, watchlist, scanner-mutation, or transfer route. Live symbol
selection changes only the read-only provider target and clears old ticker
caches before the next frame.

The optional OpenAI agent receives a reduced deterministic decision context and
no MCP client, credentials, account state, positions, orders, or tools. It
cannot change state, trigger, stop, share size, R:R, freshness, or session lock.
Conflicting or invented output is rejected and replaced with deterministic
fallback text. A ChatGPT subscription cannot authenticate this server-side API
integration; a separately billed API key is required.

## Risk and degraded modes

- Missing critical data produces `DATA_INSUFFICIENT`.
- Required quote or book older than two seconds produces `DATA_STALE`.
- Composite Alpaca quote age uses the older of the quote and last-trade
  timestamps, so one fresh component cannot hide another stale component.
- Stale data while exposed never produces `HOLD`; the app displays the last
  structural stop and says manual/broker-side controls must govern.
- Bounded retries, backoff/jitter, circuit breaking, per-resource locks,
  snapshot deduplication, cancellation-safe shutdown, and journal flushing
  prevent runaway polling and duplicate transitions.
- `DAY_STOP` persists by Eastern trading date and blocks every new `BUY`. A
  structural `SELL` alert remains permitted for an open position; live locks do
  not auto-reset.
- Research never rewrites or promotes the active strategy.
- Browser/webhook alerts carry `manual_execution=true` and no broker secret.

Secret-like fields, bearer tokens, long numeric identifiers, and full account
identifiers are redacted before persistence. Never add credentials or account
numbers to configuration, replays, tests, prompts, or Git.
