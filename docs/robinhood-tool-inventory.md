# Robinhood MCP tool inventory

Recorded on 2026-08-10 from authenticated Robinhood tool metadata: 53 tools
total, 34 read-only and 19 mutation/order-workflow tools. The standalone MCP is
now registered as `robinhood-trading`, but its final browser OAuth callback was
not completed, so the running application must re-list metadata after the user
authenticates. Descriptions and schemas below came from discovered metadata,
not invented client methods.

“Needed” means the selected-symbol monitor has a concrete use. “Allowed” means
the logical name is present in `READ_ONLY_ALLOWLIST`; every other name is denied
before transport invocation. Unknown future tools are denied by default.

| Exact tool name | Description | Class | Needed | Policy | Required arguments | Returned timestamps |
|---|---|---:|---:|---:|---|---|
| `add_option_to_watchlist` | Add option contracts to a watchlist | mutation | no | denied | `option_ids` | none documented |
| `add_to_watchlist` | Add symbols/assets to a watchlist | mutation | no | denied | `list_id` | none documented |
| `cancel_equity_order` | Cancel an equity order | mutation | no | denied | `account_number`, `order_id` | none documented |
| `cancel_option_exercise` | Cancel an option exercise request | mutation | no | denied | `account_number`, `option_id` | `created_at`, `updated_at` |
| `cancel_option_order` | Cancel an option order | mutation | no | denied | `account_number`, `order_id` | none documented |
| `create_scan` | Create a saved scanner | mutation | no | denied | `predicate`, `values` | none documented |
| `create_watchlist` | Create a watchlist | mutation | no | denied | `display_name` | none documented |
| `exercise_option` | Exercise an option contract | mutation | no | denied | `account_number`, `option_id`, `quantity` | `created_at`, `updated_at` |
| `follow_watchlist` | Follow a watchlist | mutation | no | denied | `list_id` | `created_at` |
| `get_accounts` | Read brokerage accounts | read-only | yes | allowed | none | none documented |
| `get_earnings_calendar` | Read earnings-calendar events | read-only | no | allowed | none | none documented |
| `get_earnings_results` | Read earnings results for a symbol | read-only | no | allowed | `symbol` | none documented |
| `get_equity_fundamentals` | Read equity fundamentals, including reported float/volume fields | read-only | yes | allowed | `symbols` | `market_date` |
| `get_equity_historicals` | Read equity OHLCV bars | read-only | yes | allowed | `start_time`, `symbols` | `begins_at` |
| `get_equity_orders` | Read equity-order visibility; informational only | read-only | yes | allowed | `account_number` | `created_at`, `last_transaction_at` |
| `get_equity_positions` | Read equity positions | read-only | yes | allowed | `account_number` | none documented |
| `get_equity_price_book` | Read displayed bid/ask book snapshots | read-only | yes | allowed | `symbols` | `updated_at` |
| `get_equity_quotes` | Read equity bid/ask/last quotes | read-only | yes | allowed | `symbols` | `venue_ask_time`, `venue_bid_time`, `venue_last_non_reg_trade_time`, `venue_last_trade_time` |
| `get_equity_tax_lots` | Read equity tax lots | read-only | no | allowed | `account_number`, `symbol` | none documented |
| `get_equity_technical_indicators` | Read provider-calculated technical indicators | read-only | no | allowed | `interval`, `start_time`, `symbol`, `type` | `begins_at` |
| `get_equity_tradability` | Read tradability/halt information | read-only | yes | allowed | `account_number`, `symbols` | `internal_halt_start_time`, `internal_halt_end_time` |
| `get_financials` | Read issuer financial statements | read-only | no | allowed | `symbols` | none documented |
| `get_index_historicals` | Read index historical bars | read-only | no | denied | `instrument_ids`, `interval`, `start_time` | `begins_at` |
| `get_index_quotes` | Read index quotes | read-only | no | allowed | `instrument_ids` | `updated_at`, `venue_timestamp` |
| `get_indexes` | Read index instruments | read-only | no | allowed | none | `updated_at` |
| `get_option_chains` | Read option-chain metadata | read-only | no | denied | none (an ID or underlying symbol is semantically needed) | `sellout_time_to_expiration` |
| `get_option_historicals` | Read option historical bars | read-only | no | denied | `instrument_ids`, `start_time` | `begins_at` |
| `get_option_instruments` | Read option instruments | read-only | no | denied | none (a chain input is semantically needed) | `sellout_datetime` |
| `get_option_level_upgrade_info` | Read option-level upgrade eligibility/information | read-only | no | denied | `account_number` | none documented |
| `get_option_orders` | Read option orders | read-only | no | denied | `account_number` | `created_at`, `last_transaction_at`, `updated_at` |
| `get_option_positions` | Read option positions | read-only | no | denied | `account_number` | `opened_at` |
| `get_option_quotes` | Read option quotes | read-only | no | denied | `instrument_ids` | `updated_at` |
| `get_option_watchlist` | Read option watchlist | read-only | no | denied | none | none documented |
| `get_pnl_trade_history` | Read trade-level P&L history | read-only | yes | allowed | `account_number` | trade `timestamp` |
| `get_popular_watchlists` | Read Robinhood popular watchlists | read-only | no | denied | none | none documented |
| `get_portfolio` | Read portfolio/buying-power summary | read-only | yes | allowed | `account_number` | none documented |
| `get_realized_pnl` | Read realized P&L | read-only | yes | allowed | `account_number` | `start_time`, `end_time` |
| `get_scanner_filter_specs` | Read scanner filter specifications | read-only | yes | allowed | none | none documented |
| `get_scans` | Read saved scans | read-only | yes | allowed | none | none documented |
| `get_watchlist_items` | Read items in a watchlist | read-only | no | denied | `list_id` | none documented |
| `get_watchlists` | Read watchlists | read-only | no | denied | none | none documented |
| `place_equity_order` | Place an equity order | mutation | no | denied | `account_number`, `side`, `symbol`, `open_lot_id`, `type` | `created_at`, `last_transaction_at` |
| `place_option_order` | Place an option order | mutation | no | denied | `account_number`, `legs`, `option_id`, `position_effect`, `side`, `quantity` | `created_at`, `last_transaction_at`, `updated_at` |
| `remove_from_watchlist` | Remove symbols/assets from a watchlist | mutation | no | denied | `list_id` | none documented |
| `remove_option_from_watchlist` | Remove option contracts from a watchlist | mutation | no | denied | `option_ids` | none documented |
| `review_equity_order` | Preview/review an equity order | mutation/order workflow | no | denied | `account_number`, `side`, `symbol`, `open_lot_id`, `type` | quote venue timestamps |
| `review_option_order` | Preview/review an option order | mutation/order workflow | no | denied | `account_number`, `legs`, `option_id`, `position_effect`, `side`, `quantity` | `updated_at` |
| `run_scan` | Execute a saved market scan | read-only | yes | allowed | `scan_id` | none documented |
| `search` | Search instruments by query | read-only | yes | allowed | `query` | none documented |
| `unfollow_watchlist` | Unfollow a watchlist | mutation | no | denied | `list_id` | `created_at` |
| `update_scan_config` | Change scan sorting/configuration | mutation | no | denied | `scan_id`, `sorting_column`, `sorting_direction` | none documented |
| `update_scan_filters` | Change saved scan filters | mutation | no | denied | `filters`, `predicate`, `values`, `scan_id` | none documented |
| `update_watchlist` | Change watchlist metadata | mutation | no | denied | `list_id` | none documented |

## Optional arguments observed

The inventory’s required-argument column is intentionally narrow. Relevant
optional live schema fields were:

- `get_equity_historicals`: `adjustment_type`, `bounds`, `end_time`, `interval`;
  supported minute forms include `minute` and `5minute`.
- `get_equity_fundamentals`: `bounds`.
- `get_equity_orders`: `created_at_gte`, `cursor`, `order_id`, `placed_agent`,
  `state`, `symbol`.
- `get_equity_positions`: `cursor`.
- `get_realized_pnl`: `asset_classes`, `display_currency`, `end_date`, `span`,
  `start_date`, `timezone`.
- `get_pnl_trade_history`: `cursor`, `span`, `symbol`.
- `search`: `asset_type`, `limit`.

Mutation schemas were inspected only as metadata and never invoked. Their
optional fields include prices, quantities, time-in-force, market-hours,
watchlist metadata, and scan expressions. They are deliberately not reproduced
as application APIs because the app cannot call them.

## Read-only smoke status

Earlier safe development smoke calls succeeded for `search`, `get_equity_quotes`,
`get_equity_fundamentals`, and `get_equity_price_book` using the screenshot
symbol. The returned payload shape included quotes with venue timestamps,
fundamentals with float/volume fields, and book levels with `price`, `quantity`,
and `updated_at`. No response body, secret, or full account identifier was
stored in the repository. The final standalone OAuth callback was not completed,
so account/position live smoke remains pending. No order, order-review, cancellation, account
mutation, watchlist mutation, scan mutation, or transfer tool was called.
