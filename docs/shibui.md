# Shibui Finance read-only integration

Shibui Finance is connected as a supplemental research source, not as a broker
and not as a real-time feed. Its live MCP metadata says the database contains
daily prices, fundamentals, technical indicators, ownership snapshots, and SEC
filing metadata, but **no intraday data**. Tradecopilot uses Shibui for sourced
public float, 50 completed daily-volume inputs, ownership context, and recent
SEC filing metadata. It never substitutes Shibui data for an Alpaca quote,
intraday candle, entry trigger, VWAP, spread, Robinhood Level 2 snapshot, news
article, or time-and-sales print.

## Codex registration

The global Codex registration created during implementation is:

```bash
codex mcp add shibui --url https://mcp.shibui.finance/mcp
```

The standalone Python provider uses the same HTTPS endpoint. Override it only
for a reviewed compatible proxy:

```bash
export TRADECOPILOT_SHIBUI_MCP_URL=https://mcp.shibui.finance/mcp
uv run tradecopilot doctor
```

No API key was needed during the read-only smoke test. If Shibui changes its
authentication requirements, the provider fails closed; do not put credentials
in source, replay fixtures, logs, or prompts.

## Discovered tools and policy

Live discovery on 2026-08-10 returned the following tools:

| Tool | Classification | Application policy | Use |
|---|---|---|---|
| `get_database_schema` | read-only | allowed | Required schema load before SQL |
| `get_query_patterns` | read-only | allowed | Required query-pattern load before SQL |
| `stock_data_query` | read-only | allowed | Fixed symbol-scoped float and daily-context queries |
| `load_fundamental_workflow` | read-only | denied | Not needed |
| `load_technical_workflow` | read-only | denied | Indicators are calculated locally |
| `load_screening_workflow` | read-only | denied | Not needed by the live monitor |
| `load_comparison_workflow` | read-only | denied | Not needed |
| `load_earnings_workflow` | read-only | denied | Not needed |
| `load_backtesting_workflow` | read-only | denied | Local replay evaluator is authoritative |
| `load_filing_workflow` | read-only | denied | Filing metadata is not treated as a catalyst |
| `load_insider_workflow` | read-only | denied | Not needed |
| `export_to_excel` | mutation/file export | denied | Outside analysis runtime scope |

Unknown or newly added tools are denied. The fixed query is:

```sql
SELECT ticker, shares_float
FROM shibui.ownership_stats
WHERE ticker = '<validated symbol>' AND shares_float IS NOT NULL
LIMIT 1
```

Symbols must pass a strict ticker format check before they can enter the SQL.
The query text is deterministic and is never authored by an LLM. Shibui does
not return a per-row float timestamp, so the normalized evidence records the
retrieval timestamp and carries `LIMITED` data quality.

The daily-context query requires exactly 50 available daily rows before it
returns ADV50. Its latest date and filing acceptance timestamp are normalized
separately. No daily close from this query is exposed as a live-price field.

## News and time-and-sales

Alpaca supplies raw live trade prints, but their aggressor side is unclassified
in this integration. Use a separately licensed provider if classified tape is
required. Use a separately licensed or manually managed source for factual news.
The safe file boundary is enabled by setting:

```bash
export TRADECOPILOT_SUPPLEMENTAL_FEED=/absolute/path/to/supplemental.json
```

See `examples/supplemental.example.json` for the schema. Every float, news, and
tape record requires its own source and timezone-aware timestamp. The example
is synthetic and must not be used as live evidence.
