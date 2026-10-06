"""Explicit paper-data, local inference and read-only Kronos viewer commands."""

from __future__ import annotations

import argparse
import asyncio
import json
import webbrowser
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

SYMBOLS = ("AAPL", "AMZN", "MSFT", "NVDA", "TSLA")


def add_parser(commands: Any) -> None:
    parent = commands.add_parser("kronos", help="Local Kronos forecasting with Alpaca paper-account data")
    actions = parent.add_subparsers(dest="kronos_command", required=True)
    fetch = actions.add_parser("fetch", help="GET paper account/clock and completed stock candles")
    fetch.add_argument("--start", required=True, type=date.fromisoformat)
    fetch.add_argument("--end", required=True, type=date.fromisoformat, help="exclusive New York date")
    fetch.add_argument("--symbols", nargs="+", default=SYMBOLS)
    fetch.add_argument("--feed", choices=("iex", "sip"), default="iex")
    fetch.add_argument("--output-dir", type=Path, required=True)
    pilot = actions.add_parser("pilot", help="Run the fixed development pilot and pending paper forecast")
    pilot.add_argument("--bars", type=Path, required=True)
    pilot.add_argument("--checkpoints", type=Path, required=True)
    pilot.add_argument("--output-dir", type=Path, required=True)
    pilot.add_argument("--samples", type=int, default=20)
    pilot.add_argument("--no-prospective", action="store_true")
    serve = actions.add_parser("serve", help="Display an existing immutable saved report")
    serve.add_argument("report", type=Path)
    serve.add_argument("--port", type=int, default=8767)
    serve.add_argument("--open", action="store_true", dest="open_browser")
    grade = actions.add_parser("grade", help="Join later exact candles without replacing original forecasts")
    grade.add_argument("report", type=Path)
    grade.add_argument("--bars", type=Path, required=True)
    grade.add_argument("--output-dir", type=Path, required=True)


async def _fetch(args: argparse.Namespace) -> Path:
    from tradecopilot.forecast.bars import write_bar_dataset
    from tradecopilot.forecast.paper import read_paper_snapshot

    if not args.output_dir.is_absolute() or args.output_dir.exists():
        raise ValueError("choose a new absolute source directory")
    zone = ZoneInfo("America/New_York")
    snapshot = await read_paper_snapshot(
        args.symbols,
        datetime.combine(args.start, time.min, tzinfo=zone),
        datetime.combine(args.end, time.min, tzinfo=zone),
        feed=args.feed,
    )
    args.output_dir.mkdir(parents=True, mode=0o700)
    for index, data in enumerate(snapshot.raw_pages):
        p = args.output_dir / f"page-{index:03d}.json"
        p.write_bytes(data)
        p.chmod(0o600)
    metadata = snapshot.metadata | {
        "selected_symbols": snapshot.metadata["requested_symbols"],
        "paper_account": snapshot.account,
        "clock": snapshot.clock,
        "account_connection": "verified_paper_read_only",
    }
    path = write_bar_dataset(args.output_dir / "bars", snapshot.bars, metadata)
    for p in args.output_dir.rglob("*"):
        p.chmod(0o700 if p.is_dir() else 0o600)
    return path


def dispatch(args: argparse.Namespace) -> int:
    if args.kronos_command == "fetch":
        path = asyncio.run(_fetch(args))
        print(json.dumps({"source": str(path), "mode": "paper", "feed": args.feed, "broker_orders": 0}))
    elif args.kronos_command == "pilot":
        from tradecopilot.forecast.kronos_run import run_pilot

        path = run_pilot(
            args.bars, args.checkpoints, args.output_dir, samples=args.samples, prospective=not args.no_prospective
        )
        print(json.dumps({"report": str(path), "evidence": "retrospective_development_pilot", "broker_orders": 0}))
    elif args.kronos_command == "serve":
        from tradecopilot.forecast.kronos_report import load_report
        from tradecopilot.forecast.service import create_server

        if not args.report.is_absolute():
            raise ValueError("report path must be absolute")
        load_report(args.report)
        server = create_server(args.report, port=args.port, loader=load_report, dashboard_name="kronos_dashboard.html")
        try:
            if args.open_browser:
                webbrowser.open(f"http://127.0.0.1:{server.server_port}/")
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    elif args.kronos_command == "grade":
        from tradecopilot.forecast.bars import load_bar_dataset
        from tradecopilot.forecast.kronos_report import attach_outcomes, load_report, write_report

        if not all(p.is_absolute() for p in (args.report, args.bars, args.output_dir)):
            raise ValueError("grade paths must be absolute")
        original = load_report(args.report)
        bars, source = load_bar_dataset(args.bars)
        value = attach_outcomes(original, bars, source["source_metadata"])
        value["parent_report_id"] = original["report_id"]
        value["outcome_source_id"] = source["data_id"]
        artifacts = {name: (args.report.parent / name).read_bytes() for name in original["inventory"]}
        artifacts["outcome-source.json"] = (json.dumps(source, sort_keys=True) + "\n").encode()
        path = write_report(args.output_dir, value, artifacts)
        print(json.dumps({"report": str(path), "original_preserved": True, "broker_orders": 0}))
    else:
        raise ValueError("unknown Kronos command")
    return 0
