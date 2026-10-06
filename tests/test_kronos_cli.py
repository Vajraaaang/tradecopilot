import argparse
from pathlib import Path


def test_kronos_commands_are_available_without_loading_models_or_credentials():
    from tradecopilot.forecast.cli import add_forecast_parser

    parser = argparse.ArgumentParser()
    add_forecast_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(
        [
            "forecast",
            "kronos",
            "pilot",
            "--bars",
            "/tmp/bars",
            "--checkpoints",
            "/tmp/models",
            "--output-dir",
            "/tmp/report",
            "--samples",
            "20",
        ]
    )
    assert args.kronos_command == "pilot"
    assert args.bars == Path("/tmp/bars") and args.samples == 20
    args = parser.parse_args(
        [
            "forecast",
            "kronos",
            "fetch",
            "--start",
            "2026-10-01",
            "--end",
            "2026-10-07",
            "--output-dir",
            "/tmp/source",
            "--feed",
            "sip",
        ]
    )
    assert args.kronos_command == "fetch" and args.feed == "sip"
