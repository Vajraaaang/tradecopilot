import argparse
from pathlib import Path


def test_grade_passes_the_verified_original_path_and_preserves_it(tmp_path, capsys):
    import json
    from datetime import UTC, datetime, timedelta

    from tradecopilot.forecast.bars import HistoricalBar, write_bar_dataset
    from tradecopilot.forecast.kronos_cli import dispatch
    from tradecopilot.forecast.kronos_report import load_report, write_report

    opening = datetime(2026, 10, 7, 13, 30, tzinfo=UTC)
    future = [opening + timedelta(minutes=i) for i in range(1, 16)]
    original_path = write_report(tmp_path / "original", {
        "connection": {"feed": "sip"},
        "cases": [{"group": "prospective", "symbol": "AAPL", "future_times": [t.isoformat() for t in future],
                   "forecasts": {}, "actual_close": None}],
    }, {"paths.npz": b"preserved paths"})
    original_bytes = original_path.read_bytes()
    bars = [HistoricalBar(symbol="AAPL", start_time=t - timedelta(minutes=1), end_time=t, available_at=t,
                          opening=101, high=101, low=101, close=101, volume=1, source="alpaca_sip_1min_bar")
            for t in future]
    write_bar_dataset(tmp_path / "bars", bars, {
        "feed": "sip", "receipt_at": (future[-1] + timedelta(minutes=1)).isoformat(),
    })
    args = argparse.Namespace(kronos_command="grade", report=original_path, bars=tmp_path / "bars",
                              output_dir=tmp_path / "graded")
    assert dispatch(args) == 0
    result = load_report(tmp_path / "graded" / "report.json")
    assert result["parent_report_id"] == load_report(original_path)["report_id"]
    assert result["published_at"] == load_report(original_path)["published_at"]
    assert result["cases"][0]["actual_close"] == [101.0] * 15
    assert original_path.read_bytes() == original_bytes
    assert json.loads(capsys.readouterr().out)["original_preserved"] is True


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
