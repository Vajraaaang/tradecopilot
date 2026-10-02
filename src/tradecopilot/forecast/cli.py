"""Explicit forecast workflows; offline commands never load provider credentials."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tradecopilot.forecast.contracts import ForecastConfig, ForecastExample, ForecastPrediction, content_hash

DEFAULT_STORE = Path(".tradecopilot/forecast/observations.sqlite3")


def add_forecast_parser(subparsers: Any) -> None:
    forecast = subparsers.add_parser("forecast", help="Versioned price forecasting, offline evaluation, and Jev pilots")
    commands = forecast.add_subparsers(dest="forecast_command", required=True)
    demo = commands.add_parser("demo", help="Build and evaluate a deterministic synthetic dataset without API calls")
    demo.add_argument("--output-dir", type=Path, default=Path(".tradecopilot/forecast/demo"))
    demo.add_argument("--config", type=Path)
    collect = commands.add_parser("collect", help="Collect bounded Finnhub quotes into a resumable local store")
    collect.add_argument("--cycles", type=int, required=True)
    collect.add_argument("--interval", type=float, default=15)
    collect.add_argument("--store", type=Path, default=DEFAULT_STORE)
    collect.add_argument("--config", type=Path)
    build = commands.add_parser("build", help="Create an immutable causal dataset from recorded quotes")
    build.add_argument("--store", type=Path, default=DEFAULT_STORE)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--config", type=Path)
    experiment = commands.add_parser("experiment", help="Evaluate CPU baselines with chronological session splits")
    experiment.add_argument("dataset", type=Path)
    experiment.add_argument("--output-dir", type=Path, required=True)
    experiment.add_argument("--include-jev-fixture", action="store_true", help="Add an offline stub, not Jev inference")
    experiment.add_argument(
        "--historical-source", type=Path, help="Verified source/replay metadata for a historical dataset"
    )
    predict = commands.add_parser("predict", help="Make one opt-in paid prospective Jev forecast")
    predict.add_argument("symbol")
    predict.add_argument("--store", type=Path, default=DEFAULT_STORE)
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.add_argument("--config", type=Path)
    predict.add_argument("--allow-live", action="store_true", help="Authorize one paid Jev call subject to ledger caps")
    predict.add_argument("--run-id", default="forecast-pilot-v1")
    predict.add_argument("--max-requests", type=int, default=10)
    predict.add_argument("--max-cost-usd", type=float, default=0.05)
    grade = commands.add_parser("grade", help="Join persisted prospective predictions to later observed outcomes")
    grade.add_argument("pilot", type=Path)
    grade.add_argument("--store", type=Path, default=DEFAULT_STORE)
    grade.add_argument("--output-dir", type=Path, required=True)
    status = commands.add_parser("status", help="Show collector freshness, failures, and persisted inference counts")
    status.add_argument("--store", type=Path, default=DEFAULT_STORE)
    serve = commands.add_parser("serve", help="Serve a verified report without model or provider calls")
    serve.add_argument("report", type=Path)
    serve.add_argument("--host", choices=("127.0.0.1", "0.0.0.0"), default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8766)
    serve.add_argument("--open", action="store_true", dest="open_browser")


def _config(path: Path | None) -> ForecastConfig:
    return ForecastConfig.model_validate_json(path.read_text()) if path else ForecastConfig()


def _output(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def dispatch(args: argparse.Namespace) -> int:
    try:
        return _dispatch(args)
    except OSError:
        # Paths and provider response text are not useful diagnostic payloads for these commands.
        raise ValueError(
            "Forecast file operation failed; check input paths and use a new writable output directory."
        ) from None


def _dispatch(args: argparse.Namespace) -> int:
    from tradecopilot.forecast.data import ForecastStore
    from tradecopilot.forecast.dataset import build_dataset, load_dataset, write_dataset
    from tradecopilot.forecast.experiment import load_report, run_experiment, write_pilot_report

    command = args.forecast_command
    if command == "serve":
        from tradecopilot.forecast.service import serve_report

        serve_report(args.report, host=args.host, port=args.port, open_browser=args.open_browser)
        return 0
    if command == "demo":
        from tradecopilot.forecast.demo import synthetic_observations

        config = _config(args.config)
        report_path = args.output_dir / "run" / "report.json"
        if report_path.exists():
            report = load_report(report_path)
            if report["evaluation_kind"] != "synthetic_demo" or report["dataset"]["config"] != config.model_dump(
                mode="json"
            ):
                raise ValueError("Existing demo uses a different config; choose a new output directory.")
        else:
            rows = synthetic_observations(config)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            with ForecastStore(args.output_dir / "observations.sqlite3") as store:
                store.ingest(rows)
            manifest, examples = build_dataset(rows, config)
            write_dataset(args.output_dir / "dataset", manifest, examples)
            report_path = run_experiment(manifest, examples, args.output_dir / "run")
        _output({"report": str(report_path.resolve()), "evidence": "synthetic_demo", "paid_api_calls": 0})
        return 0
    if command == "experiment":
        manifest, examples = load_dataset(args.dataset)
        historical_source = json.loads(args.historical_source.read_text()) if args.historical_source else None
        report_path = run_experiment(
            manifest,
            examples,
            args.output_dir,
            include_jev_fixture=args.include_jev_fixture,
            historical_source=historical_source,
        )
        _output({"report": str(report_path), "evidence": load_report(report_path)["evaluation_kind"]})
        return 0
    if command == "collect":
        from tradecopilot.auth import load_finnhub_api_key
        from tradecopilot.forecast.collector import collect_quotes
        from tradecopilot.providers.finnhub import FinnhubClient

        config = _config(args.config)
        if args.cycles < 1 or args.cycles > 1600:
            raise ValueError("Choose 1-1600 collection cycles per invocation.")
        key = asyncio.run(load_finnhub_api_key())
        if not key:
            raise ValueError("Configure Finnhub with `tradecopilot auth finnhub` or FINNHUB_API_KEY.")
        with ForecastStore(args.store) as store:
            metrics = asyncio.run(
                collect_quotes(
                    store,
                    config,
                    cycles=args.cycles,
                    interval_seconds=args.interval,
                    client=FinnhubClient(key),
                )
            )
        _output(metrics)
        return 0 if metrics["failures"] == 0 else 1
    if command == "build":
        config = _config(args.config)
        if not args.store.is_file():
            raise ValueError("Observation store does not exist; collect quotes before building a dataset.")
        with ForecastStore(args.store) as store:
            manifest, examples = build_dataset(store.observations(), config)
        path = write_dataset(args.output_dir, manifest, examples)
        _output({"manifest": str(path.resolve()), "examples": len(examples), "labeled": manifest.labeled_count})
        return 0
    if command == "predict":
        return _predict(args)
    if command == "grade":
        config = _config(args.pilot / "config.json")
        inputs = [
            ForecastExample.model_validate_json(line) for line in (args.pilot / "inputs.jsonl").read_text().splitlines()
        ]
        predictions = [
            ForecastPrediction.model_validate_json(line)
            for line in (args.pilot / "predictions.jsonl").read_text().splitlines()
        ]
        if not args.store.is_file():
            raise ValueError("The outcome observation store does not exist.")
        with ForecastStore(args.store) as store:
            path = write_pilot_report(
                config,
                inputs,
                predictions,
                store.observations(),
                args.output_dir,
                recorded_predictions=store.predictions(),
            )
        _output({"report": str(path), "evidence": "prospective_pilot"})
        return 0
    if command == "status":
        if not args.store.is_file():
            raise ValueError("Observation store does not exist.")
        with ForecastStore(args.store) as store:
            rows, events, predictions = store.observations(), store.events(), store.predictions()
        now = datetime.now(UTC)
        latest = {row.symbol: row for row in rows}
        _output(
            {
                "observations": len(rows),
                "predictions": len(predictions),
                "events": {
                    kind: sum(row["kind"] == kind for row in events) for kind in sorted({e["kind"] for e in events})
                },
                "latest": {
                    symbol: {
                        "receipt_timestamp": row.receipt_timestamp.isoformat(),
                        "provider_timestamp": row.provider_timestamp.isoformat(),
                        "source_age_seconds_now": (now - row.provider_timestamp).total_seconds(),
                    }
                    for symbol, row in latest.items()
                },
            }
        )
        return 0
    raise ValueError("Unknown forecast command")


def _predict(args: argparse.Namespace) -> int:
    from tradecopilot.auth import load_jev_api_key
    from tradecopilot.forecast.data import ForecastStore
    from tradecopilot.forecast.features import feature_result
    from tradecopilot.forecast.jev import JevForecaster

    if not args.allow_live:
        raise ValueError("Paid Jev inference requires --allow-live. The forecast demo is free and offline.")
    config = _config(args.config)
    if args.output_dir.exists():
        raise ValueError("Pilot output directory already exists; choose a new directory for this forecast.")
    if not args.store.is_file():
        raise ValueError("Collect fresh market observations before requesting a forecast.")
    with ForecastStore(args.store) as store:
        example, reason = feature_result(store.observations(), datetime.now(UTC), args.symbol.upper(), config)
        if example is None or example.provenance != "market":
            raise ValueError(f"Cannot forecast this input: {reason or 'market_observations_required'}.")
        key = asyncio.run(load_jev_api_key())
        if not key:
            raise ValueError("Configure Jev with `tradecopilot auth jev` or TYPESAFE_API_KEY.")
        forecaster = JevForecaster(
            key,
            run_id=args.run_id,
            max_requests=args.max_requests,
            max_cost_usd=args.max_cost_usd,
        )
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "config.json").write_text(config.model_dump_json(indent=2) + "\n")
        # Persist and flush the original input before sending the paid request.
        with (args.output_dir / "inputs.jsonl").open("x") as handle:
            handle.write(example.model_dump_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        dataset_id = content_hash({"config_id": config.config_id, "example_id": example.example_id})
        prediction = forecaster.predict(example, config, dataset_id)
        store.save_prediction(prediction)
        (args.output_dir / "predictions.jsonl").write_text(prediction.model_dump_json() + "\n")
    _output(prediction.model_dump(mode="json"))
    return 1 if prediction.status == "error" else 0
