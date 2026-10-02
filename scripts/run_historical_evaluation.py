"""Download/import free provider samples, evaluate offline, and optionally make ten retrospective Jev calls."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

from tradecopilot.auth import load_jev_api_key
from tradecopilot.forecast.contracts import ForecastConfig
from tradecopilot.forecast.data import ForecastStore
from tradecopilot.forecast.dataset import build_dataset, write_dataset
from tradecopilot.forecast.experiment import run_experiment
from tradecopilot.forecast.historical import import_frd_samples
from tradecopilot.forecast.jev import JevForecaster
from tradecopilot.forecast.retrospective import run_retrospective

SYMBOLS = ("AAPL", "MSFT", "AMZN", "NFLX", "TSLA")


def download_samples(directory: Path) -> None:
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("download into a new directory to preserve previously retrieved source files")
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for symbol in SYMBOLS:
        url = f"https://frd001.s3.us-east-2.amazonaws.com/frd_sample_stock_{symbol}.zip"
        with urlopen(url, timeout=30) as response:
            raw = response.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("provider sample exceeds the bounded download size")
        (directory / f"{symbol}_sample.zip").write_bytes(raw)
        records.append(
            {
                "symbol": symbol,
                "url": url,
                "retrieved_at": datetime.now(UTC).isoformat(),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
        )
    (directory / "downloads.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--download", action="store_true", help="Download free minute samples into a new archive directory"
    )
    parser.add_argument("--jev", action="store_true", help="Authorize up to ten paid RETROSPECTIVE Jev calls")
    parser.add_argument("--run-id", default="historical-frd-v1")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("choose a new output directory; historical runs and selections are immutable")
    if args.download:
        download_samples(args.archives)
    config = ForecastConfig(symbols=SYMBOLS)
    observations, source = import_frd_samples(args.archives, config.symbols)
    manifest, examples = build_dataset(observations, config)
    source |= {"dataset_id": manifest.dataset_id, "observations_hash": manifest.observations_hash}
    args.output_dir.mkdir(parents=True)
    with ForecastStore(args.output_dir / "observations.sqlite3") as store:
        store.ingest(observations)
    write_dataset(args.output_dir / "dataset", manifest, examples)
    (args.output_dir / "historical-source.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    baseline = run_experiment(
        manifest, examples, args.output_dir / "baseline", include_jev_fixture=False, historical_source=source
    )
    print(
        json.dumps(
            {
                "baseline_report": str(baseline.resolve()),
                "observations": len(observations),
                "examples": len(examples),
                "labeled": manifest.labeled_count,
                "sessions": len(manifest.sessions),
            },
            indent=2,
        ),
        flush=True,
    )
    if args.jev:
        key = asyncio.run(load_jev_api_key())
        if not key:
            raise ValueError("Configure Jev in the keychain or TYPESAFE_API_KEY before opting into paid inference")
        forecaster = JevForecaster(key, run_id=args.run_id, max_requests=10, max_cost_usd=0.05)
        path = run_retrospective(manifest, examples, baseline, args.output_dir / "retrospective", forecaster)
        print(json.dumps({"retrospective_report": str(path.resolve())}), flush=True)


if __name__ == "__main__":
    main()
