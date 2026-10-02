"""Import bounded historical Alpaca SIP minute bars into a new private local directory."""

from __future__ import annotations

import argparse
import asyncio
from datetime import date
from pathlib import Path

from tradecopilot.auth import AlpacaKeychainStorage
from tradecopilot.forecast.alpaca_history import download_history
from tradecopilot.forecast.contracts import ForecastConfig

SYMBOLS = ("AAPL", "MSFT", "AMZN", "NFLX", "TSLA")


async def _download(args: argparse.Namespace) -> Path:
    try:
        credentials = await AlpacaKeychainStorage().get_credentials()
    except Exception:
        raise ValueError("Unable to read Alpaca credentials; run tradecopilot auth alpaca") from None
    if credentials is None:
        raise ValueError("Missing Alpaca credentials; run tradecopilot auth alpaca")
    config = ForecastConfig(symbols=tuple(args.symbols))
    return await download_history(config, args.start, args.end, args.output_dir, *credentials)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2026, 4, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 9, 16), help="Exclusive New York date")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    args = parser.parse_args()
    try:
        path = asyncio.run(_download(args))
    except ValueError as error:
        raise SystemExit(str(error)) from None
    print(path.resolve())


if __name__ == "__main__":
    main()
