"""Offline OHLCV and stronger CPU comparison on the original validation sessions."""

import argparse
from pathlib import Path

from tradecopilot.forecast.ohlcv_study import run_ohlcv_validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(run_ohlcv_validation(args.dataset, args.archives, args.baseline, args.output_dir))


if __name__ == "__main__":
    main()
