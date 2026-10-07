"""Run the offline selective accuracy experiment on a separately acquired bar store."""

import argparse
from pathlib import Path

from tradecopilot.forecast.selective_study import run_selective_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(run_selective_study(args.bars, args.output_dir))


if __name__ == "__main__":
    main()
