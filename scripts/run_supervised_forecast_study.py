"""Run the registered direct supervised comparison on a private prepared corpus."""

import argparse
from pathlib import Path

from tradecopilot.forecast.supervised_study import run_supervised_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(run_supervised_study(args.prepared, args.registration, args.output_dir))


if __name__ == "__main__":
    main()
