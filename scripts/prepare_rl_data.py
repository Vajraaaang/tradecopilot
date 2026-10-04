"""Prepare registered causal observations without scoring or training any policy."""
import argparse
from pathlib import Path

from tradecopilot.rl.data import prepare_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", required=True, type=Path)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if not all(path.is_absolute() for path in (args.bars, args.registration, args.output_dir)):
        parser.error("all paths must be absolute")
    print(prepare_data(args.bars, args.registration, args.output_dir))


if __name__ == "__main__":
    main()
