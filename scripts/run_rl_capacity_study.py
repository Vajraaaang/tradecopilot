"""Run the sealed, offline neural capacity study with a registered shared budget."""

import argparse
from pathlib import Path

from tradecopilot.rl.study import run_study


def absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("absolute path required")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "registration", "budget", "output-dir"):
        parser.add_argument(f"--{name}", type=absolute_path, required=True)
    args = parser.parse_args()
    print(run_study(args.prepared, args.registration, args.budget, args.output_dir))


if __name__ == "__main__":
    main()
