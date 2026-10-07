"""One bounded offline training process. No credential or network access."""

import argparse
import json
from pathlib import Path

from tradecopilot.rl.training import train_seed


def absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("absolute path required")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "registration", "budget", "output-dir"):
        parser.add_argument(f"--{name}", type=absolute_path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    result = train_seed(args.prepared, args.registration, args.budget, args.candidate, args.seed, args.output_dir)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
