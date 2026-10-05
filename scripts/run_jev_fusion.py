"""Run fixed Jev forecast correction on already consumed dates using the existing cache."""

import argparse
from pathlib import Path

from tradecopilot.forecast.jev_fusion import run_jev_fusion


def absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("absolute path required")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-bars", "old-prepared", "old-registration", "cache", "output-dir"):
        parser.add_argument(f"--{name}", type=absolute_path, required=True)
    args = parser.parse_args()
    print(run_jev_fusion(args.source_bars, args.old_prepared, args.old_registration, args.cache, args.output_dir))


if __name__ == "__main__":
    main()
