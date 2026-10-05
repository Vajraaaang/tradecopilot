"""Prepare frozen supervised forecast cases without fitting or scoring a model."""

from __future__ import annotations

import argparse
from pathlib import Path

from tradecopilot.forecast.supervised_data import merge_source_parts, prepare_supervised


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bars", type=Path)
    source.add_argument("--source-parts", type=Path, nargs=2)
    parser.add_argument("--merged-bars", type=Path)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    paths = [
        args.registration,
        args.output_dir,
        *(args.source_parts or []),
        *([args.bars] if args.bars else []),
        *([args.merged_bars] if args.merged_bars else []),
    ]
    if not all(path.is_absolute() for path in paths):
        parser.error("all paths must be absolute")
    if args.source_parts:
        if args.merged_bars is None:
            parser.error("--source-parts requires --merged-bars")
        merge_source_parts(args.source_parts, args.registration, args.merged_bars)
        args.bars = args.merged_bars
    elif args.merged_bars is not None:
        parser.error("--merged-bars requires --source-parts")
    print(prepare_supervised(args.bars, args.registration, args.output_dir))


if __name__ == "__main__":
    main()
