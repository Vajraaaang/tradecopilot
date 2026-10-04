"""Prepare or collect a frozen, explicitly retrospective Jev RL forecast cache."""
import argparse
import asyncio
from pathlib import Path

from tradecopilot.auth import JevKeychainStorage
from tradecopilot.forecast.jev_rl import _load_plan, collect_cache, prepare_requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="freeze masked requests without API calls")
    for name in ("bars", "registration", "prepared", "output-dir"):
        prepare.add_argument(f"--{name}", type=Path, required=True)
    collect = commands.add_parser("collect", help="execute the sealed paid plan once")
    collect.add_argument("--plan", type=Path, required=True)
    collect.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = [value for value in vars(args).values() if isinstance(value, Path)]
    if not all(path.is_absolute() for path in paths):
        parser.error("all paths must be absolute")
    if args.command == "prepare":
        result = prepare_requests(args.bars, args.registration, args.prepared, args.output_dir)
    else:
        _load_plan(args.plan)
        # Existing Keychain entry only; no environment fallback, diagnostics, or credential logging.
        key = asyncio.run(JevKeychainStorage().get_api_key())
        if not key:
            parser.error("Jev Keychain credential is missing")
        result = collect_cache(args.plan, args.output_dir, api_key=key)
    print(result)


if __name__ == "__main__":
    main()
