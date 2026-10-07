"""Run the sealed Jev policy ablation or one isolated bounded training worker."""

import argparse
from pathlib import Path

from tradecopilot.rl.jev_ablation import PROFILES, run_ablation, train_profile


def absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("absolute path required")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "cache", "registration", "budget", "output-dir"):
        parser.add_argument(f"--{name}", type=absolute_path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44))
    args = parser.parse_args()
    if args.worker:
        if args.profile is None or args.seed is None:
            parser.error("worker requires --profile and --seed")
        result = train_profile(
            args.prepared, args.cache, args.registration, args.budget, args.profile, args.seed, args.output_dir
        )
        print(result["training_id"])
    else:
        if args.profile is not None or args.seed is not None:
            parser.error("profile/seed apply only to isolated worker mode")
        print(run_ablation(args.prepared, args.cache, args.registration, args.budget, args.output_dir))


if __name__ == "__main__":
    main()
