"""Train one registered supervised seed on frozen TRAIN and TUNE inputs only."""

import argparse
import json
from pathlib import Path

from tradecopilot.forecast.supervised_models import train_neural_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--candidate", choices=("tcn-32", "lstm-64"), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = train_neural_seed(args.prepared, args.registration, args.candidate, args.seed, args.output_dir)
    print(json.dumps({key: result[key] for key in ("status", "eligible", "model_path", "checkpoint_hash")},
                     sort_keys=True))
    if not result["eligible"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
