"""Publish derived-only validation development figures and metadata, never source bars or raw cases."""

import argparse
import json
from pathlib import Path

from render_historical_results import comparison, configure

from tradecopilot.forecast.experiment import load_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = load_report(args.report)
    if report["evaluation_kind"] != "ohlcv_validation_development":
        raise ValueError("only verified OHLCV validation development results may be rendered here")
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    configure()
    validation = report["split"]["validation"]
    comparison(
        report,
        destination / "validation-comparison.png",
        "OHLCV CPU comparison · validation development",
        f"Same {validation['examples']:,} cases / five stocks · September 25 and 28 · "
        "fixed settings; previously inspected test excluded",
    )
    source = report["data_source"]
    summary = {
        "schema_version": "derived-ohlcv-validation-v2",
        "report_id": report["report_id"],
        "evaluation_kind": report["evaluation_kind"],
        "base_dataset_id": report["dataset"]["dataset_id"],
        "bar_dataset_id": report["bar_dataset_id"],
        "bar_count": source["bar_count"],
        "feature_version": report["feature_version"],
        "split": report["split"],
        "source": report["source"],
        "data_source": source,
        "feature_audit": report["feature_audit"],
        "settings": {
            k: report["protocol"][k]
            for k in (
                "horizon_minutes",
                "flat_band_bps",
                "seed",
                "abstention_threshold",
                "hgb_settings",
                "lr_settings",
                "test_policy",
                "paid_api_calls",
            )
        },
        "models": [
            {k: model[k] for k in ("key", "label", "model_id", "execution", "metrics", "by_symbol", "by_session")}
            for model in report["models"]
        ],
        "timings": report["timings"],
        "limitations": report["limitations"],
    }
    (destination / "derived-validation.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(destination)


if __name__ == "__main__":
    main()
