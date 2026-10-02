"""Deterministic minute anchors and immutable, self-verifying dataset exports."""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from tradecopilot.forecast.contracts import (
    FEATURE_VERSION,
    LABEL_VERSION,
    DatasetManifest,
    ForecastConfig,
    ForecastExample,
    Observation,
    content_hash,
)
from tradecopilot.forecast.features import feature_result, label_example, normalize_observations
from tradecopilot.forecast.sessions import CALENDAR_VERSION, session_bounds, session_for


def build_dataset(
    observations: Sequence[Observation], config: ForecastConfig,
) -> tuple[DatasetManifest, list[ForecastExample]]:
    normalized = normalize_observations(observations)
    if not normalized:
        raise ValueError("a dataset requires at least one observation")
    if len({row.provenance for row in normalized}) != 1:
        raise ValueError("dataset provenance cannot mix synthetic and market observations")
    grouped: dict[tuple[date, str], list[Observation]] = defaultdict(list)
    exclusions: Counter[str] = Counter()
    for row in normalized:
        if row.symbol not in config.symbols:
            exclusions["observations_outside_universe"] += 1
            continue
        day = session_for(row.provider_timestamp)
        if day is None:
            exclusions["observations_outside_session"] += 1
            continue
        grouped[(day, row.symbol)].append(row)
    examples = []
    for (day, symbol), rows in sorted(grouped.items()):
        bounds = session_bounds(day)
        assert bounds is not None
        first = min(row.receipt_timestamp for row in rows)
        last = min(bounds[1], max(row.receipt_timestamp for row in rows))
        offset = max(0, math.ceil((first - bounds[0]).total_seconds() / config.anchor_seconds))
        anchor = bounds[0] + timedelta(seconds=offset * config.anchor_seconds)
        while anchor <= last:
            example, reason = feature_result(rows, anchor, symbol, config)
            if example is None:
                assert reason is not None
                exclusions[reason] += 1
            else:
                labeled = label_example(rows, example, config)
                if labeled.exclusion_reason:
                    exclusions[labeled.exclusion_reason] += 1
                examples.append(labeled)
            anchor += timedelta(seconds=config.anchor_seconds)
    examples.sort(key=lambda example: (example.as_of, example.symbol, example.example_id))
    manifest = DatasetManifest(
        dataset_id="pending",
        observations_hash=content_hash([row.model_dump(mode="json") for row in normalized]),
        examples_hash=content_hash([row.model_dump(mode="json") for row in examples]),
        config=config, calendar_version=CALENDAR_VERSION, provenance=normalized[0].provenance,
        observation_count=len(normalized), example_count=len(examples),
        labeled_count=sum(example.label is not None for example in examples),
        sessions=tuple(sorted({day for day, _ in grouped})), exclusions=dict(sorted(exclusions.items())),
    )
    return manifest.model_copy(update={"dataset_id": _manifest_hash(manifest)}), examples


def _manifest_hash(manifest: DatasetManifest) -> str:
    return content_hash(manifest.model_dump(mode="json", exclude={"dataset_id"}))


def _verify(manifest: DatasetManifest, examples: list[ForecastExample]) -> None:
    valid = (
        manifest.feature_version == FEATURE_VERSION and manifest.label_version == LABEL_VERSION
        and manifest.calendar == "XNYS" and manifest.calendar_version == CALENDAR_VERSION
        and manifest.dataset_id == _manifest_hash(manifest)
        and manifest.examples_hash == content_hash([row.model_dump(mode="json") for row in examples])
        and manifest.example_count == len(examples)
        and manifest.labeled_count == sum(row.label is not None for row in examples)
        and len({row.example_id for row in examples}) == len(examples)
        and len(manifest.observations_hash) == 64
        and all(number >= 0 for number in manifest.exclusions.values())
    )
    for row in examples:
        bounds = session_bounds(row.session_date)
        valid = valid and (
            row.config_id == manifest.config.config_id and row.provenance == manifest.provenance
            and row.symbol in manifest.config.symbols and row.session_date in manifest.sessions
            and session_for(row.as_of) == row.session_date and bounds is not None
            and row.target_time == row.as_of + timedelta(minutes=manifest.config.horizon_minutes)
            and row.target_time <= bounds[1]
            and row.features["source_age_seconds"] <= manifest.config.max_source_age_seconds
            and (row.label_observed_at is None or row.label_observed_at <= (
                row.target_time + timedelta(seconds=manifest.config.max_outcome_delay_seconds)
            ))
        )
    if not valid:
        raise ValueError("dataset integrity verification failed")


def write_dataset(out_dir: Path, manifest: DatasetManifest, examples: Sequence[ForecastExample]) -> Path:
    examples = [ForecastExample.model_validate(row.model_dump(mode="json")) for row in examples]
    manifest = DatasetManifest.model_validate(manifest.model_dump(mode="json"))
    _verify(manifest, examples)
    destination = out_dir / "manifest.json"
    if out_dir.exists() and any(out_dir.iterdir()):
        existing, stored = load_dataset(out_dir)
        if existing != manifest or stored != examples:
            raise FileExistsError("dataset directory already contains a different immutable dataset")
        return destination
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{out_dir.name}.", dir=out_dir.parent) as temporary:
        staging = Path(temporary)
        (staging / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
        (staging / "examples.jsonl").write_text(
            "".join(row.model_dump_json() + "\n" for row in examples), encoding="utf-8",
        )
        try:
            os.rename(staging, out_dir)
        except OSError:
            if not out_dir.is_dir():
                raise
            existing, stored = load_dataset(out_dir)
            if existing != manifest or stored != examples:
                raise FileExistsError("dataset directory already contains a different immutable dataset") from None
    return destination


def load_dataset(path: Path) -> tuple[DatasetManifest, list[ForecastExample]]:
    manifest_path = path / "manifest.json" if path.is_dir() else path
    try:
        manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        lines = (manifest_path.parent / "examples.jsonl").read_text(encoding="utf-8").splitlines()
        examples = [ForecastExample.model_validate(json.loads(line)) for line in lines if line.strip()]
        _verify(manifest, examples)
    except (ValueError, OSError):
        raise ValueError("dataset integrity verification failed") from None
    return manifest, examples
