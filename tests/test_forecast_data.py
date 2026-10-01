from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from tradecopilot.forecast.contracts import ForecastConfig, ForecastPrediction, Observation
from tradecopilot.models import DataQuality


def observation(at, *, symbol="AAPL", price="100", receipt=None, provenance="synthetic"):
    receipt = receipt or at
    return Observation(
        symbol=symbol,
        last=Decimal(price),
        previous_close=Decimal("100"),
        provider_timestamp=at,
        receipt_timestamp=receipt,
        age_seconds=(receipt - at).total_seconds(),
        source="synthetic_fixture" if provenance == "synthetic" else "finnhub_quote",
        quality=DataQuality.LIMITED,
        provenance=provenance,
    )


def test_exchange_sessions_cover_holidays_dst_and_early_close():
    from tradecopilot.forecast.sessions import session_bounds, session_for

    assert session_bounds(date(2026, 9, 7)) is None
    assert session_bounds(date(2026, 9, 5)) is None
    assert session_bounds(date(2026, 3, 6))[0].hour == 14
    assert session_bounds(date(2026, 3, 9))[0].hour == 13
    opened, closed = session_bounds(date(2026, 11, 27))
    assert opened == datetime(2026, 11, 27, 14, 30, tzinfo=UTC)
    assert closed == datetime(2026, 11, 27, 18, tzinfo=UTC)
    assert session_for(opened) == opened.date()
    assert session_for(closed) == opened.date()
    assert session_for(closed + timedelta(microseconds=1)) is None
    with pytest.raises(ValueError, match="timezone"):
        session_for(datetime(2026, 11, 27, 15))


def test_store_deduplicates_exact_rows_survives_restart_and_preserves_symbols(tmp_path):
    from tradecopilot.forecast.data import ForecastStore

    now = datetime(2026, 9, 1, 14, tzinfo=UTC)
    aapl = observation(now)
    msft = observation(now, symbol="MSFT")
    later_receipt = observation(now, receipt=now + timedelta(seconds=1))
    database = tmp_path / "observations.sqlite3"
    with ForecastStore(database) as one, ForecastStore(database) as two:
        assert one.ingest([aapl, aapl]) == 1
        assert two.ingest([aapl, msft, later_receipt]) == 2
        one.record_event("quote_failure", {"symbol": "AAPL", "reason": "quote_unavailable", "attempt": 1})
    with ForecastStore(database) as resumed:
        assert resumed.ingest([aapl]) == 0
        assert [row.symbol for row in resumed.observations()] == ["AAPL", "MSFT", "AAPL"]
        assert resumed.events()[0]["payload"]["reason"] == "quote_unavailable"
        assert len(resumed.observations()) == 3


def test_store_persists_prediction_ids_idempotently(tmp_path):
    from tradecopilot.forecast.data import ForecastStore

    prediction = ForecastPrediction(
        example_id="input", dataset_id="dataset", model_id="baseline",
        generated_at=datetime(2026, 9, 1, 14, tzinfo=UTC), status="ok", execution="local",
        probabilities={"DOWN": 0.2, "FLAT": 0.3, "UP": 0.5},
    )
    with ForecastStore(tmp_path / "observations.sqlite3") as store:
        store.save_prediction(prediction)
        store.save_prediction(prediction)
        assert store.predictions() == [prediction]


def test_store_rejects_invalid_rows_atomically(tmp_path):
    from tradecopilot.forecast.data import ForecastStore

    at = datetime(2026, 9, 1, 14, tzinfo=UTC)
    valid = observation(at)
    invalid = valid.model_copy(update={"last": Decimal("0")})
    with ForecastStore(tmp_path / "observations.sqlite3") as store:
        with pytest.raises(ValueError):
            store.ingest([valid, invalid])
        assert store.observations() == []


def test_store_does_not_persist_secret_fields_or_exception_text(tmp_path):
    from tradecopilot.forecast.data import ForecastStore

    with ForecastStore(tmp_path / "observations.sqlite3") as store:
        with pytest.raises(ValueError, match="operational"):
            store.record_event("quote_failure", {"api_key": "TEST_SECRET"})
        with pytest.raises(ValueError, match="operational"):
            store.record_event("quote_failure", {"exception": "TEST_SECRET raw body"})
        assert store.events() == []


def history(as_of, *, symbol="AAPL", price="100"):
    return [observation(as_of - timedelta(seconds=seconds), symbol=symbol, price=price)
            for seconds in range(300, -1, -15)]


def test_features_use_only_received_history_of_the_requested_symbol():
    from tradecopilot.forecast.contracts import FEATURE_NAMES
    from tradecopilot.forecast.features import feature_example

    as_of = datetime(2026, 9, 1, 14, tzinfo=UTC)
    rows = history(as_of)
    delayed = observation(as_of - timedelta(seconds=1), price="500", receipt=as_of + timedelta(seconds=1))
    future = observation(as_of + timedelta(seconds=1), price="500")
    other = observation(as_of, symbol="MSFT", price="500")
    result = feature_example([*rows, delayed, future, other, rows[0]], as_of, "AAPL", ForecastConfig())
    assert result is not None
    assert result.anchor_price == Decimal("100") and result.label is None
    assert result.config_id == ForecastConfig().config_id
    assert set(result.features) == set(FEATURE_NAMES)
    assert result.features["return_5m_bps"] == 0 and result.features["history_points"] == 21
    assert set(result.observation_ids) == {row.observation_id for row in rows}
    excluded = {delayed.observation_id, future.observation_id, other.observation_id}
    assert not excluded.intersection(result.observation_ids)


def test_features_require_history_coverage_fresh_endpoints_and_maximum_gaps():
    from tradecopilot.forecast.features import feature_example

    as_of = datetime(2026, 9, 1, 14, tzinfo=UTC)
    config = ForecastConfig()
    rows = history(as_of)
    assert feature_example(rows[-6:], as_of, "AAPL", config) is None
    assert feature_example(rows[:-3], as_of, "AAPL", config) is None
    assert feature_example(rows[:5] + rows[10:], as_of, "AAPL", config) is None
    assert feature_example(rows[1:], as_of, "AAPL", config) is None  # no as-of 5-minute price
    assert feature_example(rows, as_of, "MSFT", config) is None
    duplicate_provider = observation(as_of, receipt=as_of, price="101")
    with_duplicate = feature_example([*rows, duplicate_provider], as_of, "AAPL", config)
    assert with_duplicate.features["history_points"] == 21


def test_features_allow_offset_sampling_without_future_endpoint_lookahead():
    from tradecopilot.forecast.features import feature_example

    as_of = datetime(2026, 9, 1, 14, tzinfo=UTC)
    rows = [observation(as_of - timedelta(seconds=seconds), price=str(100 + seconds / 1000))
            for seconds in range(310, 0, -15)]
    example = feature_example(rows, as_of, "AAPL", ForecastConfig())
    assert example is not None
    assert rows[0].observation_id in example.observation_ids  # 5-minute price is 10 seconds before endpoint
    assert example.features["return_1m_bps"] == pytest.approx(float(
        (rows[-1].last / next(row.last for row in rows if (as_of - row.provider_timestamp).seconds == 70) - 1)
        * 10000
    ))


def test_labels_arrive_only_after_target_and_keep_input_identity():
    from tradecopilot.forecast.features import feature_example, label_example

    as_of = datetime(2026, 9, 1, 14, tzinfo=UTC)
    config = ForecastConfig()
    rows = history(as_of)
    example = feature_example(rows, as_of, "AAPL", config)
    assert example is not None
    early = observation(example.target_time - timedelta(seconds=1), price="101")
    other = observation(example.target_time, symbol="MSFT", price="101")
    assert label_example([early, other], example, config).label is None
    outcome = observation(example.target_time, price="100.1", receipt=example.target_time + timedelta(seconds=4))
    labeled = label_example([outcome], example, config)
    assert labeled.label == "FLAT"  # Exactly 10 bps is inside the inclusive flat band.
    assert labeled.target_return_bps == 10
    assert labeled.label_observed_at == outcome.receipt_timestamp
    assert labeled.observation_ids == example.observation_ids
    assert labeled.example_id == example.example_id
    assert label_example([observation(example.target_time, price="100.10001")], example, config).label == "UP"
    assert label_example([observation(example.target_time, price="99.89999")], example, config).label == "DOWN"
    with pytest.raises(ValueError, match="config"):
        label_example([outcome], example, ForecastConfig(flat_threshold_bps=20))


def test_labels_accept_the_close_within_the_receipt_window_and_reject_late_or_next_session_substitutes():
    from tradecopilot.forecast.features import feature_example, label_example

    as_of = datetime(2026, 11, 27, 17, 45, tzinfo=UTC)
    config = ForecastConfig()
    example = feature_example(history(as_of), as_of, "AAPL", config)
    assert example.target_time == datetime(2026, 11, 27, 18, tzinfo=UTC)
    close = observation(example.target_time, price="101", receipt=example.target_time + timedelta(seconds=10))
    assert label_example([close], example, config).label == "UP"
    late = observation(example.target_time, price="101", receipt=example.target_time + timedelta(seconds=61))
    assert label_example([late], example, config).label is None
    last_allowed = observation(example.target_time, price="101", receipt=example.target_time + timedelta(seconds=60))
    assert label_example([last_allowed], example, config).label == "UP"
    monday = observation(datetime(2026, 11, 30, 14, 30, tzinfo=UTC), price="101")
    assert label_example([monday], example, config).label is None
    too_late = as_of + timedelta(seconds=1)
    assert feature_example(history(too_late), too_late, "AAPL", config) is None


def test_demo_dataset_is_deterministic_versioned_and_roundtrips_immutably(tmp_path):
    from tradecopilot.forecast.dataset import build_dataset, load_dataset, write_dataset
    from tradecopilot.forecast.demo import synthetic_observations

    config = ForecastConfig()
    rows = synthetic_observations(config, sessions=3, minutes_per_session=30)
    assert rows == synthetic_observations(config, sessions=3, minutes_per_session=30)
    assert len(rows) == 3 * 5 * 121
    manifest, examples = build_dataset(rows, config)
    reordered, second = build_dataset([*reversed(rows), rows[0]], config)
    assert manifest == reordered and examples == second
    assert manifest.observation_count == len(rows)
    assert manifest.example_count == len(examples)
    assert manifest.labeled_count == sum(example.label is not None for example in examples)
    assert manifest.labeled_count > 0 and manifest.provenance == "synthetic"
    assert len(manifest.sessions) == 3
    assert {example.label for example in examples} >= {"UP", "FLAT", "DOWN"}
    assert manifest.exclusions["insufficient_history"] == 3 * 5 * 5
    assert manifest.exclusions["outcome_unavailable"] == 3 * 5 * 15
    output = write_dataset(tmp_path / "dataset", manifest, examples)
    assert load_dataset(output) == (manifest, examples)
    assert write_dataset(tmp_path / "dataset", manifest, examples) == output
    changed, changed_examples = build_dataset(rows, ForecastConfig(flat_threshold_bps=20))
    assert changed.dataset_id != manifest.dataset_id
    with pytest.raises(FileExistsError):
        write_dataset(tmp_path / "dataset", changed, changed_examples)
    manifest_file = output if output.is_file() else output / "manifest.json"
    payload = json.loads(manifest_file.read_text())
    payload["example_count"] += 1
    manifest_file.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="integrity"):
        load_dataset(output)


def test_dataset_rejects_mixed_provenance_and_invalid_data():
    from tradecopilot.forecast.dataset import build_dataset

    at = datetime(2026, 9, 1, 14, tzinfo=UTC)
    with pytest.raises(ValueError, match="provenance"):
        build_dataset([observation(at), observation(at, symbol="MSFT", provenance="market")], ForecastConfig())
    with pytest.raises(ValueError):
        build_dataset([observation(at).model_copy(update={"last": Decimal("-1")})], ForecastConfig())
    with pytest.raises(ValueError, match="observation"):
        build_dataset([], ForecastConfig())


def test_feature_sources_never_cross_session_and_dataset_hash_verifies_examples(tmp_path):
    from tradecopilot.forecast.dataset import build_dataset, load_dataset, write_dataset
    from tradecopilot.forecast.features import feature_example

    as_of = datetime(2026, 9, 2, 13, 32, tzinfo=UTC)
    previous = history(datetime(2026, 9, 1, 19, 59, tzinfo=UTC))
    today = history(as_of)[-9:]
    assert feature_example([*previous, *today], as_of, "AAPL", ForecastConfig()) is None
    later = datetime(2026, 9, 2, 14, tzinfo=UTC)
    manifest, examples = build_dataset(history(later), ForecastConfig())
    output = write_dataset(tmp_path / "dataset", manifest, examples)
    directory = output.parent if output.is_file() else output
    path = directory / "examples.jsonl"
    content = [json.loads(line) for line in path.read_text().splitlines()]
    content[0]["features"]["return_1m_bps"] = 999
    path.write_text("\n".join(json.dumps(row) for row in content) + "\n")
    with pytest.raises(ValueError, match="integrity"):
        load_dataset(output)


def test_five_minute_statistics_keep_their_meaning_with_a_longer_required_history():
    from tradecopilot.forecast.features import feature_example

    as_of = datetime(2026, 9, 1, 14, tzinfo=UTC)
    old = [observation(as_of - timedelta(seconds=seconds), price="120")
           for seconds in range(600, 300, -15)]
    example = feature_example([*old, *history(as_of)], as_of, "AAPL", ForecastConfig(lookback_minutes=10))
    assert example is not None
    assert example.features["history_points"] == 41
    assert example.features["range_5m_bps"] == 0
    assert example.features["volatility_5m_bps"] == 0


def test_outcome_selection_is_deterministic_when_rows_are_reordered():
    from tradecopilot.forecast.features import feature_example, label_example

    as_of = datetime(2026, 9, 1, 14, tzinfo=UTC)
    config = ForecastConfig()
    example = feature_example(history(as_of), as_of, "AAPL", config)
    allowed = observation(example.target_time, receipt=example.target_time + timedelta(seconds=31))
    late = observation(example.target_time + timedelta(minutes=2))
    first = label_example([allowed, late], example, config)
    second = label_example([late, allowed], example, config)
    assert first == second and first.label == "FLAT"


def test_synthetic_session_highs_and_lows_contain_only_seen_prices():
    from tradecopilot.forecast.demo import synthetic_observations

    rows = synthetic_observations(ForecastConfig(), sessions=1, minutes_per_session=30)
    seen = {}
    for row in rows:
        history_for_symbol = seen.setdefault(row.symbol, [])
        history_for_symbol.append(row.last)
        assert row.session_open == history_for_symbol[0]
        assert row.session_high == max(history_for_symbol)
        assert row.session_low == min(history_for_symbol)


def test_dataset_exclusion_counts_cover_unconfigured_and_closed_session_rows():
    from tradecopilot.forecast.dataset import build_dataset

    as_of = datetime(2026, 9, 1, 14, tzinfo=UTC)
    rows = [*history(as_of), observation(as_of, symbol="MSFT"), observation(as_of.replace(hour=12))]
    manifest, examples = build_dataset(rows, ForecastConfig(symbols=("AAPL",)))
    assert manifest.observation_count == 23
    assert manifest.example_count == len(examples) == 1
    assert manifest.exclusions == {
        "observations_outside_universe": 1, "observations_outside_session": 1,
        "insufficient_history": 5, "outcome_unavailable": 1,
    }
