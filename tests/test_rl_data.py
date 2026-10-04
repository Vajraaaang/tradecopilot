import json
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from tradecopilot.forecast.bars import HistoricalBar, write_bar_dataset
from tradecopilot.forecast.contracts import content_hash
from tradecopilot.forecast.ohlcv_features import OHLCV_FEATURE_NAMES, OhlcvFeatureBuilder
from tradecopilot.forecast.sessions import session_bounds
from tradecopilot.rl.data import load_prepared, prepare_data

SYMBOLS = ["AAPL", "AMZN", "MSFT", "NFLX", "TSLA"]


def fixture(tmp_path: Path, test_price: int = 100, *, late: bool = False,
            delay_seconds: float = 60) -> tuple[Path, Path]:
    reg = {"fixture": True, "forecast_profile": "MARKET_ONLY", "symbols": SYMBOLS,
           "source_range": {"start": "2025-11-03", "end_exclusive": "2025-11-07", "provider": "Alpaca",
                            "feed": "sip", "adjustment": "raw"},
           "splits": {"train": ["2025-11-04"] * 2, "tune": ["2025-11-05"] * 2,
                      "test": ["2025-11-06"] * 2}, "environment": {"warmup_minutes": 60}}
    reg["registration_id"] = content_hash(reg)
    registration = tmp_path / "registration.json"
    tmp_path.mkdir(parents=True, exist_ok=True)
    registration.write_text(json.dumps(reg))
    bars = []
    for day in range(3, 7):
        bounds = session_bounds(date(2025, 11, day))
        assert bounds
        for symbol in SYMBOLS:
            for minute in range(65):
                start = bounds[0] + timedelta(minutes=minute)
                price = Decimal(test_price if day == 6 else 100) + Decimal(minute) / 100
                end = start + timedelta(minutes=1)
                bars.append(HistoricalBar(symbol=symbol, start_time=start, end_time=end,
                                          available_at=(end + timedelta(seconds=delay_seconds)
                                                        if late and minute == 62 else end),
                                          opening=price, high=price, low=price, close=price,
                                          volume=Decimal(test_price if day == 6 else 100),
                                          source="alpaca_sip_1min_bar"))
    metadata = {"fixture": True, "provider": "Alpaca", "feed": "sip", "adjustment_policy": "raw",
                "start_date": "2025-11-03", "end_date_exclusive": "2025-11-07", "selected_symbols": SYMBOLS,
                "availability_assumption": "synthetic assumed bar-end delivery"}
    write_bar_dataset(tmp_path / "bars", bars, metadata)
    return tmp_path / "bars", registration


def prepare(tmp_path: Path, **kwargs):
    bars, reg = fixture(tmp_path, **kwargs)
    target = tmp_path / "prepared"
    prepare_data(bars, reg, target)
    return target


def seal(tmp_path: Path, metadata: dict, **changes) -> Path:
    value = {"stage": "selection_frozen", "registration_id": metadata["registration_id"],
             "prepared_data_id": metadata["data_id"], **changes}
    value["selection_id"] = content_hash(value)
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(value))
    return path


def test_train_only_normalization_and_disjoint_dates(tmp_path):
    target = prepare(tmp_path / "original")
    changed = prepare(tmp_path / "changed", test_price=900)
    train, metadata = load_prepared(target, "train")
    train_changed, other = load_prepared(changed, "train")
    assert metadata["normalizer"] == other["normalizer"]
    assert metadata["role_counts"] == {"train": 5, "tune": 5, "test": 5}
    assert metadata["normalizer"]["fit_dates"] == ["2025-11-04"]
    assert all(episode.features.shape == (65, 115) for episode in train)
    assert all(np.array_equal(a.features, b.features) for a, b in zip(train, train_changed, strict=True))
    tune, _ = load_prepared(target, "tune")
    test, _ = load_prepared(target, "test", selection_path=seal(tmp_path, metadata))
    assert not ({e.session_date for e in train} & {e.session_date for e in tune})
    assert not ({e.session_date for e in tune} & {e.session_date for e in test})
    assert all(not any("target" in name or "label" in name or "future" in name for name in e.feature_names)
               for e in train)


def test_label_free_asof_parity_and_missing_masks(tmp_path):
    from tradecopilot.forecast.bars import load_bar_dataset
    target = prepare(tmp_path, late=True)
    episodes, metadata = load_prepared(target, "train")
    bars, _ = load_bar_dataset(tmp_path / "bars")
    episode = episodes[0]
    bar = next(b for b in bars if b.symbol == episode.symbol and b.end_time.timestamp() == episode.ends[61])
    record = OhlcvFeatureBuilder(bars).build_as_of(bar.symbol, bar.end_time, metadata["config_id"])
    mean, scale = np.asarray(metadata["normalizer"]["mean"]), np.asarray(metadata["normalizer"]["scale"])
    for index, name in enumerate(OHLCV_FEATURE_NAMES):
        value = record.values[name]
        if value is not None:
            assert episode.features[61, index] == pytest.approx((value - mean[index]) / scale[index], abs=1e-5)
    assert np.array_equal(episode.features[62, 55:110], np.ones(55))
    assert np.array_equal(episode.features[62, :55], np.zeros(55))
    assert episode.features[0, 55] == 1  # Missing one-minute return has no forward fill.
    assert np.isfinite(episode.features).all()


def test_test_role_requires_matching_hashed_selection(tmp_path):
    target = prepare(tmp_path)
    _, metadata = load_prepared(target, "train")
    with pytest.raises(ValueError, match="locked"):
        load_prepared(target, "test")
    with pytest.raises(ValueError, match="seal mismatch"):
        load_prepared(target, "test", selection_path=seal(tmp_path, metadata, prepared_data_id="wrong"))
    path = seal(tmp_path, metadata)
    value = json.loads(path.read_text())
    value["stage"] = "not_frozen"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="selection_id"):
        load_prepared(target, "test", selection_path=path)
    assert len(load_prepared(target, "test", selection_path=seal(tmp_path, metadata))[0]) == 5


def test_subsecond_late_bar_cannot_execute_at_scheduled_end(tmp_path):
    from tradecopilot.rl.contracts import SimConfig
    from tradecopilot.rl.env import TradeCopilotEnv

    target = prepare(tmp_path, late=True, delay_seconds=0.5)
    episodes, _ = load_prepared(target, "train")
    episode = episodes[0]
    environment = TradeCopilotEnv([episode], SimConfig())
    environment.reset(seed=42)
    _, _, _, truncated, info = environment.step(2)
    assert truncated  # The next execution minute's bar is still unavailable at its end.
    assert not info["fills"]
    assert environment.state.shares == 0
    assert episode.available_at[62] == episode.ends[62] + 1
    assert np.array_equal(episode.features[62, 55:110], np.ones(55))


@pytest.mark.parametrize("filename", ["manifest.json", "episodes.npz"])
def test_immutable_and_tamper_fail_closed(tmp_path, filename):
    target = prepare(tmp_path)
    with pytest.raises(ValueError, match="immutable"):
        prepare_data(tmp_path / "bars", tmp_path / "registration.json", target)
    path = target / filename
    if filename.endswith("json"):
        value = json.loads(path.read_text())
        value["episodes"][0]["role"] = "test"
        path.write_text(json.dumps(value))
    else:
        path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match=r"data_id|blob hash"):
        load_prepared(target, "train")


@pytest.mark.parametrize("change", ["overlap", "bounds", "symbols", "registration_hash"])
def test_registration_source_validation(tmp_path, change):
    bars, registration = fixture(tmp_path)
    reg = json.loads(registration.read_text())
    if change == "overlap":
        reg["splits"]["test"] = reg["splits"]["train"]
    elif change == "bounds":
        reg["source_range"]["start"] = "2025-11-02"
    elif change == "symbols":
        reg["symbols"][0] = "GOOG"
    else:
        reg["registration_id"] = "invalid"
    if change != "registration_hash":
        reg["registration_id"] = content_hash({k: v for k, v in reg.items() if k != "registration_id"})
    registration.write_text(json.dumps(reg))
    with pytest.raises(ValueError):
        prepare_data(bars, registration, tmp_path / "prepared")
