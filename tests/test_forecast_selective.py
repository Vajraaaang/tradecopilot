from datetime import UTC, date, datetime, timedelta
from itertools import pairwise

import numpy as np
import pytest

from tradecopilot.forecast.contracts import FEATURE_NAMES, ForecastConfig, ForecastExample
from tradecopilot.forecast.sessions import session_bounds


def cases(sessions=100, per_day=9):
    config = ForecastConfig(symbols=("AAPL",))
    day = date(2025, 1, 2)
    days = []
    while len(days) < sessions:
        if session_bounds(day):
            days.append(day)
        day += timedelta(days=1)
    result = []
    for day in days:
        at = datetime.combine(day, datetime.min.time(), UTC) + timedelta(hours=15)
        for i in range(per_day):
            result.append(
                ForecastExample(
                    config_id=config.config_id,
                    symbol="AAPL",
                    as_of=at + timedelta(minutes=i),
                    target_time=at + timedelta(minutes=i + 15),
                    session_date=day,
                    anchor_price="100",
                    features=dict.fromkeys(FEATURE_NAMES, 0.0),
                    observation_ids=(f"{day}-{i}",),
                    provenance="historical",
                    label=("DOWN", "FLAT", "UP")[i % 3],
                    target_price="100",
                    target_return_bps=0,
                    label_observed_at=at + timedelta(minutes=i + 15),
                )
            )
    return result


def test_chronological_blocks_folds_purge_and_minimum_history():
    from tradecopilot.forecast.selective import chronological_blocks, tuning_folds

    rows = cases()
    blocks = chronological_blocks(rows)
    assert [len({r.session_date for r in v}) for v in blocks.values()] == [70, 10, 10, 10]
    values = list(blocks.values())
    for left, right in pairwise(values):
        assert max(r.label_observed_at for r in left) < min(r.as_of for r in right)
    folds = tuning_folds(blocks["development"])
    assert len(folds) == 3
    assert [len({r.session_date for r in train}) for train, _ in folds] == [55, 60, 65]
    assert all(len({r.session_date for r in valid}) == 5 for _, valid in folds)
    bad = rows[0].model_copy(update={"label_observed_at": blocks["calibration"][0].as_of})
    purged = chronological_blocks([bad, *rows[1:]])
    assert bad.example_id not in {r.example_id for r in purged["development"]}
    with pytest.raises(ValueError, match="100"):
        chronological_blocks(cases(99))


def test_temperature_only_changes_probability_scale_and_validates_inputs():
    from tradecopilot.forecast.selective import apply_temperature, fit_temperature

    p = np.tile([0.9, 0.05, 0.05], (60, 1))
    y = np.arange(60) % 3
    t = fit_temperature(p, y)
    assert t > 1
    q = apply_temperature(p, t)
    np.testing.assert_allclose(q.sum(axis=1), 1)
    assert np.array_equal(q.argmax(axis=1), p.argmax(axis=1))
    for invalid in (np.array([[np.nan, 0, 1]]), np.array([[0.5, 0.5]]), np.array([[0.2, 0.3, 0.7]])):
        with pytest.raises(ValueError):
            fit_temperature(invalid, np.array([0]))


def test_gate_requires_accuracy_coverage_counts_both_directions_and_sessions():
    from tradecopilot.forecast.selective import gate_qualifies, select_gate, selection_metrics

    rows = cases(10, 30)
    y = np.array([("DOWN", "FLAT", "UP").index(r.label) for r in rows])
    good = np.full((len(rows), 3), 0.05)
    good[np.arange(len(rows)), y] = 0.9
    gate = select_gate(rows, good)
    assert gate["enabled"] and gate["threshold"] is not None
    assert gate_qualifies(selection_metrics(rows, good, gate["threshold"]))
    wrong = np.roll(good, 1, axis=1)
    assert select_gate(rows, wrong)["enabled"] is False
    # A perfect HOLD-only subset cannot qualify the UP/DOWN requirement.
    hold = np.full_like(good, 1 / 3)
    hold[y == 1] = [0.05, 0.9, 0.05]
    assert select_gate(rows, hold)["enabled"] is False
    m = selection_metrics(rows, good, None)
    assert m["selected"] == 0 and m["selective_accuracy"] is None
    assert m["accuracy"] == 1
    with pytest.raises(ValueError, match="align"):
        selection_metrics(rows, good[:-1], 0.6)
