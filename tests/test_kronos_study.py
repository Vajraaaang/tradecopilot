"""Independent fixtures for causal Kronos cohorts and price metrics."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest

from tradecopilot.forecast.bars import HistoricalBar


def bars():
    start = datetime(2026, 10, 1, 13, 30, tzinfo=UTC)
    result = []
    for i in range(100):
        price = Decimal("100") + Decimal(i) / 100
        result.append(
            HistoricalBar(
                symbol="AAPL",
                start_time=start + timedelta(minutes=i),
                end_time=start + timedelta(minutes=i + 1),
                available_at=start + timedelta(minutes=i + 1),
                opening=price,
                high=price,
                low=price,
                close=price,
                volume=100,
                source="alpaca_sip_1min_bar",
            )
        )
    return result


def test_case_identity_inputs_and_target_are_exact_and_future_changes_do_not_rewrite_inputs():
    from tradecopilot.forecast.kronos_study import build_cases

    source = bars()
    cases, catalog = build_cases(source, ("AAPL",), (source[0].start_time.date(),), offsets=(61,))
    assert len(cases) == len(catalog) == 1
    case = cases[0]
    assert len(case.history) == 60
    assert case.future_times[-1] == case.as_of + timedelta(minutes=15)
    changed = [
        b.model_copy(update={"opening": b.opening * 2, "high": b.high * 2, "low": b.low * 2, "close": b.close * 2})
        if b.end_time > case.as_of
        else b
        for b in source
    ]
    other, _ = build_cases(changed, ("AAPL",), (source[0].start_time.date(),), offsets=(61,))
    assert case.case_id == other[0].case_id
    assert case.history == other[0].history
    assert case.actual != other[0].actual


def test_missing_or_late_input_is_retained_in_catalog_and_never_filled():
    from tradecopilot.forecast.kronos_study import build_cases

    source = bars()
    for changed in (
        source[:30] + source[31:],
        [
            b.model_copy(update={"available_at": b.end_time + timedelta(hours=2)}) if i == 30 else b
            for i, b in enumerate(source)
        ],
    ):
        cases, catalog = build_cases(changed, ("AAPL",), (source[0].start_time.date(),), offsets=(61,))
        assert not cases
        assert len(catalog) == 1 and not catalog[0]["eligible"]
        assert catalog[0]["reason"] in ("incomplete_input_window", "late_input")


def test_baselines_use_past_prices_only_and_linear_trend_has_an_independent_oracle():
    from tradecopilot.forecast.kronos_study import baseline_paths, build_cases

    case = build_cases(bars(), ("AAPL",), (bars()[0].start_time.date(),), offsets=(61,))[0][0]
    outputs = baseline_paths(case.history, 15)
    assert outputs["persistence"].shape == (1, 15, 6)
    assert np.all(outputs["persistence"][0, :, 3] == 100.60)
    assert outputs["momentum"][0, -1, 3] == pytest.approx(100.75)


def test_price_metrics_include_failed_cases_and_do_not_label_raw_samples_calibrated():
    from tradecopilot.forecast.kronos_study import build_cases, price_metrics

    case = build_cases(bars(), ("AAPL",), (bars()[0].start_time.date(),), offsets=(61,))[0][0]
    paths = np.zeros((20, 15, 6))
    paths[:, :, :4] = np.asarray([float(b.close) for b in case.actual])[None, :, None]
    metrics = price_metrics((case,), {case.case_id: paths}, sample_intervals=True)
    assert metrics["eligible"] == metrics["scored"] == 1
    assert metrics["terminal_mae_bps"] == pytest.approx(0, abs=1e-9)
    assert metrics["path_mae_bps"] == pytest.approx(0, abs=1e-9)
    assert metrics["direction_accuracy"] == metrics["raw_interval_coverage"] == 1
    assert metrics["intervals_calibrated"] is False
    failed = price_metrics((case,), {case.case_id: None}, sample_intervals=True)
    assert failed["eligible"] == 1 and failed["errors"] == 1 and failed["scored"] == 0
    assert failed["direction_accuracy_errors_as_incorrect"] == 0
    with pytest.raises(ValueError):
        price_metrics((case,), {}, sample_intervals=True)


def test_closed_market_future_is_explicitly_next_session_and_not_15_wall_clock_minutes():
    from tradecopilot.forecast.kronos_study import future_grid

    history = bars()[-60:]
    receipt = datetime(2026, 10, 1, 21, tzinfo=UTC)
    future, kind = future_grid(history, {"is_open": False, "next_open": "2026-10-02T13:30:00+00:00"}, receipt)
    assert kind == "next_session_open"
    assert future[0] == datetime(2026, 10, 2, 13, 31, tzinfo=UTC)
    assert future[-1] == datetime(2026, 10, 2, 13, 45, tzinfo=UTC)
    assert future[-1] - history[-1].end_time > timedelta(minutes=15)


def test_open_market_future_must_fit_session_and_have_a_fresh_anchor():
    from tradecopilot.forecast.kronos_study import future_grid

    history = bars()[-60:]
    future, kind = future_grid(history, {"is_open": True}, history[-1].end_time + timedelta(seconds=10))
    assert kind == "intraday"
    assert future[-1] == history[-1].end_time + timedelta(minutes=15)
    with pytest.raises(ValueError):
        future_grid(history, {"is_open": True}, history[-1].end_time + timedelta(minutes=5))
