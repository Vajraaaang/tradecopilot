from datetime import UTC, datetime, timedelta

import numpy as np
import pytest


@pytest.mark.parametrize("scenario", ("history", "invalid_control", "stale_prospective"))
def test_study_predicts_input_only_then_scores_and_records_pending_without_overwriting(tmp_path, monkeypatch, scenario):
    from tradecopilot.forecast.bars import HistoricalBar, write_bar_dataset
    from tradecopilot.forecast.kronos_report import load_report
    from tradecopilot.forecast.kronos_run import run_pilot

    bars = []
    for d in (1, 2, 5, 6):
        start = datetime(2026, 10, d, 13, 30, tzinfo=UTC)
        for i in range(390):
            p = 100 + i / 100
            if scenario == "invalid_control" and i >= 56:
                p = 20
            bars.append(
                HistoricalBar(
                    symbol="AAPL",
                    start_time=start + timedelta(minutes=i),
                    end_time=start + timedelta(minutes=i + 1),
                    available_at=start + timedelta(minutes=i + 1),
                    opening=p,
                    high=p,
                    low=p,
                    close=p,
                    volume=100,
                    source="alpaca_sip_1min_bar",
                )
            )
    meta = {
        "fixture": True,
        "feed": "sip",
        "selected_symbols": ["AAPL"],
        "paper_endpoint": "https://paper-api.alpaca.markets",
        "paper_account": {"status": "ACTIVE", "currency": "USD"},
        "clock": {"is_open": False, "next_open": "2090-10-02T13:30:00+00:00", "timestamp": "2026-10-06T20:00:00+00:00"},
    }
    write_bar_dataset(tmp_path / "bars", bars, meta)
    if scenario == "stale_prospective":
        import tradecopilot.forecast.kronos_run as module

        meta["clock"]["is_open"] = True
        write_bar_dataset(tmp_path / "stale-bars", bars, meta)

        class Clock:
            @staticmethod
            def now(zone):
                return datetime(2026, 10, 6, 20, 5, tzinfo=UTC)

        monkeypatch.setattr(module, "datetime", Clock)
    calls = []

    class FakeEngine:
        def __init__(self, variant, cache, threads=1):
            self.metadata = {"variant": variant, "fixture": True}

        def predict(self, history, future, seed=42, samples=20):
            assert len(history) == 60 and all(b.end_time <= history[-1].end_time for b in history)
            calls.append((history[-1].end_time, len(future)))
            paths = np.zeros((samples, len(future), 6))
            paths[:, :, :4] = float(history[-1].close)
            return paths

    path = run_pilot(
        tmp_path / ("stale-bars" if scenario == "stale_prospective" else "bars"),
        tmp_path / "cache",
        tmp_path / "run",
        samples=20,
        engine_factory=FakeEngine,
        prospective=scenario == "stale_prospective",
    )
    report = load_report(path)
    assert report["evidence_mode"] == "synthetic_contract_fixture"
    assert report["catalog_counts"] == {"planned": 20, "eligible": 20, "excluded": 0}
    assert len(calls) == 40
    assert all(m["metrics"]["eligible"] == 20 for m in report["models"].values())
    assert all(c["group"] == "historical" for c in report["cases"])
    assert report["broker_orders"] == report["jev_calls"] == 0
    if scenario == "invalid_control":
        assert report["models"]["persistence"]["metrics"]["scored"] == 20
        assert report["models"]["momentum"]["metrics"]["errors"] == 4
        assert len(report["models"]["momentum"]["failures"]) == 4
    if scenario == "stale_prospective":
        assert report["prospective_unavailable"] == [{"symbol": "AAPL", "reason": "intraday context is stale"}]
    assert report["protocol"]["implementation"]["source_sha256"]
