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


def _offline_pilot(tmp_path, monkeypatch):
    import tradecopilot.forecast.kronos_report as report_module
    import tradecopilot.forecast.kronos_run as module
    from tradecopilot.forecast.bars import HistoricalBar, write_bar_dataset

    bars = []
    for day in (1, 2, 5, 6):
        opening = datetime(2026, 10, day, 13, 30, tzinfo=UTC)
        for i in range(390):
            t = opening + timedelta(minutes=i)
            bars.append(HistoricalBar(symbol="AAPL", start_time=t, end_time=t + timedelta(minutes=1),
                                      available_at=t + timedelta(minutes=1), opening=100, high=100, low=100,
                                      close=100, volume=1, source="alpaca_sip_1min_bar"))
    write_bar_dataset(tmp_path / "bars", bars, {
        "fixture": True, "feed": "sip", "selected_symbols": ["AAPL"],
        "paper_endpoint": "https://paper-api.alpaca.markets", "paper_account": {"status": "ACTIVE", "currency": "USD"},
        "clock": {"is_open": False, "next_open": "2026-10-07T13:30:00+00:00",
                  "timestamp": "2026-10-06T20:00:00+00:00"},
    })

    class Clock(datetime):
        seconds = 0.0
        moment = datetime(2026, 10, 6, 22, tzinfo=UTC)

        @classmethod
        def now(cls, zone):
            return cls.moment + timedelta(seconds=cls.seconds)

    monkeypatch.setattr(module, "datetime", Clock)
    monkeypatch.setattr(report_module, "datetime", Clock)
    monkeypatch.setattr(module.time, "monotonic", lambda: Clock.seconds)
    calls = []

    class Engine:
        def __init__(self, variant, cache, threads=1):
            self.variant = variant
            self.metadata = {"variant": variant, "fixture": True}

        def predict(self, history, future, seed=42, samples=20):
            phase = "prospective" if future[0].date() == datetime(2026, 10, 7).date() else "historical"
            calls.append((self.variant, phase))
            paths = np.zeros((samples, 15, 6))
            paths[:, :, :4] = 100
            return paths

    def run(engine=Engine, prospective=True):
        return module.run_pilot(tmp_path / "bars", tmp_path / "cache", tmp_path / "run",
                                engine_factory=engine, prospective=prospective)

    return module, Clock, Engine, calls, run


@pytest.mark.parametrize("phase", ["historical", "prospective"])
@pytest.mark.parametrize("invalid", ["sample_count", "shape", "nonfinite", "nonpositive"])
def test_runner_validates_identical_path_contract_in_both_phases(tmp_path, monkeypatch, phase, invalid):
    from tradecopilot.forecast.kronos_report import load_report

    _, _, engine, _, run = _offline_pilot(tmp_path, monkeypatch)

    class Invalid(engine):
        def predict(self, history, future, seed=42, samples=20):
            paths = super().predict(history, future, seed, samples)
            current = "prospective" if future[0].day == 7 else "historical"
            if self.variant != "mini" or current != phase:
                return paths
            if invalid == "sample_count":
                return paths[:1]
            if invalid == "shape":
                return paths[:, :14]
            paths[0, 0, 0 if invalid == "nonfinite" else 3] = np.inf if invalid == "nonfinite" else 0
            return paths

    report = load_report(run(Invalid))
    rows = [r for r in report["cases"] if r["group"] == phase]
    assert rows and all(r["forecasts"]["mini"]["status"] == "error" for r in rows)
    assert all(r["forecasts"]["small"]["status"] == "ok" for r in rows)
    if phase == "historical":
        assert report["models"]["mini"]["metrics"]["errors"] == 20
    with np.load(tmp_path / "run" / "paths.npz") as paths:
        assert not any(name.startswith("pending_mini" if phase == "prospective" else "mini_") for name in paths.files)


@pytest.mark.parametrize("phase", ["historical", "prospective"])
def test_metadata_change_fails_closed_with_terminal_private_safe_progress(tmp_path, monkeypatch, phase):
    import json

    _, _, engine, _, run = _offline_pilot(tmp_path, monkeypatch)

    class Changed(engine):
        def predict(self, history, future, seed=42, samples=20):
            paths = super().predict(history, future, seed, samples)
            current = "prospective" if future[0].day == 7 else "historical"
            if current == phase:
                self.metadata["private_path"] = "PRIVATESENTINEL changed during model call"
            return paths

    with pytest.raises(ValueError, match="provenance"):
        run(Changed)
    assert not (tmp_path / "run").exists()
    progress = json.loads((tmp_path / "run-progress.json").read_bytes())
    assert progress == {"stage": "failed", "phase": f"{phase}_provenance", "model": "mini",
                        "error_type": "ValueError"}
    assert "PRIVATESENTINEL" not in str(progress)


def test_source_provenance_drift_has_terminal_private_safe_progress(tmp_path, monkeypatch):
    import json

    module, _, _, _, run = _offline_pilot(tmp_path, monkeypatch)
    original = module.source_provenance()
    changed = original | {"source_sha256": "b" * 64, "private_path": "PRIVATESENTINEL changed source"}
    versions = iter((original, changed))
    monkeypatch.setattr(module, "source_provenance", lambda: next(versions))
    with pytest.raises(ValueError, match="implementation changed"):
        run()
    assert not (tmp_path / "run").exists()
    progress = json.loads((tmp_path / "run-progress.json").read_bytes())
    assert progress == {"stage": "failed", "phase": "source_provenance", "error_type": "ValueError"}
    assert "PRIVATESENTINEL" not in str(progress)


def test_late_first_target_forecasts_are_retained_as_errors(tmp_path, monkeypatch):
    from tradecopilot.forecast.kronos_report import load_report

    _, clock, engine, _, run = _offline_pilot(tmp_path, monkeypatch)

    class Late(engine):
        def predict(self, history, future, seed=42, samples=20):
            paths = super().predict(history, future, seed, samples)
            if future[0].day == 7:
                clock.moment = datetime(2026, 10, 7, 13, 32, tzinfo=UTC)
            return paths

    report = load_report(run(Late))
    row = next(r for r in report["cases"] if r["group"] == "prospective")
    assert all(f["status"] == "error" for f in row["forecasts"].values())
    assert row["actual_close"] is None and row["outcome_status"] == "pending"


def test_initialization_failure_retains_every_case_and_runs_other_model_and_controls(tmp_path, monkeypatch):
    from tradecopilot.forecast.kronos_report import load_report

    _, _, engine, calls, run = _offline_pilot(tmp_path, monkeypatch)

    class Unavailable(engine):
        def __init__(self, variant, cache, threads=1):
            if variant == "mini":
                raise RuntimeError("PRIVATESENTINEL initialization details")
            super().__init__(variant, cache, threads)

    report = load_report(run(Unavailable))
    model = report["models"]["mini"]
    assert model["metrics"]["eligible"] == model["metrics"]["errors"] == 20 and model["metrics"]["scored"] == 0
    assert model["initialization"]["error_type"] == "RuntimeError"
    assert all(f["phase"] == "initialization" for f in model["failures"])
    assert not {"source", "model", "tokenizer"} & model["metadata"].keys()
    assert "PRIVATESENTINEL" not in str(report)
    assert calls == [("small", "historical")] * 20 + [("small", "prospective")]
    assert all(report["models"][name]["metrics"]["scored"] == 20 for name in ("small", "persistence", "momentum"))
    assert len(report["cases"]) == 21


@pytest.mark.parametrize("overrun", ["initialization", "historical", "prospective"])
def test_budget_covers_own_phases_checks_returns_and_retains_cohort(tmp_path, monkeypatch, overrun):
    from tradecopilot.forecast.kronos_report import load_report

    _, clock, engine, calls, run = _offline_pilot(tmp_path, monkeypatch)

    class Slow(engine):
        def __init__(self, variant, cache, threads=1):
            super().__init__(variant, cache, threads)
            if variant == "mini":
                clock.seconds += 901 if overrun == "initialization" else 200 if overrun == "prospective" else 0

        def predict(self, history, future, seed=42, samples=20):
            paths = super().predict(history, future, seed, samples)
            if self.variant == "mini":
                if overrun == "historical":
                    clock.seconds += 901
                elif overrun == "prospective":
                    clock.seconds += 101 if future[0].day == 7 else 30
            return paths

    report = load_report(run(Slow))
    mini = report["models"]["mini"]
    assert mini["metrics"]["eligible"] == 20 and report["models"]["small"]["metrics"]["scored"] == 20
    historical_calls = sum(v == "mini" and p == "historical" for v, p in calls)
    prospective_calls = sum(v == "mini" and p == "prospective" for v, p in calls)
    assert historical_calls == {"initialization": 0, "historical": 1, "prospective": 20}[overrun]
    assert prospective_calls == (1 if overrun == "prospective" else 0)
    if overrun != "prospective":
        assert mini["metrics"]["errors"] == 20 and mini["metrics"]["scored"] == 0
    row = next(r for r in report["cases"] if r["group"] == "prospective")
    assert row["forecasts"]["mini"]["status"] == "error" and row["forecasts"]["mini"]["error"] == "TimeoutError"
    assert row["forecasts"]["small"]["status"] == "ok"
    assert mini["elapsed_seconds"] >= 900
    assert report["timing"]["budget_enforcement"] == "admission_and_post_return; blocking_calls_not_cancelled"


@pytest.mark.parametrize("explicit_factory", [False, True])
def test_native_cache_integrity_initialization_error_is_not_demoted_to_unavailable(tmp_path, monkeypatch,
                                                                                 explicit_factory):
    import tradecopilot.forecast.kronos as native

    module, _, _, _, _ = _offline_pilot(tmp_path, monkeypatch)

    class Corrupt(native.KronosEngine):
        def __init__(self, *args, **kwargs):
            raise ValueError("invalid local Kronos cache/source: fixture hash mismatch")

    monkeypatch.setattr(native, "KronosEngine", Corrupt)
    with pytest.raises(ValueError, match="cache/source"):
        module.run_pilot(tmp_path / "bars", tmp_path / "cache", tmp_path / "run",
                         engine_factory=Corrupt if explicit_factory else None)
    assert not (tmp_path / "run").exists()


def test_model_historical_phase_includes_scoring_time_in_next_call_admission(tmp_path, monkeypatch):
    from tradecopilot.forecast.kronos_report import load_report

    module, clock, _, calls, run = _offline_pilot(tmp_path, monkeypatch)
    score = module.price_metrics
    scoring_calls = []

    def slow_first_scoring(*args, **kwargs):
        result = score(*args, **kwargs)
        scoring_calls.append(True)
        if len(scoring_calls) == 1:
            clock.seconds += 901
        return result

    monkeypatch.setattr(module, "price_metrics", slow_first_scoring)
    report = load_report(run())
    assert ("mini", "prospective") not in calls
    assert ("small", "prospective") in calls
    assert report["models"]["mini"]["timing"]["historical_seconds"] == 901
    assert report["models"]["mini"]["metrics"]["scored"] == 20


def test_phase_timing_and_complete_progress_include_publication_return(tmp_path, monkeypatch):
    import json

    from tradecopilot.forecast.kronos_report import load_report

    module, clock, engine, _, run = _offline_pilot(tmp_path, monkeypatch)

    class Timed(engine):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            clock.seconds += 2

        def predict(self, history, future, seed=42, samples=20):
            paths = super().predict(history, future, seed, samples)
            clock.seconds += 5 if future[0].day == 7 else 3
            return paths

    write = module.write_report
    observed = []

    def publish(*args, **kwargs):
        observed.append(json.loads((tmp_path / "run-progress.json").read_bytes())["stage"])
        path = write(*args, **kwargs)
        clock.seconds += 4
        return path

    monkeypatch.setattr(module, "write_report", publish)
    report = load_report(run(Timed))
    progress = json.loads((tmp_path / "run-progress.json").read_bytes())
    assert observed == ["publication"] and progress["stage"] == "complete"
    assert progress["elapsed_seconds"] == 138 and progress["timing"]["publication_seconds"] == 4
    utc_elapsed = datetime.fromisoformat(progress["finished_at"]) - datetime.fromisoformat(progress["started_at"])
    assert utc_elapsed == timedelta(seconds=138)
    assert report["timing"]["elapsed_seconds_before_publication"] == 134
    for variant in ("mini", "small"):
        assert report["models"][variant]["timing"] == {
            "initialization_seconds": 2, "historical_seconds": 60, "prospective_seconds": 5,
        }
        assert report["models"][variant]["elapsed_seconds"] == 67


def test_publication_failure_records_failed_phase_instead_of_complete(tmp_path, monkeypatch):
    import json

    module, _, _, _, run = _offline_pilot(tmp_path, monkeypatch)

    def fail(*args, **kwargs):
        raise ValueError("publication failed fixture")

    monkeypatch.setattr(module, "write_report", fail)
    with pytest.raises(ValueError, match="publication"):
        run()
    progress = json.loads((tmp_path / "run-progress.json").read_bytes())
    assert progress["stage"] == "failed" and progress["phase"] == "publication"
    assert progress["error_type"] == "ValueError"
    assert not (tmp_path / "run").exists()
