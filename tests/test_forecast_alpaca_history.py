"""Historical Alpaca import uses fabricated pages only; no credentials or network access."""

import asyncio
import hashlib
import importlib.util
import json
import traceback
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from tradecopilot.forecast.bars import load_bar_dataset
from tradecopilot.forecast.contracts import ForecastConfig

START = date(2026, 4, 1)
END = date(2026, 4, 4)
CONFIG = ForecastConfig(symbols=("AAPL", "MSFT"))
KEY = "private-api-key"
SECRET = "private-api-secret"


def bar(at="2026-04-01T13:30:00Z", **changes):
    return {"t": at, "o": "100.0010", "h": "101.1234", "l": "99.9876", "c": "100.7654", "v": 1234, **changes}


def run(tmp_path, pages, *, config=CONFIG, start=START, end=END):
    from tradecopilot.forecast.alpaca_history import download_history

    requests = []

    def serve(request):
        requests.append(request)
        page = pages[len(requests) - 1]
        if isinstance(page, Exception):
            raise page
        if isinstance(page, httpx.Response):
            return page
        return httpx.Response(200, json=page)

    path = asyncio.run(download_history(
        config, start, end, tmp_path / "import", KEY, SECRET, transport=httpx.MockTransport(serve),
    ))
    return path, requests


def test_paginated_import_preserves_values_availability_and_private_provenance(tmp_path):
    pages = [
        {"bars": {"MSFT": [bar()], "AAPL": [bar(), bar("2026-04-01T20:00:00Z")]}, "next_page_token": "page-two"},
        {"bars": {"AAPL": [bar("2026-04-01T19:59:00Z"), bar()]}, "next_page_token": None},
    ]
    path, requests = run(tmp_path, pages)
    assert path == tmp_path / "import" / "bars" / "manifest.json"
    bars, manifest = load_bar_dataset(path.parent)
    assert [item.symbol for item in bars] == ["AAPL", "MSFT", "AAPL"]
    assert bars[0].opening == Decimal("100.0010") and str(bars[0].opening) == "100.0010"
    assert bars[0].available_at == bars[0].end_time == bars[0].start_time + timedelta(minutes=1)
    assert bars[-1].end_time == datetime(2026, 4, 1, 20, tzinfo=UTC)
    assert all(item.source == "alpaca_sip_1min_bar" and item.provenance == "historical" for item in bars)
    metadata = manifest["source_metadata"]
    assert metadata["provider"] == "Alpaca" and metadata["feed"] == "sip"
    assert metadata["adjustment_policy"] == "raw" and metadata["calendar"] == "XNYS"
    assert metadata["bar_count"] == 3 and metadata["downloaded_bar_count"] == 5
    assert metadata["exclusions"] == {"outside_regular_session": 1, "duplicate_identical": 1}
    assert "bar end" in metadata["availability_assumption"]
    assert metadata["start_date"] == START.isoformat() and metadata["end_date_exclusive"] == END.isoformat()
    assert len(metadata["downloads"]) == 2
    for index, record in enumerate(metadata["downloads"]):
        raw_path = tmp_path / "import" / record["path"]
        raw = raw_path.read_bytes()
        assert json.loads(raw) == pages[index]
        assert hashlib.sha256(raw).hexdigest() == record["sha256"]
        assert len(raw) == record["bytes"]
        assert datetime.fromisoformat(record["retrieved_at"]).utcoffset() == timedelta(0)
        assert raw_path.stat().st_mode & 0o077 == 0
    assert (tmp_path / "import").stat().st_mode & 0o077 == 0
    for request in requests:
        assert str(request.url).startswith("https://data.alpaca.markets/v2/stocks/bars?")
        assert request.headers["APCA-API-KEY-ID"] == KEY and request.headers["APCA-API-SECRET-KEY"] == SECRET
        assert KEY not in str(request.url) and SECRET not in str(request.url)
        assert request.url.params["symbols"] == "AAPL,MSFT"
        assert request.url.params["timeframe"] == "1Min"
        assert request.url.params["feed"] == "sip" and request.url.params["adjustment"] == "raw"
        assert request.url.params["limit"] == "10000"
        assert request.url.params["start"] == "2026-04-01T00:00:00-04:00"
        assert request.url.params["end"] == "2026-04-03T23:59:59.999999-04:00"
    assert "page_token" not in requests[0].url.params and requests[1].url.params["page_token"] == "page-two"
    assert KEY not in path.read_text() and SECRET not in path.read_text()


def test_inclusive_wire_end_precedes_exclusive_date_and_midnight_bar_is_rejected(tmp_path):
    from tradecopilot.forecast.alpaca_history import download_history

    requests = []

    def serve(request):
        requests.append(request)
        return httpx.Response(200, json={"bars": {"AAPL": [bar("2026-04-04T00:00:00-04:00")]}})

    with pytest.raises(ValueError, match="range"):
        asyncio.run(download_history(
            CONFIG, START, END, tmp_path / "import", KEY, SECRET, transport=httpx.MockTransport(serve),
        ))
    boundary = datetime.fromisoformat("2026-04-04T00:00:00-04:00")
    assert datetime.fromisoformat(requests[0].url.params["end"]) == boundary - timedelta(microseconds=1)
    assert not (tmp_path / "import").exists()


def test_filters_holiday_weekend_and_premarket(tmp_path):
    rows = [bar(at) for at in [
        "2026-04-01T13:29:00Z", "2026-04-01T13:30:00Z", "2026-04-01T19:59:00Z",
        "2026-04-01T20:00:00Z", "2026-04-03T13:30:00Z", "2026-04-04T13:30:00Z",
    ]]
    path, _ = run(tmp_path, [{"bars": {"AAPL": rows}}], end=date(2026, 4, 5))
    bars, manifest = load_bar_dataset(path.parent)
    assert len(bars) == 2
    assert manifest["source_metadata"]["exclusions"] == {"outside_regular_session": 4}


@pytest.mark.parametrize("at", [
    "2026-03-31T23:59:00-04:00", "2026-04-04T00:00:00-04:00",
    "2026-04-01T13:30:01Z", "2026-04-01T13:30:00.000001Z", "2026-04-01T13:30:00", 123,
])
def test_rejects_out_of_range_unaligned_and_naive_timestamps(tmp_path, at):
    with pytest.raises(ValueError, match="bar"):
        run(tmp_path, [{"bars": {"AAPL": [bar(at)]}}])


@pytest.mark.parametrize("changes", [
    {"o": "NaN"}, {"h": "Infinity"}, {"l": "-1"}, {"c": 0}, {"v": -1},
    {"o": True}, {"h": "99"}, {"l": "101"}, {"c": None}, {"v": "NaN"},
])
def test_rejects_malformed_ohlcv(tmp_path, changes):
    with pytest.raises(ValueError, match="bar"):
        run(tmp_path, [{"bars": {"AAPL": [bar(**changes)]}}])


@pytest.mark.parametrize("payload", [
    {}, [], {"bars": []}, {"bars": {"AAPL": {}}}, {"bars": {"AAPL": [None]}},
    {"bars": {"TSLA": [bar()]}}, {"bars": {"AAPL": [{"t": "2026-04-01T13:30:00Z"}]}},
])
def test_rejects_malformed_page_shape(tmp_path, payload):
    with pytest.raises(ValueError):
        run(tmp_path, [payload])


def test_conflicting_duplicate_is_rejected_across_pages(tmp_path):
    with pytest.raises(ValueError, match="conflict"):
        run(tmp_path, [
            {"bars": {"AAPL": [bar()]}, "next_page_token": "next"},
            {"bars": {"AAPL": [bar(v=999)]}},
        ])


@pytest.mark.parametrize("token", [[], {}, 1, True, ""])
def test_nonstring_or_empty_pagination_token_rejected(tmp_path, token):
    with pytest.raises(ValueError, match="token"):
        run(tmp_path, [{"bars": {}, "next_page_token": token}])


def test_pagination_loop_rejected_without_extra_call(tmp_path):
    calls = []
    from tradecopilot.forecast.alpaca_history import download_history

    def serve(request):
        calls.append(request)
        return httpx.Response(200, json={"bars": {}, "next_page_token": "same"})

    with pytest.raises(ValueError, match="loop"):
        asyncio.run(download_history(CONFIG, START, END, tmp_path / "import", KEY, SECRET,
                                     transport=httpx.MockTransport(serve)))
    assert len(calls) == 2


@pytest.mark.parametrize("start,end", [
    (START, START), (END, START), (date(2024, 1, 1), date(2025, 1, 2)),
    (START, datetime.now(UTC).date()), (START, datetime.now(UTC).date() + timedelta(days=1)),
])
def test_date_bounds_rejected_before_network(tmp_path, start, end):
    with pytest.raises(ValueError, match=r"date|range|historical"):
        run(tmp_path, [], start=start, end=end)
    assert not (tmp_path / "import").exists()


def test_more_than_five_symbols_rejected_before_network(tmp_path):
    with pytest.raises(ValueError, match=r"five|5"):
        run(tmp_path, [], config=ForecastConfig(symbols=("AAPL", "MSFT", "AMZN", "NFLX", "TSLA", "NVDA")))


def test_model_copy_cannot_bypass_config_validation(tmp_path):
    with pytest.raises(ValueError):
        run(tmp_path, [], config=CONFIG.model_copy(update={"symbols": ("../invalid",)}))


@pytest.mark.parametrize("limit, pages, match", [
    ("MAX_PAGES", [{"bars": {}, "next_page_token": "more"}], "page"),
    ("MAX_RESPONSE_BYTES", [{"bars": {"AAPL": [bar()]}}], "size|byte"),
    ("MAX_BAR_ROWS", [{"bars": {"AAPL": [bar(), bar()]}}], "row|count"),
])
def test_hard_budgets_rejected(tmp_path, monkeypatch, limit, pages, match):
    from tradecopilot.forecast import alpaca_history

    monkeypatch.setattr(alpaca_history, limit, 1)
    with pytest.raises(ValueError, match=match):
        run(tmp_path, pages)


@pytest.mark.parametrize("status", [301, 401, 403, 429, 500])
def test_http_errors_are_sanitized_and_never_followed_or_retried(tmp_path, status):
    response = httpx.Response(status, text=f"{KEY}:{SECRET}", headers={"Location": "https://evil.example/"})
    with pytest.raises(ValueError, match=str(status)) as caught:
        run(tmp_path, [response])
    rendered = "".join(traceback.format_exception(caught.value))
    assert KEY not in rendered and SECRET not in rendered and "evil.example" not in rendered


@pytest.mark.parametrize("response", [
    httpx.Response(200, text=f"not JSON {KEY} {SECRET}"),
    httpx.Response(200, json={"bars": {"AAPL": [bar(o=SECRET)]}}),
    httpx.ConnectError(f"network {KEY} {SECRET}"),
])
def test_payload_and_transport_errors_never_reveal_credentials(tmp_path, response):
    with pytest.raises(ValueError) as caught:
        run(tmp_path, [response])
    rendered = "".join(traceback.format_exception(caught.value))
    assert KEY not in rendered and SECRET not in rendered


def test_existing_output_is_refused_without_network_or_overwrite(tmp_path):
    output = tmp_path / "import"
    output.mkdir()
    evidence = output / "keep.txt"
    evidence.write_text("prior research")
    with pytest.raises(ValueError, match=r"immutable|exist"):
        run(tmp_path, [])
    assert evidence.read_text() == "prior research"
    assert list(output.iterdir()) == [evidence]


def test_repeat_import_is_refused_even_if_identical(tmp_path):
    path, _ = run(tmp_path, [{"bars": {"AAPL": [bar()]}}])
    original = {
        item.relative_to(path.parents[1]): item.read_bytes()
        for item in path.parents[1].rglob("*") if item.is_file()
    }
    with pytest.raises(ValueError, match=r"immutable|exist"):
        run(tmp_path, [])
    assert original == {
        item.relative_to(path.parents[1]): item.read_bytes()
        for item in path.parents[1].rglob("*") if item.is_file()
    }


def cli_module():
    path = Path(__file__).parents[1] / "scripts" / "download_alpaca_history.py"
    spec = importlib.util.spec_from_file_location("download_alpaca_history", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_missing_credentials_explains_keychain_auth(tmp_path, monkeypatch, capsys):
    module = cli_module()

    async def no_credentials(self):
        return None

    monkeypatch.setattr(module.AlpacaKeychainStorage, "get_credentials", no_credentials)
    monkeypatch.setattr("sys.argv", ["download_alpaca_history.py", "--output-dir", str(tmp_path / "output")])
    with pytest.raises(SystemExit, match="tradecopilot auth alpaca"):
        module.main()
    assert capsys.readouterr().out == ""


def test_cli_defaults_and_output_only(tmp_path, monkeypatch, capsys):
    module = cli_module()
    calls = []

    async def credentials(self):
        return KEY, SECRET

    async def download(config, start, end, output_dir, api_key, secret_key):
        calls.append((config, start, end, output_dir, api_key, secret_key))
        return output_dir / "bars" / "manifest.json"

    monkeypatch.setattr(module.AlpacaKeychainStorage, "get_credentials", credentials)
    monkeypatch.setattr(module, "download_history", download)
    monkeypatch.setattr("sys.argv", ["download_alpaca_history.py", "--output-dir", str(tmp_path / "output")])
    module.main()
    assert calls[0][0].symbols == ("AAPL", "MSFT", "AMZN", "NFLX", "TSLA")
    assert calls[0][1:3] == (date(2026, 4, 20), date(2026, 9, 16))
    assert capsys.readouterr().out == f"{tmp_path / 'output' / 'bars' / 'manifest.json'}\n"
