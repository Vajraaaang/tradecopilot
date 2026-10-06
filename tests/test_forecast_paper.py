"""Paper reads use fabricated HTTP responses and never access credentials or the network."""

import asyncio
import hashlib
import json
import traceback
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import httpx
import pytest

START = datetime(2026, 4, 1, 13, 30, tzinfo=UTC)
END = START + timedelta(minutes=4)
KEY = "private-paper-key"
SECRET = "private-paper-secret"
ACCOUNT = {
    "id": "private-account-id", "account_number": "private-account-number", "cash": "123456.78",
    "equity": "234567.89", "status": "ACTIVE", "currency": "USD", "account_blocked": False,
    "trading_blocked": True, "trade_suspended_by_user": False, "transfers_blocked": True,
}
CLOCK = {
    "timestamp": "2026-04-01T09:32:00-04:00", "is_open": True,
    "next_open": "2026-04-02T09:30:00-04:00", "next_close": "2026-04-01T16:00:00-04:00",
}


def bar(at="2026-04-01T13:30:00Z", **changes):
    return {"t": at, "o": "100.0010", "h": "101.1234", "l": "99.9876", "c": "100.7654", "v": 1234, **changes}


def run(pages, *, symbols=("MSFT", "AAPL"), start=START, end=END, feed="iex", credentials=(KEY, SECRET),
        account=ACCOUNT, clock=CLOCK, requests=None):
    from tradecopilot.forecast.paper import read_paper_snapshot

    requests = [] if requests is None else requests
    bar_requests = []

    def serve(request):
        requests.append(request)
        if request.url.path == "/v2/account":
            value = account
        elif request.url.path == "/v2/clock":
            value = clock
        else:
            bar_requests.append(request)
            value = pages[len(bar_requests) - 1]
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, httpx.Response) else httpx.Response(200, json=value)

    snapshot = asyncio.run(read_paper_snapshot(
        symbols, start, end, feed=feed, credentials=credentials, transport=httpx.MockTransport(serve),
    ))
    return snapshot, requests


@pytest.mark.parametrize("feed", ["iex", "sip"])
def test_snapshot_preserves_feed_values_safe_account_clock_and_pagination(feed):
    pages = [
        {"bars": {"MSFT": [bar()], "AAPL": [bar("2026-04-01T13:31:00Z")]}, "next_page_token": "second"},
        {"bars": {"AAPL": [bar()]}, "next_page_token": None},
    ]
    snapshot, requests = run(pages, feed=feed)
    assert [(item.symbol, item.start_time) for item in snapshot.bars] == [
        ("AAPL", START), ("AAPL", START + timedelta(minutes=1)), ("MSFT", START),
    ]
    assert snapshot.bars[0].opening == Decimal("100.0010") and str(snapshot.bars[0].opening) == "100.0010"
    assert all(item.end_time == item.available_at == item.start_time + timedelta(minutes=1) for item in snapshot.bars)
    assert all(item.source == f"alpaca_{feed}_1min_bar" for item in snapshot.bars)
    assert snapshot.account == {key: value for key, value in ACCOUNT.items() if key not in {
        "id", "account_number", "cash", "equity",
    }}
    assert snapshot.account["trading_blocked"] is True
    assert snapshot.clock == {
        "timestamp": "2026-04-01T13:32:00+00:00", "is_open": True,
        "next_open": "2026-04-02T13:30:00+00:00", "next_close": "2026-04-01T20:00:00+00:00",
    }
    metadata = snapshot.metadata
    assert metadata["feed"] == feed and metadata["adjustment_policy"] == "raw"
    assert metadata["requested_symbols"] == ["AAPL", "MSFT"]
    assert metadata["requested_start_utc"] == START.isoformat() and metadata["requested_end_utc"] == END.isoformat()
    assert metadata["paper_endpoint"] == "https://paper-api.alpaca.markets"
    assert metadata["source_endpoint"] == "https://data.alpaca.markets/v2/stocks/bars"
    assert metadata["bar_count"] == metadata["downloaded_bar_count"] == 3 and metadata["exclusions"] == {}
    assert "historical assumption" in metadata["availability_assumption"]
    assert datetime.fromisoformat(metadata["receipt_at"]).utcoffset() == timedelta(0)
    assert len(snapshot.raw_pages) == len(metadata["raw_pages"]) == 2
    for index, (raw, record) in enumerate(zip(snapshot.raw_pages, metadata["raw_pages"], strict=True)):
        assert json.loads(raw) == pages[index]
        assert hashlib.sha256(raw).hexdigest() == record["sha256"] and len(raw) == record["bytes"]
        assert record["bar_count"] == (2 if index == 0 else 1)
    safe = json.dumps({"account": snapshot.account, "clock": snapshot.clock, "metadata": metadata}) + repr(snapshot)
    assert all(secret not in safe for secret in (KEY, SECRET, "private-account-id", "123456.78", "234567.89"))
    assert "raw_pages=" not in repr(snapshot)
    with pytest.raises(FrozenInstanceError):
        snapshot.bars = ()
    assert [(request.method, request.url.host, request.url.path) for request in requests] == [
        ("GET", "paper-api.alpaca.markets", "/v2/account"),
        ("GET", "paper-api.alpaca.markets", "/v2/clock"),
        ("GET", "data.alpaca.markets", "/v2/stocks/bars"),
        ("GET", "data.alpaca.markets", "/v2/stocks/bars"),
    ]
    for request in requests:
        assert request.url.scheme == "https"
        assert request.headers["APCA-API-KEY-ID"] == KEY and request.headers["APCA-API-SECRET-KEY"] == SECRET
        assert KEY not in str(request.url) and SECRET not in str(request.url)
    for request in requests[2:]:
        assert request.url.params["symbols"] == "AAPL,MSFT"
        assert request.url.params["feed"] == feed and request.url.params["adjustment"] == "raw"
        assert request.url.params["timeframe"] == "1Min" and request.url.params["limit"] == "1000"
        assert datetime.fromisoformat(request.url.params["start"]) == START
        assert datetime.fromisoformat(request.url.params["end"]) < END
    assert "page_token" not in requests[2].url.params and requests[3].url.params["page_token"] == "second"


@pytest.mark.parametrize("changes", [
    {"symbols": ()}, {"symbols": ("AAPL",) * 2}, {"symbols": ("A", "B", "C", "D", "E", "F")},
    {"symbols": ("aapl",)}, {"symbols": ("AAPL ",)}, {"symbols": ("A/APL",)}, {"symbols": "MSFT"},
    {"symbols": (42,)}, {"start": START.replace(tzinfo=None)}, {"end": END.replace(tzinfo=None)},
    {"end": START}, {"end": START - timedelta(seconds=1)}, {"end": START + timedelta(days=31, seconds=1)},
    {"feed": "best"}, {"credentials": ("", SECRET)}, {"credentials": (KEY, "bad\nsecret")},
    {"credentials": (KEY, "nonascii-\N{SNOWMAN}")}, {"credentials": (KEY,)},
])
def test_invalid_inputs_fail_before_http(changes):
    requests = []
    with pytest.raises(ValueError):
        run([], requests=requests, **changes)
    assert requests == []


@pytest.mark.parametrize("changes", [
    {"status": "CLOSED"}, {"currency": "EUR"}, {"account_blocked": "false"}, {"trading_blocked": 0},
    {"trade_suspended_by_user": None}, {"transfers_blocked": "true"},
])
def test_account_requires_active_usd_and_actual_boolean_flags(changes):
    requests = []
    with pytest.raises(ValueError):
        run([], account={**ACCOUNT, **changes}, requests=requests)
    assert len(requests) == 1


@pytest.mark.parametrize("field", ["status", "currency", "account_blocked", "trading_blocked"])
def test_missing_required_account_fields_fail_closed(field):
    with pytest.raises(ValueError):
        run([], account={key: value for key, value in ACCOUNT.items() if key != field})


@pytest.mark.parametrize("changes", [
    {"is_open": "true"}, {"timestamp": "2026-04-01T13:32:00"}, {"next_open": "bad-date"},
    {"next_close": None}, {"timestamp": 123},
])
def test_clock_requires_timezone_aware_timestamps_and_boolean_open(changes):
    with pytest.raises(ValueError):
        run([], clock={**CLOCK, **changes})


def test_only_completed_regular_xnys_minutes_survive():
    rows = [bar(at) for at in [
        "2026-03-31T23:59:00Z", "2026-04-01T13:29:00Z", "2026-04-01T13:30:00Z",
        "2026-04-01T19:59:00Z", "2026-04-01T20:00:00Z", "2026-04-03T13:30:00Z",
        "2026-04-04T13:30:00Z", "2026-04-05T00:00:00Z",
    ]]
    snapshot, _ = run([{"bars": {"AAPL": rows}}], start=START.replace(hour=0, minute=0),
                      end=START.replace(day=5, hour=0, minute=0))
    assert [item.start_time for item in snapshot.bars] == [START, START.replace(hour=19, minute=59)]
    assert snapshot.metadata["exclusions"] == {"outside_requested_range": 2, "outside_regular_session": 4}
    assert snapshot.metadata["downloaded_bar_count"] == 8 and snapshot.metadata["bar_count"] == 2


def test_partially_requested_minutes_are_excluded():
    snapshot, _ = run([{"bars": {"AAPL": [bar(), bar("2026-04-01T13:31:00Z")]}}],
                      start=START + timedelta(seconds=30), end=START + timedelta(minutes=1, seconds=30))
    assert snapshot.bars == () and snapshot.metadata["exclusions"] == {"outside_requested_range": 2}


def test_incomplete_minutes_use_actual_receipt_not_provider_clock(monkeypatch):
    from tradecopilot.forecast import paper

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 4, 1, 13, 32, 30, tzinfo=UTC)

    monkeypatch.setattr(paper, "datetime", FrozenDatetime)
    rows = [bar(f"2026-04-01T13:{minute}:00Z") for minute in range(30, 35)]
    snapshot, _ = run([{"bars": {"AAPL": rows}}], clock={**CLOCK, "timestamp": "2099-01-01T00:00:00Z"})
    assert len(snapshot.bars) == 2
    assert snapshot.metadata["exclusions"] == {"incomplete_at_receipt": 2, "outside_requested_range": 1}
    assert all(item.end_time <= datetime.fromisoformat(snapshot.metadata["receipt_at"]) for item in snapshot.bars)


def test_early_close_has_no_extra_regular_session_minute():
    snapshot, _ = run([{"bars": {"AAPL": [bar("2025-11-28T17:59:00Z"), bar("2025-11-28T18:00:00Z")]}}],
                      start=datetime(2025, 11, 28, tzinfo=UTC), end=datetime(2025, 11, 29, tzinfo=UTC))
    assert len(snapshot.bars) == 1 and snapshot.metadata["exclusions"] == {"outside_regular_session": 1}


@pytest.mark.parametrize("rows", [[bar(), bar()], [bar("2026-04-01T13:29:00Z")] * 2])
def test_duplicate_minutes_are_rejected_even_if_identical_or_excluded(rows):
    with pytest.raises(ValueError):
        run([{"bars": {"AAPL": rows}}])


def test_duplicate_minutes_across_pages_are_rejected():
    with pytest.raises(ValueError):
        run([{"bars": {"AAPL": [bar()]}, "next_page_token": "second"}, {"bars": {"AAPL": [bar()]}}])


@pytest.mark.parametrize("row", [
    [], {}, bar(t=123), bar(t="bad-date"), bar(t="2026-04-01T13:30:00"),
    bar(t="2026-04-01T13:30:01Z"), bar(t="2026-04-01T13:30:00.000001Z"),
    bar(o=True), bar(o=None), bar(o="NaN"), bar(o="Infinity"), bar(o=0), bar(v=-1),
    bar(h="99"), bar(l="102"), bar(c={}),
])
def test_invalid_candle_fields_are_rejected_without_detail(row):
    with pytest.raises(ValueError, match="Alpaca paper snapshot could not be read"):
        run([{"bars": {"AAPL": [row]}}])


@pytest.mark.parametrize("page", [
    [], {}, {"bars": []}, {"bars": {"GME": [bar()]}}, {"bars": {"AAPL": {}}},
    {"bars": {}, "next_page_token": ""}, {"bars": {}, "next_page_token": 123},
    {"bars": {}, "next_page_token": "bad\npage"}, {"bars": {}, "next_page_token": "a" * 4097},
    httpx.Response(200, content=b"not JSON"), httpx.Response(200, content=b'{"bars":{},"bad":NaN}'),
])
def test_invalid_pages_fail_closed(page):
    with pytest.raises(ValueError):
        run([page])


def test_pagination_cycle_fails_without_repeating_request():
    requests = []
    with pytest.raises(ValueError):
        run([{"bars": {}, "next_page_token": "repeat"}] * 2, requests=requests)
    assert len(requests) == 4


@pytest.mark.parametrize("bound", ["MAX_PAGES", "MAX_BAR_ROWS", "MAX_RESPONSE_BYTES"])
def test_reads_enforce_resource_bounds(monkeypatch, bound):
    from tradecopilot.forecast import paper

    requests = []
    if bound == "MAX_PAGES":
        monkeypatch.setattr(paper, bound, 2)
        pages = [{"bars": {}, "next_page_token": token} for token in ("first", "second")]
    elif bound == "MAX_BAR_ROWS":
        monkeypatch.setattr(paper, bound, 1)
        pages = [{"bars": {"AAPL": [bar(), bar("2026-04-01T13:31:00Z")]}}]
    else:
        monkeypatch.setattr(paper, bound, 512)
        pages = [httpx.Response(200, content=b" " * 513)]
    with pytest.raises(ValueError):
        run(pages, requests=requests)
    assert len(requests) == (4 if bound == "MAX_PAGES" else 3)


@pytest.mark.parametrize("endpoint", ["account", "clock", "bars"])
@pytest.mark.parametrize("failure", [
    httpx.Response(401, text=f"denied {KEY} {SECRET}"), httpx.Response(403, text=SECRET),
    httpx.Response(429, text=SECRET), httpx.Response(500, text=SECRET),
    httpx.Response(302, headers={"Location": "https://api.alpaca.markets/v2/orders"}),
    httpx.ConnectError(f"failed {KEY} {SECRET}"), httpx.ReadTimeout(f"timeout {KEY} {SECRET}"),
])
def test_failures_have_no_retries_fallback_redirects_or_secret_details(endpoint, failure, capsys):
    requests = []
    overrides = {endpoint: failure} if endpoint != "bars" else {}
    with pytest.raises(ValueError) as captured:
        run([failure], requests=requests, **overrides)
    details = "".join(traceback.format_exception(captured.value))
    assert str(captured.value) == "Alpaca paper snapshot could not be read"
    assert KEY not in details and SECRET not in details
    assert len(requests) == {"account": 1, "clock": 2, "bars": 3}[endpoint]
    assert all(request.url.host in {"paper-api.alpaca.markets", "data.alpaca.markets"} for request in requests)
    assert capsys.readouterr().out == ""


def test_explicit_credentials_ignore_keychain_and_environment(monkeypatch):
    from tradecopilot.forecast import paper

    async def forbidden(self):
        pytest.fail("explicit credentials must not read Keychain")

    monkeypatch.setattr(paper.AlpacaKeychainStorage, "get_credentials", forbidden)
    monkeypatch.setenv("APCA_API_BASE_URL", "https://api.alpaca.markets")
    monkeypatch.setenv("APCA_API_KEY_ID", "wrong-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "wrong-secret")
    snapshot, requests = run([{"bars": {}}])
    assert snapshot.bars == ()
    assert requests[0].url.host == "paper-api.alpaca.markets"
    assert all(request.extensions["timeout"]["read"] == 20 for request in requests)


@pytest.mark.parametrize("stored", [None, ValueError(f"bad credentials {KEY} {SECRET}"), (KEY, SECRET)])
def test_default_credentials_use_existing_keychain_and_redact_errors(monkeypatch, stored):
    from tradecopilot.forecast import paper

    calls = []

    async def credentials(self):
        calls.append(True)
        if isinstance(stored, Exception):
            raise stored
        return stored

    monkeypatch.setattr(paper.AlpacaKeychainStorage, "get_credentials", credentials)
    monkeypatch.setenv("APCA_API_KEY_ID", KEY)
    monkeypatch.setenv("APCA_API_SECRET_KEY", SECRET)
    if isinstance(stored, tuple):
        snapshot, _ = run([{"bars": {}}], credentials=None)
        assert snapshot.account["status"] == "ACTIVE"
    else:
        requests = []
        with pytest.raises(ValueError) as captured:
            run([], credentials=None, requests=requests)
        assert KEY not in "".join(traceback.format_exception(captured.value))
        assert SECRET not in "".join(traceback.format_exception(captured.value)) and requests == []
    assert calls == [True]


def test_bar_ids_are_stable_across_response_order_and_timezone_spelling():
    first, _ = run([{"bars": {"MSFT": [bar()], "AAPL": [bar("2026-04-01T13:31:00Z"), bar()]}}])
    second, _ = run([{"bars": {"AAPL": [bar("2026-04-01T09:30:00-04:00"), bar("2026-04-01T13:31:00Z")],
                               "MSFT": [bar()]}}])
    assert [item.bar_id for item in first.bars] == [item.bar_id for item in second.bars]
