from __future__ import annotations

import asyncio
import io
import json
from types import SimpleNamespace

import pytest
from keyring.errors import PasswordDeleteError
from rich.console import Console

from tradecopilot import auth, cli, doctor
from tradecopilot.config import StrategyConfig


@pytest.fixture(autouse=True)
def isolated_credentials(monkeypatch):
    entries: dict[tuple[str, str], str] = {}

    def delete_password(service: str, username: str) -> None:
        if entries.pop((service, username), None) is None:
            raise PasswordDeleteError("No test credential")

    def deny_call(*args, **kwargs):
        raise AssertionError("Unexpected provider or credential access")

    async def deny_provider_checks(*args, **kwargs):
        raise AssertionError("Finnhub must not call Alpaca, Robinhood, or Shibui")

    for name in ("FINNHUB_API_KEY", "TYPESAFE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(name, "")
    monkeypatch.setattr(auth.keyring, "get_password", lambda service, username: entries.get((service, username)))
    monkeypatch.setattr(
        auth.keyring, "set_password", lambda service, username, value: entries.__setitem__((service, username), value)
    )
    monkeypatch.setattr(auth.keyring, "delete_password", delete_password)
    monkeypatch.setattr(auth.getpass, "getpass", deny_call)
    monkeypatch.setattr(cli, "LiveMcpFrameProvider", deny_call)
    monkeypatch.setattr(cli, "authenticate_robinhood", deny_call)
    monkeypatch.setattr(doctor, "_provider_checks", deny_provider_checks)
    monkeypatch.setattr(doctor, "_local_checks", lambda *args: ())
    return entries


@pytest.mark.parametrize("arguments", [["serve"], ["monitor", "TEST"], ["doctor"]])
def test_data_provider_defaults_to_alpaca_and_accepts_finnhub(arguments) -> None:
    parser = cli.build_parser()
    assert parser.parse_args(arguments).data_provider == "alpaca"
    assert parser.parse_args([*arguments, "--data-provider", "finnhub"]).data_provider == "finnhub"


def test_finnhub_keychain_round_trip_and_empty_rejection(isolated_credentials, capsys) -> None:
    storage = auth.FinnhubKeychainStorage()

    async def run() -> None:
        with pytest.raises(ValueError, match="Finnhub API key is required"):
            await storage.set_api_key("  ")
        await storage.set_api_key(" test-finnhub-key ")
        assert isolated_credentials == {("tradecopilot.finnhub", "api-key"): "test-finnhub-key"}
        assert await storage.get_api_key() == "test-finnhub-key"
        await storage.clear()
        await storage.clear()
        assert await storage.get_api_key() is None

    asyncio.run(run())
    assert capsys.readouterr().out == ""


def test_finnhub_environment_key_precedes_keychain(monkeypatch) -> None:
    def deny_lookup(*args):
        raise AssertionError("Environment key must not query keychain")

    monkeypatch.setenv("FINNHUB_API_KEY", " environment-test-key ")
    monkeypatch.setattr(auth.keyring, "get_password", deny_lookup)
    assert asyncio.run(auth.load_finnhub_api_key()) == "environment-test-key"


def test_finnhub_key_loads_keychain_or_returns_none(isolated_credentials) -> None:
    assert asyncio.run(auth.load_finnhub_api_key()) is None
    isolated_credentials[("tradecopilot.finnhub", "api-key")] = "stored-test-key"
    assert asyncio.run(auth.load_finnhub_api_key()) == "stored-test-key"


def test_finnhub_auth_logout_are_hidden_and_do_not_print_key(monkeypatch, isolated_credentials) -> None:
    output = io.StringIO()
    monkeypatch.setattr(cli, "CONSOLE", Console(file=output, force_terminal=False))
    monkeypatch.setattr(auth.getpass, "getpass", lambda prompt: "private-finnhub-test-key")
    assert cli._dispatch(cli.build_parser().parse_args(["auth", "finnhub"])) == 0
    assert isolated_credentials == {("tradecopilot.finnhub", "api-key"): "private-finnhub-test-key"}
    assert cli._dispatch(cli.build_parser().parse_args(["logout", "finnhub"])) == 0
    assert isolated_credentials == {}
    assert "private-finnhub-test-key" not in output.getvalue()


def test_finnhub_auth_rejects_empty_prompt(monkeypatch, isolated_credentials) -> None:
    monkeypatch.setattr(auth.getpass, "getpass", lambda prompt: " ")
    with pytest.raises(ValueError, match="Finnhub API key is required"):
        cli._dispatch(cli.build_parser().parse_args(["auth", "finnhub"]))
    assert isolated_credentials == {}


@pytest.mark.parametrize("arguments", [["serve", "--mode", "replay"], ["monitor", "YXT", "--mode", "mock"]])
def test_finnhub_rejects_simulation_before_key_lookup(monkeypatch, arguments) -> None:
    def deny_lookup(*args):
        raise AssertionError("Simulation must not load live credentials")

    monkeypatch.setattr(auth.keyring, "get_password", deny_lookup)
    args = cli.build_parser().parse_args([*arguments, "--data-provider", "finnhub"])
    with pytest.raises(ValueError, match="--data-provider finnhub requires --mode live"):
        cli._dispatch(args)


@pytest.mark.parametrize(
    "arguments",
    [["serve", "--mode", "live", "--data-provider", "finnhub"],
     ["monitor", "TEST", "--data-provider", "finnhub"], ["finnhub-quote", "TEST"]],
)
def test_explicit_finnhub_commands_require_credentials(arguments) -> None:
    with pytest.raises(ValueError, match="tradecopilot auth finnhub"):
        cli._dispatch(cli.build_parser().parse_args(arguments))


@pytest.mark.parametrize("command", ["serve", "monitor"])
def test_selected_finnhub_provider_avoids_all_mcp_dependencies(monkeypatch, command) -> None:
    created = []
    launched = []

    class FakeProvider:
        def __init__(self, symbol, api_key):
            created.append((symbol, api_key))

    async def monitor(provider, *args):
        launched.append(provider)
        return 0

    monkeypatch.setenv("FINNHUB_API_KEY", "test-finnhub-key")
    monkeypatch.setattr(cli, "FinnhubFrameProvider", FakeProvider, raising=False)
    monkeypatch.setattr(cli, "_serve_provider", lambda provider, *args, **kwargs: launched.append(provider) or 0)
    monkeypatch.setattr(cli, "_live_monitor", monitor)
    arguments = ["serve", "--symbol", "TEST", "--mode", "live"] if command == "serve" else ["monitor", "TEST"]
    assert cli._dispatch(cli.build_parser().parse_args([*arguments, "--data-provider", "finnhub"])) == 0
    assert created == [("TEST", "test-finnhub-key")]
    assert isinstance(launched[0], FakeProvider)


def test_default_alpaca_does_not_load_finnhub_credentials(monkeypatch) -> None:
    def deny_lookup(*args):
        raise AssertionError("Unselected Finnhub must not load credentials")

    monkeypatch.setattr(auth.keyring, "get_password", deny_lookup)
    monkeypatch.setattr(cli, "LiveMcpFrameProvider", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "_serve_provider", lambda *args, **kwargs: 0)
    assert cli._dispatch(cli.build_parser().parse_args(["serve", "--mode", "live"])) == 0


def test_finnhub_quote_prints_one_public_model_dump(monkeypatch) -> None:
    output = io.StringIO()
    calls = []

    class FakeClient:
        def __init__(self, api_key):
            assert api_key == "private-test-key"

        def quote(self, symbol):
            calls.append(symbol)
            return SimpleNamespace(model_dump=lambda **kwargs: {"symbol": symbol, "last": "123.45"})

    monkeypatch.setenv("FINNHUB_API_KEY", "private-test-key")
    monkeypatch.setattr(cli, "FinnhubClient", FakeClient, raising=False)
    monkeypatch.setattr(cli, "CONSOLE", Console(file=output, force_terminal=False))
    assert cli._dispatch(cli.build_parser().parse_args(["finnhub-quote", "TEST"])) == 0
    assert calls == ["TEST"]
    assert json.loads(output.getvalue()) == {"symbol": "TEST", "last": "123.45"}
    assert "private-test-key" not in output.getvalue()


def test_finnhub_quote_error_does_not_echo_provider_details(monkeypatch) -> None:
    output = io.StringIO()
    calls = []

    class FakeClient:
        def __init__(self, api_key):
            pass

        def quote(self, symbol):
            calls.append(symbol)
            raise ConnectionError("SECRET_SENTINEL token=private-test-key")

    monkeypatch.setenv("FINNHUB_API_KEY", "private-test-key")
    monkeypatch.setattr(cli, "FinnhubClient", FakeClient, raising=False)
    monkeypatch.setattr(cli, "CONSOLE", Console(file=output, force_terminal=False))
    with pytest.raises(SystemExit) as result:
        cli.main(["finnhub-quote", "TEST"])
    assert result.value.code == 2
    assert calls == ["TEST"]
    assert "SECRET_SENTINEL" not in output.getvalue()
    assert "private-test-key" not in output.getvalue()


def test_finnhub_doctor_reads_one_quote_and_marks_strategy_degraded(monkeypatch, tmp_path) -> None:
    calls = []

    class FakeClient:
        def __init__(self, api_key):
            assert api_key == "test-finnhub-key"

        def quote(self, symbol):
            calls.append(symbol)
            return SimpleNamespace(symbol=symbol)

    monkeypatch.setenv("FINNHUB_API_KEY", "test-finnhub-key")
    monkeypatch.setattr(doctor, "FinnhubClient", FakeClient, raising=False)
    checks = doctor.run_doctor(tmp_path / "unused.sqlite3", StrategyConfig(), symbol="test", data_provider="finnhub")
    assert calls == ["TEST"]
    assert any(check.ok and "Finnhub" in check.name for check in checks)
    coverage = next(check for check in checks if "strategy coverage" in check.name.lower())
    assert not coverage.ok
    assert all(word in coverage.detail.lower() for word in ("minute", "bid/ask", "account"))


def test_finnhub_doctor_missing_credentials_is_degraded_without_request(tmp_path) -> None:
    checks = doctor.run_doctor(tmp_path / "unused.sqlite3", StrategyConfig(), data_provider="finnhub")
    assert any(not check.ok and "tradecopilot auth finnhub" in check.detail for check in checks)


def test_finnhub_doctor_provider_errors_are_sanitized(monkeypatch, tmp_path) -> None:
    class FakeClient:
        def __init__(self, api_key):
            pass

        def quote(self, symbol):
            raise RuntimeError("SECRET_SENTINEL token=private-test-key")

    monkeypatch.setenv("FINNHUB_API_KEY", "private-test-key")
    monkeypatch.setattr(doctor, "FinnhubClient", FakeClient, raising=False)
    checks = doctor.run_doctor(tmp_path / "unused.sqlite3", StrategyConfig(), data_provider="finnhub")
    assert any(not check.ok and "Finnhub" in check.name for check in checks)
    assert not any("SECRET_SENTINEL" in check.detail or "private-test-key" in check.detail for check in checks)


def test_cli_doctor_forwards_selected_provider(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "run_doctor", lambda *args, **kwargs: calls.append(kwargs) or ())
    monkeypatch.setattr(cli, "CONSOLE", Console(file=io.StringIO(), force_terminal=False))
    assert cli._dispatch(cli.build_parser().parse_args(["doctor", "--data-provider", "finnhub"])) == 0
    assert calls[0]["data_provider"] == "finnhub"
