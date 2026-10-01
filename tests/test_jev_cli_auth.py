from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

import pytest
from keyring.errors import PasswordDeleteError
from rich.console import Console

from tradecopilot import auth, cli
from tradecopilot.config import StrategyConfig


@pytest.fixture(autouse=True)
def isolated_credentials(monkeypatch):
    entries: dict[tuple[str, str], str] = {}

    def delete_password(service: str, username: str) -> None:
        if entries.pop((service, username), None) is None:
            raise PasswordDeleteError("No test credential")

    def deny_prompt(prompt: str) -> str:
        raise AssertionError(f"Unexpected credential prompt: {prompt}")

    async def deny_robinhood_auth():
        raise AssertionError("Jev must not use Robinhood authentication")

    monkeypatch.setenv("TYPESAFE_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(auth.keyring, "get_password", lambda service, username: entries.get((service, username)))
    monkeypatch.setattr(
        auth.keyring, "set_password", lambda service, username, value: entries.__setitem__((service, username), value)
    )
    monkeypatch.setattr(auth.keyring, "delete_password", delete_password)
    monkeypatch.setattr(auth.getpass, "getpass", deny_prompt)
    monkeypatch.setattr(cli, "authenticate_robinhood", deny_robinhood_auth)
    return entries


@pytest.mark.parametrize("command", ["auth", "logout"])
def test_parser_accepts_jev_credential_commands(command) -> None:
    args = cli.build_parser().parse_args([command, "jev"])
    assert args.provider == "jev"


def test_parser_accepts_explicit_jev_check() -> None:
    assert cli.build_parser().parse_args(["jev-check"]).command == "jev-check"


def test_jev_keychain_round_trip_and_clear_are_silent(isolated_credentials, capsys) -> None:
    storage = auth.JevKeychainStorage()

    async def run() -> None:
        await storage.set_api_key("  test-jev-key  ")
        assert await storage.get_api_key() == "test-jev-key"
        assert isolated_credentials == {("tradecopilot.typesafe-jev", "api-key"): "test-jev-key"}
        await storage.clear()
        await storage.clear()
        assert await storage.get_api_key() is None

    asyncio.run(run())
    assert capsys.readouterr().out == ""


def test_jev_keychain_rejects_empty_key(isolated_credentials) -> None:
    with pytest.raises(ValueError, match="Jev API key is required"):
        asyncio.run(auth.JevKeychainStorage().set_api_key("  "))
    assert isolated_credentials == {}


def test_jev_key_environment_takes_precedence_without_keychain_lookup(monkeypatch) -> None:
    def deny_lookup(*args):
        raise AssertionError("Environment key must not query keychain")

    monkeypatch.setenv("TYPESAFE_API_KEY", " env-jev-key ")
    monkeypatch.setattr(auth.keyring, "get_password", deny_lookup)
    assert asyncio.run(auth.load_jev_api_key()) == "env-jev-key"


def test_jev_key_loads_keychain_and_reports_missing_credentials(isolated_credentials) -> None:
    assert asyncio.run(auth.load_jev_api_key()) is None
    isolated_credentials[("tradecopilot.typesafe-jev", "api-key")] = "stored-jev-key"
    assert asyncio.run(auth.load_jev_api_key()) == "stored-jev-key"


def test_jev_auth_and_logout_dispatch_do_not_print_key(monkeypatch, isolated_credentials) -> None:
    output = io.StringIO()
    monkeypatch.setattr(cli, "CONSOLE", Console(file=output, force_terminal=False))
    monkeypatch.setattr(auth.getpass, "getpass", lambda prompt: "private-test-key")
    assert cli._dispatch(cli.build_parser().parse_args(["auth", "jev"])) == 0
    assert isolated_credentials == {("tradecopilot.typesafe-jev", "api-key"): "private-test-key"}
    assert cli._dispatch(cli.build_parser().parse_args(["logout", "jev"])) == 0
    assert isolated_credentials == {}
    assert "private-test-key" not in output.getvalue()
    assert "operating-system keychain" in output.getvalue()


def test_jev_auth_rejects_empty_prompt_without_storage(monkeypatch, isolated_credentials) -> None:
    monkeypatch.setattr(auth.getpass, "getpass", lambda prompt: "  ")
    with pytest.raises(ValueError, match="Jev API key is required"):
        cli._dispatch(cli.build_parser().parse_args(["auth", "jev"]))
    assert isolated_credentials == {}


def test_live_serve_without_jev_does_not_lookup_credentials(monkeypatch) -> None:
    def deny_lookup(*args):
        raise AssertionError("Disabled Jev must not load credentials")

    calls = []
    monkeypatch.setattr(auth.keyring, "get_password", deny_lookup)
    monkeypatch.setattr(cli, "_serve_provider", lambda *args, **kwargs: calls.append(kwargs) or 0)
    assert cli._dispatch(cli.build_parser().parse_args(["serve", "--mode", "live"])) == 0
    assert len(calls) == 1
    assert calls[0].get("jev_advisor") is None


def test_replay_serve_rejects_jev_before_loading_credentials(monkeypatch) -> None:
    def deny_lookup(*args):
        raise AssertionError("Replay must not load Jev credentials")

    monkeypatch.setattr(auth.keyring, "get_password", deny_lookup)
    args = cli.build_parser().parse_args(["serve", "--mode", "replay", "--jev"])
    with pytest.raises(ValueError, match="--jev requires --mode live"):
        cli._dispatch(args)


@pytest.mark.parametrize("arguments", [["serve", "--mode", "live", "--jev"], ["jev-check"]])
def test_explicit_jev_commands_require_credentials(arguments) -> None:
    with pytest.raises(ValueError, match="tradecopilot auth jev"):
        cli._dispatch(cli.build_parser().parse_args(arguments))


def test_live_serve_passes_enabled_advisor_without_database_budget_path(monkeypatch, tmp_path) -> None:
    created = []
    served = []

    class FakeAdvisor:
        def __init__(self, api_key: str) -> None:
            created.append(api_key)

    monkeypatch.setenv("TYPESAFE_API_KEY", "env-jev-key")
    monkeypatch.setattr(cli, "JevAdvisor", FakeAdvisor, raising=False)
    monkeypatch.setattr(cli, "_serve_provider", lambda *args, **kwargs: served.append(kwargs) or 0)
    args = cli.build_parser().parse_args(["serve", "--mode", "live", "--jev", "--db", str(tmp_path / "other.db")])
    assert cli._dispatch(args) == 0
    assert created == ["env-jev-key"]
    assert isinstance(served[0]["jev_advisor"], FakeAdvisor)


def test_serve_provider_forwards_advisor_to_dashboard(monkeypatch) -> None:
    forwarded = []
    advisor = object()
    monkeypatch.setattr(cli, "serve_dashboard", lambda *args, **kwargs: forwarded.append(kwargs))
    assert cli._serve_provider(object(), 8765, True, Path("unused.db"), StrategyConfig(), jev_advisor=advisor) == 0
    assert forwarded[0]["jev_advisor"] is advisor


def test_jev_check_prints_only_public_diagnostic(monkeypatch) -> None:
    output = io.StringIO()
    calls = []

    class FakeAdvisor:
        def __init__(self, api_key: str) -> None:
            assert api_key == "env-jev-key"

        def check(self):
            calls.append("check")
            return {"status": "available", "prediction_validated": False}

    monkeypatch.setenv("TYPESAFE_API_KEY", "env-jev-key")
    monkeypatch.setattr(cli, "JevAdvisor", FakeAdvisor, raising=False)
    monkeypatch.setattr(cli, "CONSOLE", Console(file=output, force_terminal=False))
    assert cli._dispatch(cli.build_parser().parse_args(["jev-check"])) == 0
    assert calls == ["check"]
    assert json.loads(output.getvalue()) == {"status": "available", "prediction_validated": False}
    assert "env-jev-key" not in output.getvalue()


@pytest.mark.parametrize("status", ["unavailable", "budget_exhausted", "cooldown"])
def test_jev_check_returns_nonzero_for_failed_diagnostic(monkeypatch, status) -> None:
    output = io.StringIO()

    class FakeAdvisor:
        def __init__(self, api_key: str) -> None:
            pass

        def check(self):
            return {"status": status}

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(cli, "JevAdvisor", FakeAdvisor, raising=False)
    monkeypatch.setattr(cli, "CONSOLE", Console(file=output, force_terminal=False))
    assert cli._dispatch(cli.build_parser().parse_args(["jev-check"])) == 1
    assert json.loads(output.getvalue())["status"] == status
