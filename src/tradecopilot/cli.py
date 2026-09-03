from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

from rich.console import Console
from rich.table import Table

from tradecopilot.alerts import TransitionAlertDispatcher, WebhookAlertSink
from tradecopilot.auth import AlpacaKeychainStorage, KeychainOAuthStorage, prompt_for_alpaca_credentials
from tradecopilot.backtest import ReplayBacktestRunner, report_summary, write_nightly_report
from tradecopilot.chat import OpenAITradeChatAgent, TradeChatAgent
from tradecopilot.config import StrategyConfig
from tradecopilot.doctor import run_doctor
from tradecopilot.explain import OpenAIExplanationAgent, SafeExplanationService
from tradecopilot.journal import Journal
from tradecopilot.models import CatalystEvidence, DataQuality, ExperimentCandidate, FloatEvidence, MarketFrame, RunMode
from tradecopilot.monitor import Monitor, transition_states
from tradecopilot.providers.base import FrameProvider
from tradecopilot.providers.live import LiveMcpFrameProvider, authenticate_robinhood
from tradecopilot.providers.mock import MockProvider
from tradecopilot.providers.replay import ReplayProvider
from tradecopilot.research import ResearchController
from tradecopilot.security import BLOCKED_WRITE_TOOLS, READ_ONLY_ALLOWLIST
from tradecopilot.webapp import serve_dashboard

DEFAULT_DB = Path(os.getenv("TRADECOPILOT_DB_PATH", ".tradecopilot/journal.sqlite3"))
CONSOLE = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tradecopilot",
        description="Analysis-only low-float momentum copilot; manual execution only",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Validate local safety and dependencies")
    doctor.add_argument("--db", type=Path, default=DEFAULT_DB)
    doctor.add_argument("--symbol", default="AAPL")
    doctor.add_argument("--account-last4")
    doctor.add_argument("--alpaca-feed", choices=("iex", "sip"), default="sip")

    auth = subparsers.add_parser("auth", help="Store provider credentials in the operating-system keychain")
    auth.add_argument("provider", choices=("robinhood", "alpaca"))

    logout = subparsers.add_parser("logout", help="Clear application credentials from the operating-system keychain")
    logout.add_argument("provider", choices=("robinhood", "alpaca"))

    monitor = subparsers.add_parser("monitor", help="Monitor a selected symbol")
    monitor.add_argument("symbol")
    monitor.add_argument("--mode", choices=("live", "mock"), default="live")
    monitor.add_argument("--risk-usd", type=Decimal, default=Decimal("25"))
    monitor.add_argument("--daily-loss-usd", type=Decimal, default=Decimal("75"))
    monitor.add_argument("--float-shares", type=int)
    monitor.add_argument("--float-source")
    monitor.add_argument("--catalyst")
    monitor.add_argument("--catalyst-source")
    monitor.add_argument("--catalyst-time")
    monitor.add_argument("--db", type=Path, default=DEFAULT_DB)
    monitor.add_argument("--account-last4")
    monitor.add_argument("--alpaca-feed", choices=("iex", "sip"), default="sip")
    monitor.add_argument("--scan-title")

    replay = subparsers.add_parser("replay", help="Replay deterministic JSONL without look-ahead")
    replay.add_argument("path", type=Path)
    replay.add_argument("--speed", type=float, default=10.0)
    replay.add_argument("--db", type=Path, default=DEFAULT_DB)
    replay.add_argument("--compact", action="store_true")

    app = subparsers.add_parser("app", help="Launch the local visual analysis desk")
    app.add_argument("--mode", choices=("replay", "mock"), default="replay")
    app.add_argument("--replay", type=Path, default=Path("examples/yxt_replay.jsonl"))
    app.add_argument("--speed", type=float, default=10.0)
    app.add_argument("--port", type=int, default=8765)
    app.add_argument("--no-open", action="store_true")
    app.add_argument("--db", type=Path, default=DEFAULT_DB)

    serve = subparsers.add_parser("serve", help="Launch the replay or live visual analysis desk")
    serve.add_argument("--mode", choices=("replay", "live"), default="replay")
    serve.add_argument("--symbol", default="AAPL")
    serve.add_argument("--replay", type=Path, default=Path("examples/yxt_replay.jsonl"))
    serve.add_argument("--speed", type=float, default=10.0)
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--no-open", action="store_true")
    serve.add_argument("--db", type=Path, default=DEFAULT_DB)
    serve.add_argument("--account-last4")
    serve.add_argument("--alpaca-feed", choices=("iex", "sip"), default="sip")
    serve.add_argument("--scan-title")

    report = subparsers.add_parser("report", help="Summarize a journal date")
    report.add_argument("--date", required=True, type=date.fromisoformat)
    report.add_argument("--db", type=Path, default=DEFAULT_DB)

    research = subparsers.add_parser("research", help="Controlled after-hours research")
    research_sub = research.add_subparsers(dest="research_command", required=True)
    propose = research_sub.add_parser("propose")
    propose.add_argument("--parameter")
    propose.add_argument("--value")
    propose.add_argument("--hypothesis", default="")
    propose.add_argument("--rationale", default="")
    propose.add_argument("--db", type=Path, default=DEFAULT_DB)
    nightly = research_sub.add_parser("nightly", help="Run chronological replay evaluation and write a report")
    nightly.add_argument("--replay-dir", type=Path, default=Path("examples"))
    nightly.add_argument("--output-dir", type=Path, default=Path(".tradecopilot/research"))

    strategy = subparsers.add_parser("strategy", help="Inspect or explicitly promote versions")
    strategy_sub = strategy.add_subparsers(dest="strategy_command", required=True)
    strategy_sub.add_parser("list")
    compare = strategy_sub.add_parser("compare")
    compare.add_argument("base")
    compare.add_argument("candidate")
    compare.add_argument("--db", type=Path, default=DEFAULT_DB)
    promote = strategy_sub.add_parser("promote")
    promote.add_argument("candidate")
    promote.add_argument("--human-confirm", action="store_true")
    promote.add_argument("--db", type=Path, default=DEFAULT_DB)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        exit_code = _dispatch(args)
    except (ValueError, PermissionError, ConnectionError) as exc:
        CONSOLE.print(f"[red]ERROR:[/red] {exc}")
        raise SystemExit(2) from exc
    raise SystemExit(exit_code)


def _dispatch(args: argparse.Namespace) -> int:
    config = StrategyConfig()
    if args.command == "doctor":
        return _doctor(args.db, config, args.symbol, args.account_last4, args.alpaca_feed)
    if args.command == "auth":
        return asyncio.run(_auth(args.provider))
    if args.command == "logout":
        return asyncio.run(_logout(args.provider))
    if args.command == "replay":
        return asyncio.run(_replay(args.path, args.speed, args.db, args.compact, config))
    if args.command == "app":
        return _app(args.mode, args.replay, args.speed, args.port, args.no_open, args.db, config)
    if args.command == "serve":
        if args.mode == "replay":
            return _app("replay", args.replay, args.speed, args.port, args.no_open, args.db, config)
        provider = LiveMcpFrameProvider(
            args.symbol,
            args.account_last4,
            config,
            alpaca_feed=args.alpaca_feed,
            scan_title=args.scan_title,
        )
        return _serve_provider(provider, args.port, args.no_open, args.db, config)
    if args.command == "monitor":
        _validate_manual_evidence(args)
        config = StrategyConfig.model_validate(
            {
                **config.model_dump(),
                "maximum_risk_per_trade_usd": args.risk_usd,
                "maximum_daily_loss_usd": args.daily_loss_usd,
            }
        )
        if args.mode == "live":
            manual_float, manual_catalyst = _manual_evidence(args)
            provider = LiveMcpFrameProvider(
                args.symbol,
                args.account_last4,
                config,
                alpaca_feed=args.alpaca_feed,
                scan_title=args.scan_title,
                manual_float=manual_float,
                manual_catalyst=manual_catalyst,
            )
            return asyncio.run(_live_monitor(provider, args.db, config))
        if args.symbol.upper() != "YXT":
            raise ValueError("mock mode currently provides only the synthetic YXT fixture")
        default_replay = Path("examples/yxt_replay.jsonl")
        return asyncio.run(_mock(default_replay, args.db, config))
    if args.command == "report":
        with Journal(args.db, config.raw_snapshot_retention_rows) as journal:
            CONSOLE.print_json(json.dumps(journal.report(args.date)))
        return 0
    if args.command == "research" and args.research_command == "propose":
        with Journal(args.db, config.raw_snapshot_retention_rows) as journal:
            if not args.parameter or args.value is None:
                CONSOLE.print(
                    "MORE_DATA — pass --parameter, --value, --hypothesis, and --rationale; no candidate was created"
                )
                return 0
            candidate = ResearchController(journal, config).propose(
                args.parameter, args.value, args.hypothesis, args.rationale
            )
            CONSOLE.print_json(candidate.model_dump_json())
        return 0
    if args.command == "research" and args.research_command == "nightly":
        if not args.replay_dir.is_dir():
            raise ValueError(f"replay directory does not exist: {args.replay_dir}")
        paths = tuple(sorted(args.replay_dir.glob("*.jsonl")))
        report = asyncio.run(ReplayBacktestRunner(config).run(paths))
        target = write_nightly_report(report, args.output_dir)
        CONSOLE.print_json(report_summary(report))
        CONSOLE.print(f"Nightly report: {target}")
        return 0
    if args.command == "strategy":
        return _strategy(args, config)
    raise ValueError("unsupported command")


def _doctor(
    path: Path,
    config: StrategyConfig,
    symbol: str,
    account_last4: str | None,
    alpaca_feed: str,
) -> int:
    table = Table(title="tradecopilot doctor (read-only)")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    checks = run_doctor(path, config, symbol=symbol, account_last4=account_last4, alpaca_feed=alpaca_feed)
    for check in checks:
        table.add_row(check.name, "PASS" if check.ok else "DEGRADED", check.detail)
    CONSOLE.print(table)
    return 0 if all(check.ok for check in checks) else 1


async def _auth(provider: str) -> int:
    if provider == "alpaca":
        api_key, secret_key = prompt_for_alpaca_credentials()
        await AlpacaKeychainStorage().set_credentials(api_key, secret_key)
        CONSOLE.print("Alpaca market-data credentials stored in the operating-system keychain.")
        return 0
    tools = await authenticate_robinhood()
    visible = set(tools) & READ_ONLY_ALLOWLIST
    if visible & BLOCKED_WRITE_TOOLS:
        raise PermissionError("Robinhood write tool reached the application-visible surface")
    CONSOLE.print(f"Robinhood OAuth complete; {len(visible)} reviewed read-only tools are application-visible.")
    CONSOLE.print("No Robinhood tool was called during authentication.")
    return 0


async def _logout(provider: str) -> int:
    if provider == "alpaca":
        await AlpacaKeychainStorage().clear()
    else:
        await KeychainOAuthStorage().clear()
    CONSOLE.print(f"Cleared local {provider} credentials from the operating-system keychain.")
    return 0


async def _replay(
    path: Path,
    speed: float,
    database_path: Path,
    compact: bool,
    config: StrategyConfig,
) -> int:
    if not path.exists():
        raise ValueError(f"replay file does not exist: {path}")
    with Journal(database_path, config.raw_snapshot_retention_rows) as journal:
        monitor = Monitor(
            config,
            journal,
            _explanation_service(),
            CONSOLE,
            compact=compact,
            alert_dispatcher=_alert_dispatcher(),
        )
        decisions = await monitor.run(ReplayProvider(path, speed))
    states = transition_states(decisions)
    CONSOLE.print("Transitions: " + " -> ".join(state.value for state in states))
    return 0


async def _mock(path: Path, database_path: Path, config: StrategyConfig) -> int:
    if not path.exists():
        raise ValueError(f"mock fixture does not exist: {path}")
    replay_frames = [frame async for frame in ReplayProvider(path, speed=0).frames()]
    frames = [frame.model_copy(update={"mode": RunMode.MOCK}) for frame in replay_frames]
    with Journal(database_path, config.raw_snapshot_retention_rows) as journal:
        monitor = Monitor(
            config,
            journal,
            _explanation_service(),
            CONSOLE,
            alert_dispatcher=_alert_dispatcher(),
        )
        decisions = await monitor.run(MockProvider(frames))
    states = transition_states(decisions)
    CONSOLE.print("Transitions: " + " -> ".join(state.value for state in states))
    return 0


async def _live_monitor(provider: FrameProvider, database_path: Path, config: StrategyConfig) -> int:
    with Journal(database_path, config.raw_snapshot_retention_rows) as journal:
        monitor = Monitor(
            config,
            journal,
            _explanation_service(),
            CONSOLE,
            alert_dispatcher=_alert_dispatcher(),
        )
        await monitor.run(provider)
    return 0


def _app(
    mode: str,
    replay_path: Path,
    speed: float,
    port: int,
    no_open: bool,
    database_path: Path,
    config: StrategyConfig,
) -> int:
    if not replay_path.exists():
        raise ValueError(f"replay file does not exist: {replay_path}")
    if mode == "replay":
        provider: FrameProvider = ReplayProvider(replay_path, speed)
    else:
        provider = MockProvider(asyncio.run(_load_mock_frames(replay_path)))
    chat_agent = _chat_agent()

    serve_dashboard(
        provider,
        config,
        database_path,
        port=port,
        open_browser=not no_open,
        console=CONSOLE,
        chat_agent=chat_agent,
        explanation_service=_explanation_service(),
        alert_dispatcher=_alert_dispatcher(),
    )
    return 0


def _chat_agent() -> TradeChatAgent | None:
    chat_agent: TradeChatAgent | None = None
    chat_mode = os.getenv(
        "TRADECOPILOT_CHAT_MODE",
        "openai" if os.getenv("OPENAI_API_KEY") else "deterministic",
    )
    if chat_mode == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            CONSOLE.print("[yellow]GPT chat disabled:[/yellow] OPENAI_API_KEY is not configured; using local fallback.")
        else:
            chat_agent = OpenAITradeChatAgent(
                os.getenv("TRADECOPILOT_OPENAI_MODEL", "gpt-5.6"),
                effort=os.getenv("TRADECOPILOT_OPENAI_REASONING_EFFORT", "high"),
                reasoning_mode="pro",
                safety_identifier=os.getenv("TRADECOPILOT_OPENAI_SAFETY_IDENTIFIER"),
            )
    return chat_agent


def _serve_provider(
    provider: FrameProvider,
    port: int,
    no_open: bool,
    database_path: Path,
    config: StrategyConfig,
) -> int:
    serve_dashboard(
        provider,
        config,
        database_path,
        port=port,
        open_browser=not no_open,
        console=CONSOLE,
        chat_agent=_chat_agent(),
        symbol_selector=getattr(provider, "select_symbol", None),
        explanation_service=_explanation_service(),
        alert_dispatcher=_alert_dispatcher(),
    )
    return 0


async def _load_mock_frames(replay_path: Path) -> list[MarketFrame]:
    replay_frames = [frame async for frame in ReplayProvider(replay_path, speed=0).frames()]
    return [frame.model_copy(update={"mode": RunMode.MOCK}) for frame in replay_frames]


def _explanation_service() -> SafeExplanationService:
    mode = os.getenv(
        "TRADECOPILOT_EXPLANATION_MODE",
        "openai" if os.getenv("OPENAI_API_KEY") else "deterministic",
    )
    if mode != "openai":
        return SafeExplanationService()
    if not os.getenv("OPENAI_API_KEY"):
        CONSOLE.print(
            "[yellow]OpenAI explanations disabled:[/yellow] OPENAI_API_KEY is not configured; "
            "using deterministic explanations."
        )
        return SafeExplanationService()
    agent = OpenAIExplanationAgent(
        os.getenv("TRADECOPILOT_OPENAI_MODEL", "gpt-5.6"),
        effort=os.getenv("TRADECOPILOT_OPENAI_REASONING_EFFORT", "high"),
        reasoning_mode="pro",
        safety_identifier=os.getenv("TRADECOPILOT_OPENAI_SAFETY_IDENTIFIER"),
    )
    return SafeExplanationService(agent=agent)


def _alert_dispatcher() -> TransitionAlertDispatcher | None:
    url = os.getenv("TRADECOPILOT_ALERT_WEBHOOK_URL")
    if not url:
        return None
    return TransitionAlertDispatcher(WebhookAlertSink(url))


def _strategy(args: argparse.Namespace, config: StrategyConfig) -> int:
    if args.strategy_command == "list":
        CONSOLE.print_json(config.strategy_version().model_dump_json())
        return 0
    if args.strategy_command == "compare":
        with Journal(args.db, config.raw_snapshot_retention_rows) as journal:
            raw = journal.get_experiment(args.candidate)
        if raw is None:
            raise ValueError(f"unknown candidate: {args.candidate}")
        candidate = ExperimentCandidate.model_validate(raw)
        if candidate.base_version != args.base:
            raise ValueError(f"candidate base is {candidate.base_version}, not requested base {args.base}")
        result = {
            "base": candidate.base_version,
            "candidate": candidate.candidate_version,
            "changed_parameter": candidate.changed_parameter,
            "old_value": candidate.old_value,
            "new_value": candidate.new_value,
            "baseline_metrics": candidate.baseline_metrics,
            "candidate_metrics": candidate.candidate_metrics,
            "regressions": candidate.regressions,
            "status": candidate.research_result,
        }
        CONSOLE.print_json(json.dumps(result))
        return 0
    if args.strategy_command == "promote":
        if not args.human_confirm:
            raise PermissionError("--human-confirm is required")
        with Journal(args.db, config.raw_snapshot_retention_rows) as journal:
            raw = journal.get_experiment(args.candidate)
            if raw is None:
                raise ValueError(f"unknown candidate: {args.candidate}")
            candidate = ExperimentCandidate.model_validate(raw)
            approved = ResearchController(journal, config).promote(candidate, human_confirm=args.human_confirm)
        CONSOLE.print_json(approved.model_dump_json())
        CONSOLE.print("Human approval recorded. Production configuration was not rewritten automatically.")
        return 0
    raise ValueError("unsupported strategy command")


def _validate_manual_evidence(args: argparse.Namespace) -> None:
    if args.risk_usd <= 0 or args.daily_loss_usd <= 0:
        raise ValueError("--risk-usd and --daily-loss-usd must be positive")
    if args.float_shares is not None and not args.float_source:
        raise ValueError("--float-source is required with --float-shares")
    if args.float_shares is not None and args.float_shares <= 0:
        raise ValueError("--float-shares must be positive")
    catalyst_fields = (args.catalyst, args.catalyst_source, args.catalyst_time)
    if any(catalyst_fields) and not all(catalyst_fields):
        raise ValueError("--catalyst, --catalyst-source, and --catalyst-time are all required")
    if args.catalyst_time:
        parsed = datetime.fromisoformat(args.catalyst_time.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("--catalyst-time must include a timezone offset")


def _manual_evidence(args: argparse.Namespace) -> tuple[FloatEvidence | None, CatalystEvidence | None]:
    now = datetime.now(UTC)
    normalized = args.symbol.strip().upper()
    float_evidence = None
    if args.float_shares is not None:
        float_evidence = FloatEvidence(
            provider_timestamp=now,
            receipt_timestamp=now,
            age_seconds=0,
            source=f"manual_cli:{normalized}",
            quality=DataQuality.LIMITED,
            shares=args.float_shares,
            verified=True,
            reference=args.float_source,
        )
    catalyst_evidence = None
    if args.catalyst is not None:
        catalyst_time = datetime.fromisoformat(args.catalyst_time.replace("Z", "+00:00"))
        catalyst_evidence = CatalystEvidence(
            provider_timestamp=catalyst_time,
            receipt_timestamp=now,
            age_seconds=(now - catalyst_time).total_seconds(),
            source=f"manual_cli:{normalized}",
            quality=DataQuality.LIMITED,
            description=args.catalyst,
            verified=True,
            reference=args.catalyst_source,
        )
    return float_evidence, catalyst_evidence
