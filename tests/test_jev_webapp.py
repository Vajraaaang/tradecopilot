from __future__ import annotations

import json
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tradecopilot.config import StrategyConfig
from tradecopilot.explain import DeterministicExplainer
from tradecopilot.jev import JevAdvisor
from tradecopilot.models import DecisionState, RunMode
from tradecopilot.strategy import DecisionEngine
from tradecopilot.webapp import STATIC_ROOT, DashboardState, create_dashboard_server


class RecordingAdvisor:
    def __init__(self) -> None:
        self.contexts: list[dict[str, Any]] = []
        self.during_request: Callable[[], None] | None = None
        self.result_update: dict[str, Any] = {}

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "model": "jev-test",
            "requests_used": len(self.contexts),
            "request_limit": 100,
            "estimated_cost_usd": 0.001 * len(self.contexts),
        }

    def advise(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        self.contexts.append(json.loads(json.dumps(snapshot)))
        if self.during_request is not None:
            self.during_request()
        return {
            **self.metadata(),
            "status": "available",
            "action": "BUY",
            "probabilities": {"BUY": 0.8, "HOLD": 0.1, "SELL": 0.05, "WAIT": 0.05},
            "confidence": 0.8,
            "message": "Decision confidence, not probability of profit.",
            "symbol": snapshot["meta"]["symbol"],
            "source_time": snapshot["meta"]["quote_time"],
            "expires_at": (datetime.now(UTC) + timedelta(seconds=15)).isoformat(),
            "input_tokens": 20,
            "cached": False,
            **self.result_update,
        }


def _dashboard(
    yxt_frames, advisor=None, *, mode=RunMode.LIVE, decision_state=None,
    quote_age=0, quote_time_age=None, missing=(),
) -> DashboardState:
    config = StrategyConfig()
    state = DashboardState(config, jev_advisor=advisor)
    frame = yxt_frames[0].model_copy(update={"mode": mode})
    decision = DecisionEngine(config).evaluate(frame)
    if decision_state is not None:
        decision = decision.model_copy(update={"state": decision_state})
    if missing:
        decision = decision.model_copy(update={"missing_data": missing})
    now = datetime.now(UTC)
    assert frame.quote is not None
    quote = frame.quote.model_copy(
        update={
            "provider_timestamp": now - timedelta(seconds=quote_age if quote_time_age is None else quote_time_age),
            "receipt_timestamp": now,
            "age_seconds": quote_age,
        }
    )
    frame = frame.model_copy(update={"event_time": now, "quote": quote})
    explanation = DeterministicExplainer().explain(decision, ()).model_copy(update={"missing_data": missing})
    state.update(decision, frame, explanation)
    return state


@contextmanager
def _server(state: DashboardState) -> Iterator[str]:
    server = create_dashboard_server(state)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _post(base: str, body: object, headers: dict[str, str] | None = None) -> tuple[int, Any]:
    request = urllib.request.Request(
        f"{base}/api/jev",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Origin": base, **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def test_jev_endpoint_disabled_without_advisor(yxt_frames) -> None:
    state = _dashboard(yxt_frames)
    with _server(state) as base:
        status, payload = _post(base, {"symbol": "YXT"})
    assert status == 200
    assert payload["status"] == "disabled"
    assert payload["action"] is None
    assert state.snapshot()["jev"]["enabled"] is False


def test_jev_endpoint_with_real_advisor_does_not_spend_on_replay(yxt_frames, tmp_path) -> None:
    def fail_if_called(payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        pytest.fail("Replay must not call the hosted Jev transport")

    advisor = JevAdvisor(
        "test-only-key",
        ledger_path=tmp_path / "jev.sqlite3",
        transport=fail_if_called,
        clock=lambda: yxt_frames[0].event_time,
    )
    state = _dashboard(yxt_frames, advisor, mode=RunMode.REPLAY)
    with _server(state) as base:
        status, payload = _post(base, {"symbol": "YXT"})
    assert status == 200
    assert payload["status"] == "blocked"
    assert advisor.metadata()["requests_used"] == 0


def test_jev_endpoint_uses_current_server_context_and_updates_budget(yxt_frames) -> None:
    advisor = RecordingAdvisor()
    state = _dashboard(yxt_frames, advisor)
    before = state.snapshot()
    with _server(state) as base:
        status, payload = _post(
            base,
            {
                "symbol": "YXT",
                "source_time": before["meta"]["quote_time"],
                "snapshot": {"quote": {"last": 999999}, "meta": {"symbol": "FORGED"}},
                "quote": {"last": 999999},
            },
        )
    assert status == 200
    assert payload["action"] == "BUY"
    assert len(advisor.contexts) == 1
    assert advisor.contexts[0]["quote"] == before["quote"]
    assert advisor.contexts[0]["meta"] == before["meta"]
    assert state.snapshot()["meta"]["state"] == before["meta"]["state"]
    assert state.snapshot()["jev"]["requests_used"] == 1


@pytest.mark.parametrize("body", [{"symbol": "OTHER"}, {"symbol": "YXT", "source_time": "old"}])
def test_jev_endpoint_does_not_spend_for_changed_context(yxt_frames, body) -> None:
    advisor = RecordingAdvisor()
    state = _dashboard(yxt_frames, advisor)
    with _server(state) as base:
        status, _ = _post(base, body)
    assert status == 409
    assert advisor.contexts == []


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "https://untrusted.example"},
        {"Origin": "null"},
        {"Host": "untrusted.example"},
        {"Host": "127.0.0.1:1"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Sec-Fetch-Site": "same-site"},
        {"Content-Type": "text/plain"},
    ],
)
def test_jev_endpoint_rejects_cross_origin_or_non_json_spend(yxt_frames, headers) -> None:
    advisor = RecordingAdvisor()
    state = _dashboard(yxt_frames, advisor)
    with _server(state) as base:
        status, _ = _post(base, {"symbol": "YXT"}, headers)
    assert status in {403, 415}
    assert advisor.contexts == []


def test_jev_request_does_not_hold_snapshot_lock_and_discards_new_provider_error(yxt_frames) -> None:
    advisor = RecordingAdvisor()
    state = _dashboard(yxt_frames, advisor)
    advisor.during_request = state.mark_error
    with _server(state) as base:
        status, payload = _post(base, {"symbol": "YXT"})
    assert status == 200
    assert payload["status"] == "blocked"
    assert payload["action"] is None
    assert payload["probabilities"] == {}
    assert len(advisor.contexts) == 1


def test_jev_accepts_equivalent_source_timestamp_formats(yxt_frames) -> None:
    advisor = RecordingAdvisor()
    state = _dashboard(yxt_frames, advisor)
    advisor.result_update = {"source_time": state.snapshot()["meta"]["quote_time"].replace("+00:00", "Z")}
    with _server(state) as base:
        status, payload = _post(base, {"symbol": "YXT"})
    assert status == 200
    assert payload["status"] == "available"


@pytest.mark.parametrize(
    "update",
    [
        {"expires_at": "2000-01-01T00:00:00+00:00"},
        {"source_time": "invalid"},
        {"symbol": "OTHER"},
    ],
)
def test_jev_discards_expired_or_mismatched_results(yxt_frames, update) -> None:
    advisor = RecordingAdvisor()
    advisor.result_update = update
    state = _dashboard(yxt_frames, advisor)
    with _server(state) as base:
        status, payload = _post(base, {"symbol": "YXT"})
    assert status == 200
    assert payload["status"] == "blocked"
    assert payload["action"] is None
    assert payload["probabilities"] == {}


@pytest.mark.parametrize("result_status", ["unavailable", "cooldown", "budget_exhausted"])
def test_jev_endpoint_preserves_service_messages_without_result_timestamps(yxt_frames, result_status) -> None:
    advisor = RecordingAdvisor()
    advisor.result_update = {
        "status": result_status,
        "symbol": None,
        "source_time": None,
        "action": None,
        "probabilities": {},
        "message": "The local request limit or cooldown applies.",
        "expires_at": None,
    }
    state = _dashboard(yxt_frames, advisor)
    with _server(state) as base:
        status, payload = _post(base, {"symbol": "YXT"})
    assert status == 200
    assert payload["status"] == result_status
    assert payload["symbol"] == "YXT"
    assert payload["source_time"] == state.snapshot()["meta"]["quote_time"]
    assert payload["message"] == advisor.result_update["message"]


@pytest.mark.parametrize("context", [{"quote_age": 3}, {"quote_time_age": 3}, {"missing": ("float",)}])
def test_jev_endpoint_blocks_stalled_or_incomplete_data_without_spending(yxt_frames, context) -> None:
    advisor = RecordingAdvisor()
    state = _dashboard(yxt_frames, advisor, **context)
    with _server(state) as base:
        status, payload = _post(base, {"symbol": "YXT"})
    assert status == 200
    assert payload["status"] == "blocked"
    assert advisor.contexts == []


@pytest.mark.parametrize(
    "decision_state",
    [
        DecisionState.DATA_STALE, DecisionState.DATA_INSUFFICIENT, DecisionState.NO_TRADE,
        DecisionState.DAY_STOP, DecisionState.SELL,
    ],
)
def test_jev_endpoint_keeps_risk_state_authoritative(yxt_frames, decision_state) -> None:
    advisor = RecordingAdvisor()
    state = _dashboard(yxt_frames, advisor, decision_state=decision_state)
    with _server(state) as base:
        status, payload = _post(base, {"symbol": "YXT"})
    assert status == 200
    assert payload["status"] == "blocked"
    assert payload["action"] is None
    assert advisor.contexts == []


def test_jev_browser_controls_and_positions_runtime() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed for the browser JavaScript runtime check")
    script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync(process.argv[1], 'utf8');
const nodes = new Map();
function element() {
  return {
    textContent: '', children: [], hidden: false, disabled: false,
    classList: { toggle() {} },
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; },
    setAttribute(name, value) { this[name] = value; },
  };
}
const document = {
  getElementById(id) { if (!nodes.has(id)) nodes.set(id, element()); return nodes.get(id); },
  createElement() { return element(); },
};
const runtime = `
(async () => {
  renderPositions([{symbol:'YXT', status:'OPEN', current_r:1.2, quote_age:0.4}]);
  assert.equal($('positions').children.length, 1);
  function snapshot() {
    return {
      ready:true, complete:false, error:null, missing:[], positions:[], plan:{}, day_stop:{locked:false},
      meta:{symbol:'YXT', mode:'live', state:'WATCH', position_status:'FLAT',
        quote_time:new Date().toISOString(), quote_age:0},
      jev:{enabled:true, model:'jev-test', requests_used:0, request_limit:100,
        estimated_cost_usd:0, maximum_quote_age_seconds:2},
    };
  }
  function assessment() {
    return {status:'available', action:'BUY', symbol:'YXT', model:'jev-test',
      source_time:new Date().toISOString(), expires_at:new Date(Date.now()+30000).toISOString(),
      confidence:0.8, probabilities:{BUY:0.8,HOLD:0.1,SELL:0.05,WAIT:0.05},
      message:'<img src=x onerror=alert(1)>', requests_used:1, estimated_cost_usd:0.001};
  }
  let calls = 0;
  fetch = async (url, options) => {
    calls += 1;
    assert.equal(url, '/api/jev');
    assert.equal(JSON.stringify(JSON.parse(options.body)), JSON.stringify({symbol:'YXT'}));
    return {ok:true, json:async () => assessment()};
  };
  ui.snapshot = snapshot();
  ui.snapshot.jev.enabled = false;
  renderJev();
  await requestJev();
  assert.equal(calls, 0);
  ui.snapshot = snapshot();
  ui.snapshot.meta.mode = 'replay';
  renderJev();
  await requestJev();
  assert.equal(calls, 0);
  ui.snapshot = snapshot();
  renderJev();
  assert.equal(calls, 0);
  await requestJev();
  assert.equal(calls, 1);
  assert.equal(ui.snapshot.meta.state, 'WATCH');
  assert.equal($('jevMessage').textContent, '<img src=x onerror=alert(1)>');
  assert.equal($('jevProbabilities').children.length, 4);
  assert.match($('jevDetails').textContent, /80.0% decision confidence/);
  assert.match($('jevDetails').textContent, /as of /);
  ui.snapshot.meta.quote_time = new Date(Date.now()-3000).toISOString();
  renderJev();
  assert.equal(ui.jev.result, null);
  assert.equal($('jevRequest').disabled, true);
  assert.equal(calls, 1);
  for (const state of ['DATA_STALE','DATA_INSUFFICIENT','NO_TRADE','DAY_STOP','SELL']) {
    ui.snapshot = snapshot();
    ui.jev.result = {...assessment(), contextKey:jevContextKey(ui.snapshot)};
    ui.snapshot.meta.state = state;
    renderJev();
    assert.equal(ui.jev.result, null);
    assert.equal($('jevRequest').disabled, true);
  }
  ui.snapshot = snapshot();
  ui.jev.result = {...assessment(), contextKey:jevContextKey(ui.snapshot)};
  ui.snapshot.meta.symbol = 'OTHER';
  renderJev();
  assert.equal(ui.jev.result, null);
  ui.snapshot = snapshot();
  ui.jev.result = {...assessment(), expires_at:new Date(Date.now()-1).toISOString(),
    contextKey:jevContextKey(ui.snapshot)};
  renderJev();
  assert.equal(ui.jev.result, null);
  assert.match($('jevMessage').textContent, /expired/);
  ui.snapshot = snapshot();
  let resolveRequest;
  fetch = () => new Promise((resolve) => { resolveRequest = resolve; });
  renderJev();
  const pending = requestJev();
  ui.snapshot.meta.state = 'SELL';
  renderJev();
  resolveRequest({ok:true, json:async () => assessment()});
  await pending;
  assert.equal(ui.jev.result, null);
  assert.equal(ui.snapshot.meta.state, 'SELL');
  ui.snapshot = snapshot();
  ui.snapshot.jev.requests_used = 100;
  renderJev();
  assert.equal($('jevRequest').disabled, true);
})()
`;
Promise.resolve(vm.runInNewContext(
  source.slice(0, source.indexOf('\nfunction renderScreener')) + runtime,
  {document, assert, fetch:undefined},
)).catch((error) => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run(
        [node, "-e", script, str(STATIC_ROOT / "app.js")], capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
