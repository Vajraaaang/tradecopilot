from __future__ import annotations

import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from tradecopilot.config import StrategyConfig
from tradecopilot.explain import DeterministicExplainer
from tradecopilot.jev import JevAdvisor
from tradecopilot.models import DataQuality, DecisionState, MarketFrame, RunMode
from tradecopilot.strategy import DecisionEngine
from tradecopilot.webapp import STATIC_ROOT, DashboardState


def _price_frame(yxt_frames, *, symbol="AAPL", last="123.45", age=0) -> MarketFrame:
    from tradecopilot.models import PriceSnapshot

    now = datetime.now(UTC)
    price = PriceSnapshot(
        symbol=symbol,
        last=Decimal(last),
        previous_close=Decimal("120"),
        provider_timestamp=now - timedelta(seconds=age),
        receipt_timestamp=now,
        age_seconds=age,
        source="finnhub_quote",
        quality=DataQuality.LIMITED,
    )
    return yxt_frames[0].model_copy(
        update={
            "event_time": now,
            "mode": RunMode.LIVE,
            "quote": None,
            "price_snapshot": price,
            "bars_1m": (),
            "bars_5m": (),
            "level2_history": (),
            "time_and_sales": None,
            "position": None,
            "positions": (),
            "account_risk": None,
            "float_evidence": None,
            "catalyst_evidence": None,
            "resistance_levels": (),
            "gap_percent": None,
            "market_leader": False,
            "tradability_known": False,
            "historical_context": None,
            "screener_candidates": (),
        }
    )


def _update(state: DashboardState, frame: MarketFrame) -> None:
    decision = DecisionEngine(state.config).evaluate(frame)
    state.update(decision, frame, DeterministicExplainer().explain(decision, ()))


def test_finnhub_dashboard_uses_real_price_without_fabricating_quote_or_positions(yxt_frames) -> None:
    state = DashboardState(StrategyConfig())
    frame = _price_frame(yxt_frames)
    _update(state, frame)
    snapshot = state.snapshot()
    assert snapshot["meta"]["symbol"] == "AAPL"
    assert snapshot["meta"]["state"] == DecisionState.DATA_INSUFFICIENT
    assert snapshot["meta"]["price_only"] is True
    assert snapshot["meta"]["data_provider"] == "finnhub"
    assert snapshot["meta"]["position_status"] == "UNKNOWN"
    assert snapshot["quote"]["last"] == 123.45
    assert snapshot["quote"]["change"] == 3.45
    assert snapshot["quote"]["change_percent"] == 2.875
    assert snapshot["quote"]["bid"] is None
    assert snapshot["quote"]["ask"] is None
    assert snapshot["quote"]["total_volume"] is None
    assert snapshot["positions"] == []
    assert snapshot["charts"] == {"1m": [], "5m": []}
    assert "price only" in snapshot["capabilities"]["reason"].lower()
    assert snapshot["capabilities"]["minute_ohlcv"] is False
    assert snapshot["capabilities"]["broker_account"] is False
    assert snapshot["screener"][0]["symbol"] == "AAPL"
    assert snapshot["screener"][0]["price"] == 123.45
    assert snapshot["screener"][0]["gain_percent"] == 2.875
    assert snapshot["screener"][0]["pillars"] is None
    assert snapshot["quality"]["pillars_passed"] is None


def test_finnhub_dashboard_keeps_last_trade_age_after_market_close(yxt_frames) -> None:
    state = DashboardState(StrategyConfig())
    frame = _price_frame(yxt_frames, age=3600)
    _update(state, frame)
    snapshot = state.snapshot()
    assert frame.price_snapshot is not None
    assert snapshot["meta"]["quote_time"] == frame.price_snapshot.provider_timestamp.isoformat()
    assert snapshot["meta"]["quote_age"] == 3600
    assert snapshot["meta"]["state"] == DecisionState.DATA_INSUFFICIENT


def test_finnhub_symbol_change_does_not_reuse_old_price_or_allow_jev_spend(yxt_frames, tmp_path) -> None:
    calls: list[dict[str, Any]] = []

    def transport(payload: dict[str, Any], key: str) -> dict[str, Any]:
        calls.append(payload)
        pytest.fail("Incomplete price context must never spend a Jev request")

    advisor = JevAdvisor("test-only-key", ledger_path=tmp_path / "jev.sqlite3", transport=transport)
    state = DashboardState(StrategyConfig(), jev_advisor=advisor)
    _update(state, _price_frame(yxt_frames))
    _update(state, _price_frame(yxt_frames, symbol="MSFT", last="400.25"))
    with pytest.raises(ValueError, match="context changed"):
        state.jev_advice("AAPL")
    result = state.jev_advice("MSFT")
    assert result["status"] == "blocked"
    assert result["action"] is None
    assert state.snapshot()["quote"]["last"] == 400.25
    assert state.snapshot()["meta"]["symbol"] == "MSFT"
    assert advisor.metadata()["requests_used"] == 0
    assert calls == []


def test_finnhub_does_not_present_prior_broker_positions_as_current(yxt_frames) -> None:
    state = DashboardState(StrategyConfig())
    for frame in yxt_frames[:6]:
        _update(state, frame)
    assert state.snapshot()["positions"]
    _update(state, _price_frame(yxt_frames))
    assert state.snapshot()["positions"] == []
    assert state.snapshot()["meta"]["position_status"] == "UNKNOWN"


def test_complete_quote_still_controls_display_when_price_snapshot_is_also_present(yxt_frames) -> None:
    state = DashboardState(StrategyConfig())
    price = _price_frame(yxt_frames).price_snapshot
    frame = yxt_frames[0].model_copy(update={"price_snapshot": price})
    _update(state, frame)
    snapshot = state.snapshot()
    assert frame.quote is not None
    assert snapshot["quote"]["last"] == float(frame.quote.last)
    assert snapshot["quote"]["bid"] == float(frame.quote.bid)
    assert snapshot["quote"]["total_volume"] == frame.quote.total_volume
    assert snapshot["meta"]["price_only"] is False


def test_jev_market_opinion_uses_session_context_without_broker_data(yxt_frames, tmp_path) -> None:
    calls = []

    def transport(payload, key):
        calls.append(payload)
        return {
            "model": "jev-1.13.0",
            "answers": {
                "action": {
                    "type": "choice",
                    "choice": "BUY",
                    "confidence": 0.8,
                    "probabilities": {"BUY": 0.85, "HOLD": 0.05, "SELL": 0.05, "WAIT": 0.05},
                }
            },
            "usage": {"input_tokens": 900, "output_tokens": 32},
        }

    frame = _price_frame(yxt_frames, age=8 * 3600)
    frame = frame.model_copy(
        update={
            "price_snapshot": frame.price_snapshot.model_copy(
                update={
                    "session_open": Decimal("121"),
                    "session_high": Decimal("125"),
                    "session_low": Decimal("119"),
                }
            )
        }
    )
    advisor = JevAdvisor("test-key", tmp_path / "jev.sqlite3", transport=transport)
    state = DashboardState(StrategyConfig(), jev_advisor=advisor)
    _update(state, frame)
    before = state.snapshot()
    result = state.jev_advice("AAPL")
    assert result["status"] == "available" and result["action"] == "BUY"
    assert result["assessment_mode"] == "market_opinion"
    assert result["source_time"] == before["meta"]["quote_time"]
    assert state.snapshot()["meta"] == before["meta"]
    assert state.snapshot()["meta"]["state"] == "DATA_INSUFFICIENT"
    assert state.snapshot()["jev"]["opinion"]["ready"] is True
    assert calls[0]["state"]["session"]["session_high"] == 125
    assert "positions" not in calls[0]["state"]


def test_finnhub_browser_display_is_explicitly_limited() -> None:
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
  const classes = new Set();
  return {
    textContent:'', children:[], hidden:false, disabled:false,
    classList:{toggle(name,on) { if(on) classes.add(name); else classes.delete(name); },
      contains(name) { return classes.has(name); }},
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; },
    setAttribute(name,value) { this[name] = value; },
    querySelector() { return element(); },
  };
}
const document = {
  getElementById(id) { if(!nodes.has(id)) nodes.set(id,element()); return nodes.get(id); },
  createElement() { return element(); }, querySelector() { return element(); },
};
const runtime = `
renderCharts = () => updateChartReadout([]);
const snapshot = {
  ready:true, complete:false, error:null, missing:['full quote','minute bars','account risk'],
  positions:[], plan:{}, quality:{pillars_passed:null}, pattern:{}, indicators:{},
  next:[], history:[], alert:null, charts:{'1m':[],'5m':[]},
  meta:{symbol:'AAPL',mode:'live',state:'DATA_INSUFFICIENT',price_only:true,data_provider:'finnhub',
    position_status:'UNKNOWN',quote_age:3600,quote_time:new Date(Date.now()-3600000).toISOString(),
    event_time_pt:new Date().toISOString(),event_time_et:new Date().toISOString()},
  quote:{last:123.45,change:3.45,change_percent:2.875,bid:null,ask:null,total_volume:null},
  capabilities:{reason:'Finnhub price only; no minute OHLCV or broker connection.'},
  screener:[{symbol:'AAPL',state:'DATA_INSUFFICIENT',price:123.45,gain_percent:2.875,
    price_only:true,pillars:null,feedback:'Price only; setup not assessed'}],
  chat:{provider:'deterministic'},action:'Insufficient data',
  jev:{enabled:true,model:'jev-test',requests_used:0,request_limit:100,estimated_cost_usd:0},
};
renderSnapshot(snapshot);
assert.equal($('sessionMode').textContent,'FINNHUB PRICE · LIMITED DATA');
assert.equal($('lastPrice').textContent,'$123.45');
assert.match($('topClock').textContent,/LAST TRADE/);
assert.ok(parseFloat($('stripAge').textContent) >= 3600);
assert.match($('dataState').textContent,/LIMITED/);
assert.equal($('dataState').classList.contains('live'),false);
assert.match($('positions').children[0].textContent,/No broker connection/);
assert.ok($('screener').children[0].children.at(-1).textContent.includes('$123.45'));
assert.match($('screener').children[0].children.at(-1).textContent,/not assessed/i);
assert.equal($('briefQuality').textContent,'Not assessed');
assert.equal($('briefStructure').textContent,'Unavailable');
assert.equal($('briefData').textContent,'Price only');
assert.equal($('jevRequest').disabled,true);
assert.match($('chartStatus').textContent,/minute OHLCV/);
const canvasText = [];
chartFrame = () => ({ctx:{fillText(text) {canvasText.push(text);}},width:800,height:400});
drawPriceChart([],{});
assert.ok(canvasText.some((text) => /minute OHLCV/.test(text)));
snapshot.jev.opinion = {ready:true,reason:'Session-summary market opinion',context_kind:'session_summary',
  maximum_source_age_seconds:4*86400,sample_count:1};
renderSnapshot(snapshot);
assert.equal($('jevRequest').disabled,false);
assert.match($('jevTitle').textContent,/market opinion/i);
ui.jev.result = {status:'available',action:'BUY',assessment_mode:'market_opinion',context_kind:'session_summary',
  model:'jev-test',source_time:snapshot.meta.quote_time,expires_at:new Date(Date.now()+30000).toISOString(),
  confidence:0.8,probabilities:{BUY:0.85,HOLD:0.05,SELL:0.05,WAIT:0.05},
  message:'Informational opinion',contextKey:jevContextKey(snapshot)};
const changedRisk = JSON.parse(JSON.stringify(snapshot));
changedRisk.meta.state = 'SELL';
changedRisk.day_stop = {locked:true};
renderSnapshot(changedRisk);
assert.equal($('jevRequest').disabled,false);
assert.match($('jevResult').textContent,/BUY-leaning/);
assert.match($('jevDetails').textContent,/session summary/i);
changedRisk.meta.quote_time = new Date(Date.now()-5*86400000).toISOString();
changedRisk.meta.quote_age = 5*86400;
renderSnapshot(changedRisk);
assert.equal($('jevRequest').disabled,true);
assert.equal($('jevResult').textContent,'');
`;
vm.runInNewContext(source.slice(0,source.indexOf('\nconst priceCanvas')) + runtime,{document,assert});
"""
    result = subprocess.run(
        [node, "-e", script, str(STATIC_ROOT / "app.js")],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
