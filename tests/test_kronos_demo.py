"""The actual Kronos viewer can open honest, credential-free synthetic evidence."""

import argparse
import importlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

ROOT = Path(__file__).parents[1]
QUALITY_METRICS = (
    "terminal_mae_bps", "path_mae_bps", "direction_accuracy",
    "direction_accuracy_errors_as_incorrect", "raw_interval_coverage",
)


def _write_demo(directory):
    spec = importlib.util.find_spec("tradecopilot.forecast.kronos_demo")
    assert spec is not None, "The credential-free Kronos demo producer is missing"
    return importlib.import_module(spec.name).write_demo(directory)


def test_demo_seals_four_explicit_synthetic_cases_without_evaluating_toy_outcomes(tmp_path):
    from tradecopilot.forecast.contracts import content_hash
    from tradecopilot.forecast.kronos_report import load_report

    path = _write_demo(tmp_path / "demo")
    report = load_report(path)
    fixture = json.loads((path.parent / "fixture.json").read_bytes())
    assert path == tmp_path / "demo" / "report.json"
    assert report["evidence_mode"] == "synthetic_contract_fixture"
    assert report["connection"] == {"mode": "synthetic"}
    assert report["source_data_id"] == content_hash(fixture)
    assert set(report["inventory"]) == {"fixture.json", "protocol.json"}
    assert report["catalog_counts"] == {"planned": 4, "eligible": 4, "excluded": 0}
    assert report["model_inference_calls"] == report["provider_calls"] == report["broker_orders"] == 0
    assert report["prospective_unavailable"] == []
    protocol = report["protocol"]
    assert protocol["fixture"] is True
    assert (protocol["lookback"], protocol["horizon"], protocol["samples"], protocol["flat_bps"]) == (60, 15, 5, 10)
    assert {row["symbol"] for row in report["cases"]} == {"DEMOA", "DEMOB"}
    assert len({row["as_of"] for row in report["cases"]}) == 2
    assert len(report["cases"]) == 4
    for model in report["models"].values():
        assert model["metrics"]["evaluated"] is False
        assert all(model["metrics"][key] is None for key in QUALITY_METRICS)
    model = report["models"]["synthetic_fixture"]
    assert model["metadata"]["model"]["name"] == "Synthetic path fixture"
    assert model["metadata"]["kind"] == "synthetic_fixture"
    assert model["execution"] == model["metadata"]["execution"] == "fixture"
    assert report["models"]["persistence"]["metadata"]["kind"] == "fixed_past_only_control"
    for row, inputs in zip(report["cases"], fixture["cases"], strict=True):
        assert row["group"] == "historical" and row["synthetic"] is True
        assert row["outcome_status"] == "simulated"
        assert len(row["history_close"]) == len(row["history_times"]) == 60
        assert len(row["actual_close"]) == len(row["future_times"]) == 15
        assert row["history_times"][-1] == row["as_of"]
        assert datetime.fromisoformat(row["future_times"][0]) > datetime.fromisoformat(row["as_of"])
        assert row["history_close"] == inputs["history_close"]
        assert row["actual_close"] == inputs["actual_close"]
        for values in (row["history_close"], row["actual_close"]):
            assert all(math.isfinite(value) and value > 0 for value in values)
        forecast = row["forecasts"]["synthetic_fixture"]
        paths = inputs["toy_paths"]
        assert len(paths) == forecast["samples"] == 5
        assert all(len(values) == 15 for values in paths)
        assert all(math.isfinite(value) and value > 0 for values in paths for value in values)
        for key in ("mean_close", "lower_close", "upper_close"):
            assert len(forecast[key]) == 15
            assert all(math.isfinite(value) and value > 0 for value in forecast[key])
        for step in range(15):
            assert forecast["mean_close"][step] == pytest.approx(sum(values[step] for values in paths) / 5)
            assert forecast["lower_close"][step] <= forecast["mean_close"][step] <= forecast["upper_close"][step]
        assert row["forecasts"]["persistence"]["mean_close"] == [row["history_close"][-1]] * 15


def test_fixture_inputs_are_deterministic_and_new_bundles_are_immutable(tmp_path):
    from tradecopilot.forecast.kronos_report import load_report

    first = _write_demo(tmp_path / "first")
    original = {p.name: p.read_bytes() for p in first.parent.iterdir()}
    second = _write_demo(tmp_path / "second")
    assert (first.parent / "fixture.json").read_bytes() == (second.parent / "fixture.json").read_bytes()
    assert load_report(first)["source_data_id"] == load_report(second)["source_data_id"]
    with pytest.raises(FileExistsError, match="immutable"):
        _write_demo(first.parent)
    assert original == {p.name: p.read_bytes() for p in first.parent.iterdir()}
    with pytest.raises(ValueError, match="absolute"):
        _write_demo(Path("relative-demo"))


def test_demo_inventory_detects_fixture_tampering(tmp_path):
    from tradecopilot.forecast.kronos_report import load_report

    path = _write_demo(tmp_path / "demo")
    (path.parent / "fixture.json").write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="integrity"):
        load_report(path)


def test_demo_producer_runs_without_credentials_network_or_optional_model_imports(tmp_path):
    _write_demo(tmp_path / "check-feature")
    code = """
import builtins
import sys
from pathlib import Path
from tradecopilot.forecast.kronos_demo import write_demo
assert not {'torch', 'pandas', 'huggingface_hub', 'safetensors', 'einops', 'matplotlib'} & sys.modules.keys()
original = builtins.__import__
def guarded(name, *args, **kwargs):
    forbidden = {'torch', 'huggingface_hub', 'safetensors', 'einops', 'matplotlib', 'keyring', 'alpaca'}
    assert name.split('.')[0] not in forbidden
    assert not name.startswith(('tradecopilot.auth', 'tradecopilot.providers', 'tradecopilot.forecast.paper',
                                'tradecopilot.forecast.kronos_run', 'tradecopilot._vendor'))
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
def offline(event, args):
    assert event != 'socket.connect', 'The offline demo attempted a network connection'
sys.addaudithook(offline)
path = write_demo(Path(sys.argv[1]))
from tradecopilot.forecast.kronos_report import load_report
assert load_report(path)['connection'] == {'mode': 'synthetic'}
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "guarded")],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_explicit_kronos_demo_cli_writes_the_same_viewer_report(tmp_path, capsys):
    from tradecopilot.forecast.kronos_cli import add_parser, dispatch
    from tradecopilot.forecast.kronos_report import load_report

    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers(dest="forecast_command", required=True))
    args = parser.parse_args(["kronos", "demo", "--output-dir", str(tmp_path / "demo")])
    assert dispatch(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["evidence"] == "synthetic_contract_fixture"
    assert output["model_inference_calls"] == output["provider_calls"] == 0
    assert load_report(Path(output["report"]))["connection"] == {"mode": "synthetic"}


def test_demo_http_viewer_is_integrity_checked_and_read_only(tmp_path):
    from tradecopilot.forecast.kronos_report import load_report
    from tradecopilot.forecast.service import create_server

    path = _write_demo(tmp_path / "demo")
    original = {p.name: p.read_bytes() for p in path.parent.iterdir()}
    server = create_server(path, port=0, loader=load_report, dashboard_name="kronos_dashboard.html")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/api/report") as response:
            assert json.load(response)["report_id"] == load_report(path)["report_id"]
        with urlopen(base + "/health") as response:
            assert json.load(response) == {"ok": True, "write_capabilities": False}
        with urlopen(base + "/") as response:
            html = response.read()
            assert b"__CSP_NONCE__" not in html
            assert html.count(b"fetch('/api/report')") == 1
            assert "script-src 'nonce-" in response.headers["Content-Security-Policy"]
        for method in ("POST", "PUT", "PATCH", "DELETE", "OPTIONS"):
            with pytest.raises(HTTPError) as failure:
                urlopen(Request(base + "/api/report", data=b"{}", method=method))
            assert failure.value.code == 405
            assert failure.value.headers["Allow"] == "GET, HEAD"
        assert original == {p.name: p.read_bytes() for p in path.parent.iterdir()}
        (path.parent / "fixture.json").write_bytes(b"{}\n")
        with pytest.raises(HTTPError) as failure:
            urlopen(base + "/api/report")
        assert failure.value.code == 503
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_public_real_result_export_rejects_offline_demo(tmp_path):
    path = _write_demo(tmp_path / "demo")
    spec = importlib.util.spec_from_file_location("render_kronos_results", ROOT / "scripts/render_kronos_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="real development"):
        module.public_summary(path)


def test_viewer_labels_synthetic_cases_controls_and_empty_queue_without_changing_paper_semantics(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to execute the existing viewer JavaScript")
    path = _write_demo(tmp_path / "demo")
    script = r"""
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(process.argv[1],'utf8'),payload=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const source=html.match(/<script[^>]*>([\s\S]*?)<\/script>/)[1];
function element(name='div'){
  return {textContent:'',value:'',children:[],dataset:{},attributes:{},listeners:{},hidden:false,disabled:false,
    classList:{toggle(){}},clientWidth:1120,clientHeight:430,
    append(...children){this.children.push(...children);if(name==='select'&&!this.value)this.value=children[0]?.value||''},
    replaceChildren(...children){this.children=[];this.value='';this.append(...children)},
    setAttribute(key,value){this.attributes[key]=String(value)},getAttribute(key){return this.attributes[key]},
    addEventListener(key,callback){this.listeners[key]=callback},contains(){return false},focus(){}};
}
async function boot(report){
  const nodes=new Map();
  for(const match of html.matchAll(/<([\w-]+)[^>]*id="([^"]+)"[^>]*>([^<]*)/g)){
    const value=element(match[1]);value.textContent=match[3];nodes.set(match[2],value);
  }
  const $=id=>nodes.get(id);
  $('error').hidden=true;$('group').value='historical';let fetches=0;
  const document={getElementById:$,createElement:element,createElementNS:(_,name)=>element(name),activeElement:null};
  const context=vm.createContext({document,fetch:async url=>{
    assert.equal(url,'/api/report');fetches++;return {ok:true,json:async()=>report};
  }});
  vm.runInContext(source+'\nglobalThis.actions={chooseGroup,chooseCase,draw};',context);
  await new Promise(setImmediate);
  assert.equal($('error').hidden,true);assert.equal(fetches,1);
  return {$,actions:context.actions,document};
}
(async()=>{
  const {$,actions,document}=await boot(payload);
  assert.equal($('mode').textContent,'OFFLINE DEMO');assert.match(document.title,/synthetic/i);
  assert.equal($('account').textContent,'No broker connection');assert.equal($('feed').textContent,'Synthetic fixture');
  assert.equal($('connection-title').textContent,'Synthetic data source');
  assert.match($('market').textContent,/no market/i);
  assert.equal($('count').textContent,'4');assert.equal($('pending').textContent,'0');
  assert.equal($('model').value,'synthetic_fixture');
  assert.equal($('instrument-type').textContent,'Synthetic path fixture');
  assert.equal($('case-state').textContent,'SYNTHETIC');assert.match($('evidence-label').textContent,/not evaluated/i);
  assert.match($('forecast-status').textContent,/no inference/i);
  assert.match($('execution').textContent,/no model inference/i);
  assert.match($('model-revision').textContent,/no model loaded/i);
  assert.match($('tokenizer-revision').textContent,/no tokenizer/i);
  assert.match($('actuallegend').textContent,/simulated/i);assert.match($('bandlegend').textContent,/toy/i);
  assert.match($('chart').getAttribute('aria-label'),/synthetic context/i);
  assert.equal($('queuecount').textContent,'0');assert.match($('outcomes').textContent,/no prospective forecasts/i);
  assert.match($('queue').children[0].textContent,/no prospective forecasts/i);
  assert.match($('comparison-title').textContent,/not evaluated/i);
  assert.equal($('metrics').children[0].children[0].children[0].textContent,'Synthetic fixture');
  assert.ok($('metrics').children.every(row=>row.children.some(cell=>cell.textContent==='Not evaluated')));
  $('case').value='2';actions.chooseCase();assert.equal($('chart-symbol').textContent,'DEMOB');
  $('model').value='persistence';actions.draw();assert.match($('instrument-type').textContent,/synthetic/i);
  assert.match($('bandlegend').textContent,/control/i);assert.match($('execution').textContent,/no model inference/i);
  $('group').value='prospective';actions.chooseGroup();
  assert.equal($('case').disabled,true);assert.equal($('model').disabled,true);
  assert.match($('case-state').textContent,/synthetic/i);
  assert.match($('chart').getAttribute('aria-label'),/no prospective/i);
  assert.match($('queue-note').textContent,/no prospective forecasts/i);
  const real=structuredClone(payload);
  real.evidence_mode='retrospective_development_pilot';
  real.connection={mode:'paper',account:{status:'ACTIVE'},feed:'sip',clock:{timestamp:'2026-10-07T13:30:00+00:00',is_open:true}};
  real.models.synthetic_fixture.metadata={kind:'model_backend',model:{name:'Kronos-mini'}};
  real.models.synthetic_fixture.metrics={eligible:4,scored:4,errors:0,terminal_mae_bps:5,path_mae_bps:3,
    direction_accuracy_errors_as_incorrect:0.5,raw_interval_coverage:0.4};
  const paper=await boot(real);
  assert.equal(paper.$('mode').textContent,'ALPACA PAPER · RESEARCH');
  assert.equal(paper.$('account').textContent,'ACTIVE · PAPER');assert.equal(paper.$('feed').textContent,'SIP');
  assert.equal(paper.$('instrument-type').textContent,'Kronos forecast');
  assert.equal(paper.$('bandlegend').textContent,'Raw p10\u2013p90 · uncalibrated');
  assert.equal(paper.$('metrics').children[0].children[0].children[0].textContent,'Pinned local model');
  paper.$('model').value='persistence';paper.actions.draw();
  assert.equal(paper.$('instrument-type').textContent,'Past-only control');
})().catch(error=>{console.error(error);process.exitCode=1});
"""
    result = subprocess.run(
        [node, "-e", script, str(ROOT / "src/tradecopilot/forecast/kronos_dashboard.html"), str(path)],
        capture_output=True, text=True, timeout=15, check=False,
    )
    assert result.returncode == 0, result.stderr
