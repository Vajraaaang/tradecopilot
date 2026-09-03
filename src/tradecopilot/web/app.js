"use strict";

const ui = {
  timeframe: "1m",
  autoRotate: true,
  snapshot: null,
  windows: { "1m": null, "5m": null },
  hoverIndex: null,
  pinnedTimestamp: null,
  priceGeometry: null,
  dragging: false,
  dragMoved: false,
  dragStartX: 0,
  dragWindow: null,
  crosshair: true,
  chartType: "candles",
  logScale: false,
  wheelDelta: 0,
  studies: { volume: true, vwap: true, ema9: true, ema20: true, rsi: true, macd: true },
  drawingMode: null,
  drawingStart: null,
  drawings: { "1m": [], "5m": [] },
  alertsEnabled: typeof Notification !== "undefined" && Notification.permission === "granted",
  lastAlertSequence: 0,
};

const $ = (id) => document.getElementById(id);

const money = (value) => value == null ? "—" : `$${Number(value).toFixed(2)}`;
const fixed = (value, digits = 2) => value == null ? "—" : Number(value).toFixed(digits);
const compact = (value) => {
  if (value == null) return "—";
  return new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(value);
};

function setText(id, value) {
  const node = $(id);
  if (node) node.textContent = value == null ? "—" : String(value);
}

function setInputValue(id, value) {
  const node = $(id);
  if (node && document.activeElement !== node) node.value = value == null ? "" : String(value);
}

function stateTone(state) {
  if (["SELL", "EXIT_WARNING", "DATA_STALE", "DAY_STOP"].includes(state)) return "warning";
  if (["WATCH", "ARMED", "REENTRY_WATCH", "DATA_INSUFFICIENT", "NO_TRADE"].includes(state)) return "info";
  return "";
}

function updateAlertToggle() {
  const button = $("alertToggle");
  const supported = typeof Notification !== "undefined";
  button.disabled = !supported;
  button.textContent = !supported ? "ALERTS UNAVAILABLE" : ui.alertsEnabled ? "ALERTS ON" : "ALERTS OFF";
  button.classList.toggle("active", ui.alertsEnabled);
  button.setAttribute("aria-pressed", String(ui.alertsEnabled));
}

function deliverBrowserAlert(alert) {
  if (!alert || alert.sequence <= ui.lastAlertSequence) return;
  ui.lastAlertSequence = alert.sequence;
  if (!ui.alertsEnabled || Notification.permission !== "granted") return;
  const notification = new Notification(`${alert.symbol} · ${alert.state}`, {
    body: `${alert.action} Analysis signal only; execution is manual.`,
    tag: `tradecopilot-${alert.symbol}-${alert.state}`,
  });
  notification.onclick = () => window.focus();
}

function renderSnapshot(snapshot) {
  ui.snapshot = snapshot;
  if (!snapshot.ready) return;
  const meta = snapshot.meta;
  const quote = snapshot.quote;
  const indicators = snapshot.indicators;
  const plan = snapshot.plan;
  const simulated = ["replay", "mock"].includes(meta.mode);
  setText(
    "sessionMode",
    simulated
      ? `${meta.mode.toUpperCase()} SIMULATION · ${snapshot.complete ? "COMPLETE" : "PLAYING"} · NOT LIVE`
      : "LIVE READ-ONLY STREAM",
  );
  setText(
    "topClock",
    `${simulated ? "REPLAY " : "EVENT "}${meta.event_time_pt.slice(11, 19)} PT · ${meta.event_time_et.slice(11, 19)} ET`,
  );
  setText("decisionKind", simulated ? "REPLAYED ENGINE STATE" : "DETERMINISTIC LIVE STATE");
  setText("manualLabel", simulated ? "HISTORICAL SIMULATION · NOT LIVE" : "ANALYSIS ONLY · MANUAL EXECUTION");
  setInputValue("symbol", meta.symbol);
  setText("lastPrice", money(quote.last));
  setText("priceChange", `${quote.change >= 0 ? "+" : ""}${fixed(quote.change)} (${quote.change_percent >= 0 ? "+" : ""}${fixed(quote.change_percent)}%)`);
  $("priceChange").classList.toggle("negative", quote.change < 0);
  setText("state", meta.state);
  setText("action", simulated ? `Replay result: ${snapshot.action}` : snapshot.action);
  const ribbon = $("decisionRibbon");
  ribbon.className = `decision-ribbon ${stateTone(meta.state)}`.trim();
  setText("legendTrigger", money(plan.trigger));
  setText("legendStop", money(plan.stop));
  setText("stripTrigger", money(plan.trigger));
  setText("stripStop", money(plan.stop));
  setText("stripShares", plan.maximum_shares ?? "—");
  setText("stripRr", plan.reward_risk == null ? "—" : `${fixed(plan.reward_risk)}R`);
  setText("stripAge", meta.quote_age == null ? "—" : `${fixed(meta.quote_age)}s`);
  setText("stripMissing", snapshot.missing.length ? `${snapshot.missing.length} flagged` : "None");
  const dataState = $("dataState");
  dataState.textContent = snapshot.error ? "DEGRADED" : simulated ? `${meta.mode.toUpperCase()} DATA` : "LIVE DATA";
  dataState.classList.toggle("live", !snapshot.error);
  const chatMeta = snapshot.chat || { provider: "deterministic", model: "local", reasoning: "local" };
  const modelBadge = $("chatModelBadge");
  const gptEnabled = chatMeta.provider === "openai";
  modelBadge.textContent = gptEnabled ? `${chatMeta.model} · ${chatMeta.reasoning}` : "GPT OFFLINE";
  modelBadge.classList.toggle("openai", gptEnabled);
  $("chatSetup").hidden = gptEnabled;
  $("chatInput").disabled = !gptEnabled;
  $("chatInput").placeholder = gptEnabled ? "Ask GPT-5.6 Pro about this setup…" : "Configure OPENAI_API_KEY to enable GPT chat";
  $("chatForm").querySelector("button").disabled = !gptEnabled;
  document.querySelector(".quick-actions").hidden = !gptEnabled;
  deliverBrowserAlert(snapshot.alert);
  renderPositions(snapshot.positions);
  renderScreener(snapshot.screener);
  renderBrief(snapshot);
  renderCharts();
}

function renderPositions(positions) {
  const root = $("positions");
  root.replaceChildren();
  root.className = "widget-body";
  if (!positions.length) {
    root.className = "widget-body empty-state";
    const empty = document.createElement("p");
    empty.textContent = "No open or recent replay position.";
    root.append(empty);
    return;
  }
  positions.forEach((position) => {
    const card = document.createElement("article");
    card.className = "position-card";
    const top = document.createElement("div");
    top.className = "position-top";
    const symbol = document.createElement("span");
    symbol.className = "position-symbol";
    symbol.textContent = position.account_alias ? `${position.symbol} · ${position.account_alias}` : position.symbol;
    const status = document.createElement("span");
    status.className = `position-status ${position.status.startsWith("FLAT") ? "flat" : ""}`;
    status.textContent = position.status;
    top.append(symbol, status);
    const metrics = document.createElement("div");
    metrics.className = "position-metrics";
    [
      ["Quantity", position.quantity],
      ["Avg entry", money(position.average_entry)],
      ["Unrealized", money(position.unrealized_pnl)],
      ["Current R", position.current_r == null ? "—" : `${number(position.current_r, 2)}R`],
      ["MFE", money(position.mfe)],
      ["MAE", money(position.mae)],
      ["Stop", money(position.structural_stop)],
      ["Higher low", money(position.latest_higher_low)],
      ["VWAP", money(position.vwap)],
      ["EMA9", money(position.ema9)],
      ["Resistance", money(position.nearest_resistance)],
      ["Quote age", position.quote_age == null ? "—" : `${number(position.quote_age, 1)}s`],
    ].forEach(([label, value]) => {
      const cell = document.createElement("div");
      const small = document.createElement("span");
      small.textContent = label;
      const bold = document.createElement("b");
      bold.textContent = value == null ? "—" : String(value);
      cell.append(small, bold);
      metrics.append(cell);
    });
    const feedback = document.createElement("p");
    feedback.className = "position-feedback";
    feedback.textContent = position.feedback;
    card.append(top, metrics, feedback);
    root.append(card);
  });
}

function renderScreener(rows) {
  const root = $("screener");
  root.replaceChildren();
  root.className = "widget-body";
  setText("screenerCount", rows.length);
  if (!rows.length) {
    root.className = "widget-body empty-state";
    const empty = document.createElement("p");
    empty.textContent = "No qualifying candidates.";
    root.append(empty);
    return;
  }
  rows.forEach((row) => {
    const result = document.createElement("article");
    result.className = "screener-row";
    const values = [
      [row.symbol, "symbol"],
      [row.state, "state"],
      [`${fixed(row.gain_percent)}%`, "gain"],
      [fixed(row.rvol, 1), "rvol"],
      [compact(row.float_shares), "float"],
    ];
    values.forEach(([value, kind]) => {
      const cell = kind === "symbol" ? document.createElement("strong") : document.createElement("span");
      if (kind === "state") {
        cell.className = `screener-state ${stateTone(row.state) === "warning" ? "warn" : ""}`;
      }
      cell.textContent = value;
      result.append(cell);
    });
    const feedback = document.createElement("p");
    feedback.className = "screener-feedback";
    feedback.textContent = `${row.pillars}/5 pillars · ${row.feedback}`;
    result.append(feedback);
    root.append(result);
  });
}

function setupCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  if (canvas.width !== width * ratio || canvas.height !== height * ratio) {
    canvas.width = width * ratio;
    canvas.height = height * ratio;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
}

function renderCharts() {
  if (!ui.snapshot?.ready) return;
  const bars = visibleChartBars();
  $("rsiPanel").hidden = !ui.studies.rsi;
  $("macdPanel").hidden = !ui.studies.macd;
  $("priceChart").closest(".chart-widget").classList.toggle("hide-rsi", !ui.studies.rsi);
  $("priceChart").closest(".chart-widget").classList.toggle("hide-macd", !ui.studies.macd);
  document.querySelectorAll("[data-legend-study]").forEach((row) => {
    row.hidden = !ui.studies[row.dataset.legendStudy];
  });
  drawPriceChart(bars, ui.snapshot.plan);
  if (ui.studies.rsi) drawRsiChart(bars);
  if (ui.studies.macd) drawMacdChart(bars);
  updateChartReadout(bars);
}

function allChartBars() {
  return ui.snapshot?.charts?.[ui.timeframe] || [];
}

function visibleChartBars() {
  const bars = allChartBars();
  const window = ui.windows[ui.timeframe];
  if (!window) return bars;
  const count = Math.min(window.end - window.start, bars.length);
  const start = Math.max(0, Math.min(window.start, bars.length - count));
  return bars.slice(start, start + count);
}

function focusedBar(bars) {
  if (ui.pinnedTimestamp) {
    const pinned = bars.find((bar) => bar.t === ui.pinnedTimestamp);
    if (pinned) return pinned;
  }
  if (ui.hoverIndex != null && bars[ui.hoverIndex]) return bars[ui.hoverIndex];
  return bars.at(-1) || null;
}

function updateChartReadout(bars) {
  const bar = focusedBar(bars);
  setText("legendVolume", compact(bar?.v));
  setText("legendVwap", money(bar?.vwap));
  setText("legendEma9", money(bar?.ema9));
  setText("legendEma20", money(bar?.ema20));
  setText("rsiValue", fixed(bar?.rsi));
  setText("macdValue", `${fixed(bar?.macd, 3)} · signal ${fixed(bar?.macd_signal, 3)}`);
  const total = allChartBars().length;
  const suffix = ui.pinnedTimestamp && bars.some((candidate) => candidate.t === ui.pinnedTimestamp) ? " · PINNED" : "";
  setText(
    "chartStatus",
    `${ui.timeframe} · ${ui.chartType} · ${ui.logScale ? "log" : "linear"} · ${ui.snapshot.indicators.vwap_session || "VWAP unavailable"} · ${bars.length}/${total} bars${suffix}`,
  );
  updateTooltip(bar);
}

function chartFrame(canvasId) {
  const canvas = $(canvasId);
  const frame = setupCanvas(canvas);
  const { ctx, width, height } = frame;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  return { ...frame, canvas };
}

function drawGrid(ctx, width, height, left, right, top, bottom, rows = 5) {
  ctx.strokeStyle = "rgba(42, 46, 57, 0.09)";
  ctx.lineWidth = 1;
  for (let row = 0; row <= rows; row += 1) {
    const y = top + ((height - top - bottom) * row / rows);
    ctx.beginPath();
    ctx.moveTo(left, Math.round(y) + 0.5);
    ctx.lineTo(width - right, Math.round(y) + 0.5);
    ctx.stroke();
  }
  for (let column = 0; column <= 6; column += 1) {
    const x = left + ((width - left - right) * column / 6);
    ctx.beginPath();
    ctx.moveTo(Math.round(x) + 0.5, top);
    ctx.lineTo(Math.round(x) + 0.5, height - bottom);
    ctx.stroke();
  }
}

function drawPriceChart(bars, plan) {
  const { ctx, width, height } = chartFrame("priceChart");
  const left = 10;
  const right = 58;
  const top = 14;
  const bottom = 30;
  if (!bars.length) {
    drawEmpty(ctx, width, height, "Waiting for normalized OHLCV bars");
    return;
  }
  const priceValues = bars.flatMap((bar) => [bar.h, bar.l, bar.vwap, bar.ema9, bar.ema20]).filter((value) => value != null);
  [plan.trigger, plan.stop].forEach((value) => { if (value != null) priceValues.push(value); });
  let min = Math.min(...priceValues);
  let max = Math.max(...priceValues);
  const pad = Math.max((max - min) * 0.08, max * 0.002);
  min -= pad;
  max += pad;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const volumeHeight = Math.min(58, plotHeight * 0.2);
  const priceBottom = height - bottom - volumeHeight - 7;
  const priceHeight = priceBottom - top;
  const x = (index) => left + ((index + 0.5) * plotWidth / bars.length);
  const scalePrice = (value) => ui.logScale ? Math.log(Math.max(value, .000001)) : value;
  const unscalePrice = (value) => ui.logScale ? Math.exp(value) : value;
  const scaledMin = scalePrice(min);
  const scaledMax = scalePrice(max);
  const y = (value) => top + ((scaledMax - scalePrice(value)) / (scaledMax - scaledMin || 1)) * priceHeight;
  const priceFromY = (screenY) => unscalePrice(
    scaledMax - ((screenY - top) / priceHeight) * (scaledMax - scaledMin || 1),
  );
  ui.priceGeometry = { bars, left, right, top, bottom, width, height, plotWidth, priceBottom, x, y, priceFromY };
  drawGrid(ctx, width, priceBottom, left, right, top, 0, 5);

  ctx.font = "10px ui-sans-serif, system-ui";
  ctx.textAlign = "left";
  ctx.fillStyle = "#5f636e";
  for (let row = 0; row <= 5; row += 1) {
    const price = unscalePrice(scaledMax - ((scaledMax - scaledMin) * row / 5));
    const labelY = top + (priceHeight * row / 5);
    ctx.fillText(price.toFixed(2), width - right + 8, labelY + 3);
  }

  const maxVolume = Math.max(...bars.map((bar) => bar.v), 1);
  const slot = plotWidth / bars.length;
  const candleWidth = Math.max(2, Math.min(11, slot * 0.58));
  bars.forEach((bar, index) => {
    const cx = x(index);
    const rising = bar.c >= bar.o;
    const color = rising ? "#089981" : "#f23645";
    if (ui.chartType === "candles") {
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx, y(bar.h));
      ctx.lineTo(cx, y(bar.l));
      ctx.stroke();
      const bodyTop = y(Math.max(bar.o, bar.c));
      const bodyBottom = y(Math.min(bar.o, bar.c));
      ctx.fillRect(cx - candleWidth / 2, bodyTop, candleWidth, Math.max(1.5, bodyBottom - bodyTop));
    }
    if (ui.studies.volume) {
      const vh = (bar.v / maxVolume) * volumeHeight;
      ctx.fillStyle = color;
      ctx.globalAlpha = .22;
      ctx.fillRect(cx - candleWidth / 2, height - bottom - vh, candleWidth, vh);
      ctx.globalAlpha = 1;
    }
  });

  if (ui.chartType === "line") drawSeries(ctx, bars, "c", x, y, "#2962ff", 1.8);

  if (ui.studies.vwap) drawSeries(ctx, bars, "vwap", x, y, "#e91e63", 1.35);
  if (ui.studies.ema9) drawSeries(ctx, bars, "ema9", x, y, "#2962ff", 1.2);
  if (ui.studies.ema20) drawSeries(ctx, bars, "ema20", x, y, "#7e57c2", 1.35);
  drawPriceLevel(ctx, plan.trigger, y, left, width - right, "#089981", "TRIGGER");
  drawPriceLevel(ctx, plan.stop, y, left, width - right, "#f23645", "STOP");
  drawUserDrawings(ctx, bars, x, y, left, width - right, top, priceBottom);
  drawCrosshair(ctx, bars, x, y, left, width - right, top, priceBottom);

  const labelEvery = Math.max(1, Math.ceil(bars.length / 6));
  ctx.fillStyle = "#6f727b";
  ctx.textAlign = "center";
  bars.forEach((bar, index) => {
    if (index % labelEvery !== 0 && index !== bars.length - 1) return;
    const label = new Date(bar.t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    ctx.fillText(label, x(index), height - 8);
  });
}

function drawCrosshair(ctx, bars, x, y, left, right, top, bottom) {
  if (!ui.crosshair) return;
  let index = ui.hoverIndex;
  if (ui.pinnedTimestamp) index = bars.findIndex((bar) => bar.t === ui.pinnedTimestamp);
  if (index == null || index < 0 || !bars[index]) return;
  const bar = bars[index];
  const cx = x(index);
  const cy = y(bar.c);
  ctx.save();
  ctx.setLineDash([3, 4]);
  ctx.strokeStyle = ui.pinnedTimestamp ? "rgba(126,87,194,.92)" : "rgba(80,84,96,.48)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cx, top);
  ctx.lineTo(cx, bottom);
  ctx.moveTo(left, cy);
  ctx.lineTo(right, cy);
  ctx.stroke();
  ctx.restore();
  ctx.fillStyle = ui.pinnedTimestamp ? "#7e57c2" : "#5f636e";
  ctx.beginPath();
  ctx.arc(cx, cy, 3, 0, Math.PI * 2);
  ctx.fill();
}

function drawUserDrawings(ctx, bars, x, y, left, right, top, bottom) {
  const drawings = ui.drawings[ui.timeframe] || [];
  ctx.save();
  ctx.strokeStyle = "#2962ff";
  ctx.fillStyle = "#2962ff";
  ctx.lineWidth = 1.4;
  drawings.forEach((drawing) => {
    ctx.beginPath();
    if (drawing.kind === "horizontal") {
      const py = y(drawing.price);
      if (py < top || py > bottom) return;
      ctx.setLineDash([6, 4]);
      ctx.moveTo(left, py);
      ctx.lineTo(right, py);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = "700 9px ui-sans-serif, system-ui";
      ctx.textAlign = "right";
      ctx.fillText(`LEVEL ${drawing.price.toFixed(2)}`, right - 4, py - 5);
      return;
    }
    const first = bars.findIndex((bar) => bar.t === drawing.from.t);
    const second = bars.findIndex((bar) => bar.t === drawing.to.t);
    if (first < 0 || second < 0) return;
    ctx.setLineDash([]);
    ctx.moveTo(x(first), y(drawing.from.price));
    ctx.lineTo(x(second), y(drawing.to.price));
    ctx.stroke();
    [drawing.from, drawing.to].forEach((point, index) => {
      const barIndex = index === 0 ? first : second;
      ctx.beginPath();
      ctx.arc(x(barIndex), y(point.price), 3, 0, Math.PI * 2);
      ctx.fill();
    });
  });
  ctx.restore();
}

function updateTooltip(bar) {
  const tooltip = $("chartTooltip");
  const geometry = ui.priceGeometry;
  if (!bar || !geometry || (ui.hoverIndex == null && !ui.pinnedTimestamp)) {
    tooltip.hidden = true;
    return;
  }
  const index = geometry.bars.findIndex((candidate) => candidate.t === bar.t);
  if (index < 0) {
    tooltip.hidden = true;
    return;
  }
  const timestamp = new Date(bar.t).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  tooltip.textContent = `${timestamp} · ${ui.timeframe}${ui.pinnedTimestamp ? " · PINNED" : ""}\nO ${money(bar.o)}   H ${money(bar.h)}\nL ${money(bar.l)}   C ${money(bar.c)}\nVolume ${compact(bar.v)}\nVWAP ${money(bar.vwap)}   EMA9 ${money(bar.ema9)}\nEMA20 ${money(bar.ema20)}   RSI ${fixed(bar.rsi)}`;
  const cx = geometry.x(index);
  const cy = geometry.y(bar.c);
  tooltip.style.left = `${Math.max(245, Math.min(geometry.width - 230, cx + 12))}px`;
  tooltip.style.top = `${Math.max(50, Math.min(geometry.height - 138, cy - 54))}px`;
  tooltip.hidden = false;
}

function drawSeries(ctx, bars, field, x, y, color, width) {
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  let drawing = false;
  bars.forEach((bar, index) => {
    const value = bar[field];
    if (value == null) {
      drawing = false;
      return;
    }
    if (!drawing) ctx.moveTo(x(index), y(value));
    else ctx.lineTo(x(index), y(value));
    drawing = true;
  });
  ctx.stroke();
}

function drawPriceLevel(ctx, value, y, left, right, color, label) {
  if (value == null) return;
  const py = y(value);
  ctx.save();
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = color;
  ctx.globalAlpha = .72;
  ctx.beginPath();
  ctx.moveTo(left, py);
  ctx.lineTo(right, py);
  ctx.stroke();
  ctx.restore();
  ctx.fillStyle = color;
  ctx.font = "bold 9px ui-sans-serif, system-ui";
  ctx.textAlign = "right";
  ctx.fillText(`${label} ${value.toFixed(2)}`, right - 4, py - 4);
}

function drawRsiChart(bars) {
  const { ctx, width, height } = chartFrame("rsiChart");
  const left = 10;
  const right = 58;
  const top = 18;
  const bottom = 10;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const y = (value) => top + ((100 - value) / 100) * plotHeight;
  [70, 30].forEach((level) => {
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "#2d8cff";
    ctx.globalAlpha = .8;
    ctx.beginPath();
    ctx.moveTo(left, y(level));
    ctx.lineTo(width - right, y(level));
    ctx.stroke();
    ctx.restore();
    ctx.fillStyle = "#687487";
    ctx.font = "9px ui-sans-serif, system-ui";
    ctx.fillText(String(level), width - right + 9, y(level) + 3);
  });
  if (!bars.some((bar) => bar.rsi != null)) return;
  const x = (index) => left + ((index + .5) * plotWidth / bars.length);
  drawSeries(ctx, bars, "rsi", x, y, "#8fd5e6", 1.35);
}

function drawMacdChart(bars) {
  const { ctx, width, height } = chartFrame("macdChart");
  const left = 10;
  const right = 58;
  const top = 20;
  const bottom = 10;
  const available = bars.flatMap((bar) => [bar.macd, bar.macd_signal]).filter((value) => value != null);
  if (!available.length) {
    drawEmpty(ctx, width, height, "MACD unavailable — at least 34 bars required");
    return;
  }
  let min = Math.min(0, ...available);
  let max = Math.max(0, ...available);
  const spread = Math.max(max - min, .001);
  min -= spread * .12;
  max += spread * .12;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const x = (index) => left + ((index + .5) * plotWidth / bars.length);
  const y = (value) => top + ((max - value) / (max - min)) * plotHeight;
  ctx.strokeStyle = "rgba(130,145,165,.22)";
  ctx.beginPath();
  ctx.moveTo(left, y(0));
  ctx.lineTo(width - right, y(0));
  ctx.stroke();
  const slot = plotWidth / bars.length;
  bars.forEach((bar, index) => {
    if (bar.macd == null || bar.macd_signal == null) return;
    const histogram = bar.macd - bar.macd_signal;
    ctx.fillStyle = histogram >= 0 ? "rgba(19,241,149,.58)" : "rgba(255,91,25,.7)";
    const py = y(histogram);
    const zero = y(0);
    ctx.fillRect(x(index) - Math.min(5, slot * .32), Math.min(py, zero), Math.min(10, slot * .64), Math.max(1, Math.abs(zero - py)));
  });
  drawSeries(ctx, bars, "macd", x, y, "#8fd5e6", 1.25);
  drawSeries(ctx, bars, "macd_signal", x, y, "#e29b17", 1.25);
}

function drawEmpty(ctx, width, height, label) {
  ctx.fillStyle = "#657084";
  ctx.font = "12px ui-sans-serif, system-ui";
  ctx.textAlign = "center";
  ctx.fillText(label, width / 2, height / 2);
}

function renderBrief(snapshot) {
  if (!snapshot?.ready) return;
  const { meta, quality, pattern, plan, indicators } = snapshot;
  setText("briefState", meta.state);
  setText("briefAction", snapshot.action);
  setText("briefQuality", `${quality.pillars_passed}/5 pillars`);
  setText(
    "briefQualityDetail",
    `RVOL ${fixed(quality.rvol, 1)} · gain ${fixed(quality.gain_percent)}% · float ${compact(quality.float_shares)}`,
  );
  setText(
    "briefStructure",
    pattern.pullback_low == null ? "Pattern forming" : `${fixed(pattern.retracement_percent)}% retracement`,
  );
  setText(
    "briefStructureDetail",
    `Impulse ${money(pattern.impulse_low)}–${money(pattern.impulse_high)} · pullback ${money(pattern.pullback_low)}`,
  );
  setText("briefPlan", `${money(plan.trigger)} trigger`);
  setText(
    "briefPlanDetail",
    `Stop ${money(plan.stop)} · ${plan.maximum_shares ?? "—"} max shares · ${fixed(plan.reward_risk)}R available`,
  );
  setText("briefData", snapshot.missing.length ? `${snapshot.missing.length} limitations` : "Required data ready");
  setText(
    "briefDataDetail",
    `VWAP ${money(indicators.vwap)} · EMA9 ${money(indicators.ema9_1m)} · quote age ${fixed(meta.quote_age)}s`,
  );

  const reminders = $("briefReminders");
  reminders.replaceChildren();
  [
    ...snapshot.next.slice(0, 3),
    ...(snapshot.missing.length ? snapshot.missing.slice(0, 2) : ["Analysis signal only; every order remains manual."]),
  ].forEach((item) => {
    const row = document.createElement("li");
    row.textContent = item;
    reminders.append(row);
  });

  const history = $("briefHistory");
  history.replaceChildren();
  (snapshot.history || []).slice().reverse().forEach((transition) => {
    const row = document.createElement("div");
    row.className = "history-row";
    const state = document.createElement("b");
    state.textContent = transition.state;
    const time = document.createElement("time");
    time.dateTime = transition.timestamp;
    time.textContent = new Date(transition.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const reason = document.createElement("span");
    reason.textContent = transition.reason;
    row.append(state, time, reason);
    history.append(row);
  });
}

function setAnalysisDesk(open) {
  $("briefBackdrop").hidden = !open;
  $("briefDrawer").hidden = !open;
  $("analysisDeskTab").classList.toggle("active", open);
  $("analysisDeskTab").setAttribute("aria-pressed", String(open));
  $("stockCopilotTab").classList.toggle("active", !open);
  $("stockCopilotTab").setAttribute("aria-pressed", String(!open));
  if (open) $("briefClose").focus();
}

function closeToolMenus(except = null) {
  [
    ["chartTypeToggle", "chartTypeMenu"],
    ["indicatorToggle", "indicatorMenu"],
    ["drawingToggle", "drawingMenu"],
  ].forEach(([toggleId, menuId]) => {
    if (menuId === except) return;
    $(menuId).hidden = true;
    $(toggleId).setAttribute("aria-expanded", "false");
  });
}

function toggleToolMenu(toggleId, menuId) {
  const menu = $(menuId);
  const opening = menu.hidden;
  closeToolMenus(opening ? menuId : null);
  menu.hidden = !opening;
  $(toggleId).setAttribute("aria-expanded", String(opening));
}

function setDrawingMode(mode) {
  ui.drawingMode = mode;
  ui.drawingStart = null;
  const hint = $("drawingHint");
  hint.hidden = !mode;
  hint.textContent = mode === "trend"
    ? "Trend line: select the first point"
    : mode === "horizontal" ? "Horizontal line: select a price" : "";
  $("drawingToggle").classList.toggle("active", Boolean(mode));
  $("priceChart").classList.toggle("drawing", Boolean(mode));
}

function drawingPointForEvent(event) {
  const geometry = ui.priceGeometry;
  const index = hoverIndexForEvent(event);
  if (!geometry || index == null || !geometry.bars[index]) return null;
  const rect = $("priceChart").getBoundingClientRect();
  const localY = clamp(event.clientY - rect.top, geometry.top, geometry.priceBottom);
  return { t: geometry.bars[index].t, price: geometry.priceFromY(localY) };
}

function handleDrawingClick(event) {
  const point = drawingPointForEvent(event);
  if (!point) return;
  if (ui.drawingMode === "horizontal") {
    ui.drawings[ui.timeframe].push({ kind: "horizontal", price: point.price });
    setDrawingMode(null);
    renderCharts();
    return;
  }
  if (!ui.drawingStart) {
    ui.drawingStart = point;
    $("drawingHint").textContent = "Trend line: select the second point";
    renderCharts();
    return;
  }
  ui.drawings[ui.timeframe].push({ kind: "trend", from: ui.drawingStart, to: point });
  setDrawingMode(null);
  renderCharts();
}

function setVisibleMinutes(value) {
  const bars = allChartBars();
  if (!bars.length || value === "all") {
    ui.windows[ui.timeframe] = null;
  } else {
    const intervalMinutes = ui.timeframe === "5m" ? 5 : 1;
    const count = Math.min(bars.length, Math.max(2, Math.ceil(Number(value) / intervalMinutes)));
    ui.windows[ui.timeframe] = count >= bars.length ? null : { start: bars.length - count, end: bars.length };
  }
  document.querySelectorAll("[data-window-minutes]").forEach((button) => {
    button.classList.toggle("active", button.dataset.windowMinutes === String(value));
  });
  ui.pinnedTimestamp = null;
  ui.hoverIndex = null;
  renderCharts();
}

async function fetchSnapshot() {
  try {
    const response = await fetch("/api/snapshot", { cache: "no-store" });
    if (!response.ok) throw new Error("snapshot unavailable");
    renderSnapshot(await response.json());
  } catch (_) {
    const badge = $("dataState");
    badge.textContent = "OFFLINE";
    badge.classList.remove("live");
  }
}

let eventSource = null;
let reconnectTimer = null;

function connectEventStream() {
  if (!("EventSource" in window)) return;
  if (eventSource) eventSource.close();
  eventSource = new EventSource("/api/events");
  eventSource.addEventListener("snapshot", (event) => {
    try {
      renderSnapshot(JSON.parse(event.data));
    } catch (_) {
      fetchSnapshot();
    }
  });
  eventSource.onerror = () => {
    eventSource.close();
    eventSource = null;
    fetchSnapshot();
    window.clearTimeout(reconnectTimer);
    reconnectTimer = window.setTimeout(connectEventStream, 1500);
  };
}

function selectTimeframe(timeframe, fromRotation = false) {
  ui.timeframe = timeframe;
  ui.hoverIndex = null;
  ui.pinnedTimestamp = null;
  document.querySelectorAll(".interval").forEach((button) => {
    button.classList.toggle("active", button.dataset.timeframe === timeframe);
  });
  if (!fromRotation) {
    ui.autoRotate = false;
    updateRotationButton();
  }
  animateChartChange();
  if (ui.snapshot?.ready) renderSnapshot(ui.snapshot);
}

function animateChartChange() {
  const chart = $("priceChart").closest(".chart-widget");
  chart.classList.remove("chart-refresh");
  requestAnimationFrame(() => chart.classList.add("chart-refresh"));
}

function renderCurrentMarketStatus() {
  const eastern = Object.fromEntries(
    new Intl.DateTimeFormat("en-US", {
      timeZone: "America/New_York",
      weekday: "short",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    }).formatToParts(new Date()).map((part) => [part.type, part.value]),
  );
  const pacific = Object.fromEntries(
    new Intl.DateTimeFormat("en-US", {
      timeZone: "America/Los_Angeles",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    }).formatToParts(new Date()).map((part) => [part.type, part.value]),
  );
  const minutes = Number(eastern.hour) * 60 + Number(eastern.minute);
  const weekday = !["Sat", "Sun"].includes(eastern.weekday);
  const open = weekday && minutes >= 570 && minutes < 960;
  setText(
    "marketNow",
    `${open ? "REGULAR MARKET OPEN" : "REGULAR MARKET CLOSED"} · NOW ${pacific.hour}:${pacific.minute}:${pacific.second} PT`,
  );
  $("marketNow").classList.toggle("open", open);
}

function activeWindowRange() {
  const bars = allChartBars();
  const selected = ui.windows[ui.timeframe];
  return selected ? { ...selected } : { start: 0, end: bars.length };
}

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

function zoomChart(anchorRatio, factor) {
  const bars = allChartBars();
  if (bars.length < 2) return;
  const range = activeWindowRange();
  const currentCount = range.end - range.start;
  const minimumCount = Math.min(5, bars.length);
  let nextCount = clamp(Math.round(currentCount * factor), minimumCount, bars.length);
  if (nextCount === currentCount && factor < 1 && currentCount > minimumCount) nextCount -= 1;
  if (nextCount === currentCount && factor > 1 && currentCount < bars.length) nextCount += 1;
  if (nextCount === bars.length) {
    ui.windows[ui.timeframe] = null;
  } else {
    const anchor = range.start + anchorRatio * Math.max(1, currentCount - 1);
    const start = clamp(Math.round(anchor - anchorRatio * nextCount), 0, bars.length - nextCount);
    ui.windows[ui.timeframe] = { start, end: start + nextCount };
  }
  ui.hoverIndex = null;
  document.querySelectorAll("[data-window-minutes]").forEach((button) => button.classList.remove("active"));
  renderCharts();
}

function panChart(steps) {
  const bars = allChartBars();
  const range = activeWindowRange();
  const count = range.end - range.start;
  if (count >= bars.length) return;
  const start = clamp(range.start + steps, 0, bars.length - count);
  ui.windows[ui.timeframe] = { start, end: start + count };
  ui.hoverIndex = null;
  document.querySelectorAll("[data-window-minutes]").forEach((button) => button.classList.remove("active"));
  renderCharts();
}

function hoverIndexForEvent(event) {
  const geometry = ui.priceGeometry;
  if (!geometry?.bars?.length) return null;
  const rect = $("priceChart").getBoundingClientRect();
  const localX = event.clientX - rect.left;
  const raw = ((localX - geometry.left) / geometry.plotWidth) * geometry.bars.length;
  return clamp(Math.floor(raw), 0, geometry.bars.length - 1);
}

const priceCanvas = $("priceChart");

priceCanvas.addEventListener("pointerdown", (event) => {
  if (ui.drawingMode) return;
  ui.dragging = true;
  ui.dragMoved = false;
  ui.dragStartX = event.clientX;
  ui.dragWindow = activeWindowRange();
  try {
    priceCanvas.setPointerCapture(event.pointerId);
  } catch (_) {
    // Synthetic/test pointers and older browsers may not support capture.
  }
});

priceCanvas.addEventListener("pointermove", (event) => {
  if (ui.dragging && ui.dragWindow) {
    const distance = event.clientX - ui.dragStartX;
    if (Math.abs(distance) > 3) ui.dragMoved = true;
    if (ui.dragMoved) {
      const all = allChartBars();
      const count = ui.dragWindow.end - ui.dragWindow.start;
      const slot = Math.max(1, (ui.priceGeometry?.plotWidth || 1) / count);
      const start = clamp(ui.dragWindow.start - Math.round(distance / slot), 0, all.length - count);
      ui.windows[ui.timeframe] = count >= all.length ? null : { start, end: start + count };
    }
  } else {
    ui.hoverIndex = hoverIndexForEvent(event);
  }
  renderCharts();
});

priceCanvas.addEventListener("pointerup", (event) => {
  if (ui.drawingMode) {
    handleDrawingClick(event);
    return;
  }
  if (!ui.dragMoved) {
    ui.hoverIndex = hoverIndexForEvent(event);
    const bar = visibleChartBars()[ui.hoverIndex];
    ui.pinnedTimestamp = bar?.t === ui.pinnedTimestamp ? null : bar?.t || null;
  }
  ui.dragging = false;
  ui.dragWindow = null;
  if (priceCanvas.hasPointerCapture?.(event.pointerId)) priceCanvas.releasePointerCapture(event.pointerId);
  renderCharts();
});

priceCanvas.addEventListener("pointercancel", () => {
  ui.dragging = false;
  ui.dragWindow = null;
});

priceCanvas.addEventListener("pointerleave", () => {
  if (!ui.dragging && !ui.pinnedTimestamp) {
    ui.hoverIndex = null;
    renderCharts();
  }
});

priceCanvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  if (Math.abs(event.deltaX) > Math.abs(event.deltaY) && !event.ctrlKey && !event.metaKey) {
    panChart(Math.sign(event.deltaX) * Math.max(1, Math.round(Math.abs(event.deltaX) / 24)));
    return;
  }
  const rect = priceCanvas.getBoundingClientRect();
  const geometry = ui.priceGeometry;
  const plotLeft = geometry?.left || 0;
  const plotWidth = geometry?.plotWidth || rect.width;
  const anchor = clamp((event.clientX - rect.left - plotLeft) / Math.max(1, plotWidth), 0, 1);
  const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? rect.height : 1;
  ui.wheelDelta += event.deltaY * unit;
  if (Math.abs(ui.wheelDelta) < 12) return;
  const steps = clamp(ui.wheelDelta / 90, -2.5, 2.5);
  ui.wheelDelta = 0;
  zoomChart(anchor, Math.exp(steps * .16));
}, { passive: false });

priceCanvas.addEventListener("dblclick", () => {
  ui.windows[ui.timeframe] = null;
  ui.pinnedTimestamp = null;
  ui.hoverIndex = null;
  renderCharts();
});

priceCanvas.addEventListener("keydown", (event) => {
  if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
    event.preventDefault();
    panChart(event.key === "ArrowLeft" ? -1 : 1);
  } else if (event.key === "+" || event.key === "=") {
    event.preventDefault();
    zoomChart(.5, .8);
  } else if (event.key === "-" || event.key === "_") {
    event.preventDefault();
    zoomChart(.5, 1.2);
  } else if (event.key === "Escape") {
    ui.pinnedTimestamp = null;
    ui.hoverIndex = null;
    renderCharts();
  }
});

function updateRotationButton() {
  $("rotationToggle").textContent = `AUTO ROTATE ${ui.autoRotate ? "ON" : "OFF"}`;
  $("rotationToggle").classList.toggle("active", ui.autoRotate);
  $("rotationToggle").setAttribute("aria-pressed", String(ui.autoRotate));
}

document.querySelectorAll(".interval").forEach((button) => {
  button.addEventListener("click", () => selectTimeframe(button.dataset.timeframe));
});

$("rotationToggle").addEventListener("click", () => {
  ui.autoRotate = !ui.autoRotate;
  updateRotationButton();
});

async function askCopilot(query) {
  if (!query) return;
  appendMessage(query, "user");
  const submit = $("chatForm").querySelector("button");
  submit.disabled = true;
  submit.setAttribute("aria-busy", "true");
  const pending = appendMessage("Reasoning over the sanitized deterministic decision…", "assistant", "WORKING");
  pending.classList.add("pending");
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    if (!response.ok) throw new Error("chat unavailable");
    const payload = await response.json();
    pending.remove();
    appendMessage(payload.answer, "assistant", `${payload.model || "local"} · ${payload.reasoning || "deterministic"}`);
  } catch (_) {
    pending.remove();
    appendMessage("The explanation endpoint is unavailable. No action was taken.", "assistant", "SAFE FALLBACK");
  } finally {
    submit.disabled = false;
    submit.setAttribute("aria-busy", "false");
  }
}

$("chatForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = $("chatInput");
  const query = input.value.trim();
  input.value = "";
  await askCopilot(query);
});

document.querySelectorAll(".quick-actions button").forEach((button) => {
  button.addEventListener("click", () => askCopilot(button.dataset.query));
});

$("stockCopilotTab").addEventListener("click", () => setAnalysisDesk(false));
$("analysisDeskTab").addEventListener("click", () => setAnalysisDesk(true));
$("briefClose").addEventListener("click", () => setAnalysisDesk(false));
$("briefBackdrop").addEventListener("click", () => setAnalysisDesk(false));

$("indicatorToggle").addEventListener("click", (event) => {
  event.stopPropagation();
  toggleToolMenu("indicatorToggle", "indicatorMenu");
});

$("drawingToggle").addEventListener("click", (event) => {
  event.stopPropagation();
  toggleToolMenu("drawingToggle", "drawingMenu");
});

$("chartTypeToggle").addEventListener("click", (event) => {
  event.stopPropagation();
  toggleToolMenu("chartTypeToggle", "chartTypeMenu");
});

document.querySelectorAll("[data-chart-type]").forEach((button) => {
  button.addEventListener("click", () => {
    ui.chartType = button.dataset.chartType;
    $("chartTypeToggle").textContent = ui.chartType === "candles" ? "Candles" : "Line";
    document.querySelectorAll("[data-chart-type]").forEach((option) => {
      option.classList.toggle("active", option.dataset.chartType === ui.chartType);
    });
    closeToolMenus();
    animateChartChange();
    renderCharts();
  });
});

document.querySelectorAll("[data-study]").forEach((checkbox) => {
  checkbox.addEventListener("change", () => {
    ui.studies[checkbox.dataset.study] = checkbox.checked;
    renderCharts();
  });
});

document.querySelectorAll("[data-drawing]").forEach((button) => {
  button.addEventListener("click", () => {
    const mode = button.dataset.drawing;
    if (mode === "undo") {
      ui.drawings[ui.timeframe].pop();
      setDrawingMode(null);
      renderCharts();
    } else if (mode === "clear") {
      ui.drawings[ui.timeframe] = [];
      setDrawingMode(null);
      renderCharts();
    } else {
      setDrawingMode(mode);
    }
    closeToolMenus();
  });
});

$("crosshairToggle").addEventListener("click", () => {
  ui.crosshair = !ui.crosshair;
  $("crosshairToggle").classList.toggle("active", ui.crosshair);
  $("crosshairToggle").setAttribute("aria-pressed", String(ui.crosshair));
  renderCharts();
});

$("alertToggle").addEventListener("click", async () => {
  if (typeof Notification === "undefined") return;
  if (Notification.permission !== "granted") {
    const permission = await Notification.requestPermission();
    ui.alertsEnabled = permission === "granted";
  } else {
    ui.alertsEnabled = !ui.alertsEnabled;
  }
  updateAlertToggle();
});

$("logScaleToggle").addEventListener("click", () => {
  ui.logScale = !ui.logScale;
  $("logScaleToggle").classList.toggle("active", ui.logScale);
  $("logScaleToggle").setAttribute("aria-pressed", String(ui.logScale));
  renderCharts();
});

$("zoomIn").addEventListener("click", () => zoomChart(.5, .82));
$("zoomOut").addEventListener("click", () => zoomChart(.5, 1.22));
$("resetChart").addEventListener("click", () => $("fitChart").click());

$("fitChart").addEventListener("click", () => {
  ui.windows[ui.timeframe] = null;
  ui.pinnedTimestamp = null;
  ui.hoverIndex = null;
  setVisibleMinutes("all");
});

$("fullscreenChart").addEventListener("click", async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await document.documentElement.requestFullscreen();
  } catch (_) {
    setText("chartStatus", "Fullscreen is unavailable in this browser");
  }
});

document.addEventListener("fullscreenchange", () => {
  $("fullscreenChart").textContent = document.fullscreenElement ? "Exit full screen" : "Full screen";
  requestAnimationFrame(renderCharts);
});

document.querySelectorAll("[data-window-minutes]").forEach((button) => {
  button.addEventListener("click", () => setVisibleMinutes(button.dataset.windowMinutes));
});

document.addEventListener("click", (event) => {
  if (!event.target.closest(".tool-group")) closeToolMenus();
});

document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  closeToolMenus();
  if (!$("briefDrawer").hidden) setAnalysisDesk(false);
  if (ui.drawingMode) setDrawingMode(null);
});

function appendMessage(text, role, meta = null) {
  const log = $("chat");
  const message = document.createElement("div");
  message.className = `message ${role}`;
  const body = document.createElement("span");
  body.textContent = text;
  message.append(body);
  if (meta) {
    const detail = document.createElement("small");
    detail.textContent = meta;
    message.append(detail);
  }
  log.append(message);
  log.scrollTop = log.scrollHeight;
  return message;
}

$("symbolForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = $("symbol");
  const symbol = input.value.trim().toUpperCase();
  input.value = symbol;
  const feedback = $("symbolFeedback");
  feedback.hidden = false;
  feedback.className = "symbol-feedback loading";
  feedback.textContent = `Checking ${symbol || "symbol"}…`;
  try {
    const response = await fetch("/api/symbol", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol }),
    });
    if (!response.ok) throw new Error("symbol lookup unavailable");
    const payload = await response.json();
    feedback.className = `symbol-feedback ${payload.accepted ? "success" : "warning"}`;
    feedback.textContent = payload.message;
    if (!payload.accepted && ui.snapshot?.ready) input.value = ui.snapshot.meta.symbol;
  } catch (_) {
    feedback.className = "symbol-feedback warning";
    feedback.textContent = "Symbol lookup is unavailable; the current chart was not changed.";
  }
  window.setTimeout(() => { feedback.hidden = true; }, 5000);
});

window.addEventListener("resize", renderCharts);
setInterval(renderCurrentMarketStatus, 1000);
setInterval(() => {
  if (ui.autoRotate) selectTimeframe(ui.timeframe === "1m" ? "5m" : "1m", true);
}, 8000);
updateRotationButton();
updateAlertToggle();
renderCurrentMarketStatus();
fetchSnapshot();
connectEventStream();
