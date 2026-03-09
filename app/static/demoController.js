my import { pushHistory, drawHistory } from "./historyChart.js";

const API_BASE = window.location.origin;
const STREAM_URL = `${API_BASE}/comparison/stream?max_frames=500`;

let es = null;

let reconnectTimer = null;
let hasReceivedFrame = false;
let lastFrameTimestamp = null
let staleCheckTimer = null

function updateLastFrameTime(){

  const el = document.getElementById("lastFrameTime")
  if(!el) return

  const now = Date.now()

  if(!lastFrameTimestamp){
    el.textContent = "last frame — waiting"
    return
  }

  const delta = Math.floor((now - lastFrameTimestamp)/1000)

  const date = new Date(lastFrameTimestamp)
  const time = date.toLocaleTimeString()

  el.textContent = `last frame ${time} · ${delta}s ago`

  if(delta > 6){
    el.classList.add("frame-stale")
  }else{
    el.classList.remove("frame-stale")
  }
}

function setStreamStatus(state, detail = "") {
  const el = document.getElementById("streamStatus");
  if (!el) return;

  el.className = "stream-badge";

  if (state === "live") {
    el.classList.add("stream-live");
    el.textContent = detail ? `LIVE SSE · ${detail}` : "LIVE SSE";
  } else if (state === "reconnecting") {
    el.classList.add("stream-reconnecting");
    el.textContent = detail ? `RECONNECTING · ${detail}` : "RECONNECTING";
  } else {
    el.classList.add("stream-disconnected");
    el.textContent = detail ? `DISCONNECTED · ${detail}` : "DISCONNECTED";
  }
}

function labelFromHypothesisIndex(idx) {
  return [
    "BULLISH TREND",
    "VOLATILITY EVENT",
    "MEAN REVERSION"
  ][idx] || "UNKNOWN";
}

function mapBackendHypotheses(frame) {
  const hs = frame?.decision_state?.hypothesis_set?.hypotheses || [];
  if (!hs.length) return null;

  const buckets = {
    trend: 0,
    vol: 0,
    mean: 0
  };

  for (const h of hs) {
    const label = (h.label || "").toLowerCase();
    const pct = Math.round((h.weight || 0) * 100);

    if (label.includes("bull") || label.includes("trend")) {
      buckets.trend += pct;
    } else if (label.includes("vol")) {
      buckets.vol += pct;
    } else if (label.includes("mean") || label.includes("revert")) {
      buckets.mean += pct;
    }
  }

  let vals = [buckets.trend, buckets.vol, buckets.mean];
  const total = vals.reduce((a, b) => a + b, 0);

  if (total === 0) return null;

  vals = vals.map(v => Math.round((v / total) * 100));
  const diff = 100 - vals.reduce((a, b) => a + b, 0);
  vals[0] += diff;

  return vals;
}

function updateHypothesisBars(vals) {
  vals.forEach((v, i) => {
    const fill = document.getElementById(`hypFill${i}`);
    const val = document.getElementById(`hypVal${i}`);
    const staticFill = document.getElementById(`hypStatic${i}`);
    const staticVal = document.getElementById(`hypStaticVal${i}`);

    if (fill) fill.style.width = `${v}%`;
    if (val) val.textContent = `${v}%`;
    if (staticFill) staticFill.style.width = `${v}%`;
    if (staticVal) staticVal.textContent = `${v}%`;
  });

  const topIdx = vals.indexOf(Math.max(...vals));
  const note = document.getElementById("hypNote");
  if (!note) return;

  if (topIdx === 0) {
    note.textContent = "Trend continuation currently leads, but alternatives remain live.";
  } else if (topIdx === 1) {
    note.textContent = "Volatility risk is rising; the system is preserving multiple paths.";
  } else {
    note.textContent = "Mean reversion is gaining weight as incoming evidence shifts.";
  }
}

function updateLLM(frame, vals) {
  const answerEl = document.getElementById("llmAnswer");
  const subEl = document.getElementById("llmAnswerSub");
  const noteEl = document.getElementById("llmTimelineNote");

  const parsed = frame?.llm_state?.parsed_label;
  let idx = vals.indexOf(Math.max(...vals));

  if (parsed) {
    const p = parsed.toLowerCase();
    if (p.includes("vol")) idx = 1;
    else if (p.includes("mean") || p.includes("revert")) idx = 2;
    else if (p.includes("bull") || p.includes("trend")) idx = 0;
  }

  const labels = [
    ["BULLISH TREND", "The LLM surface compresses the stream into one dominant directional label."],
    ["VOLATILITY EVENT", "The LLM surface flips to a volatility-centric interpretation."],
    ["MEAN REVERSION", "The LLM surface now emits a reversion narrative as the single answer."]
  ];

  if (answerEl) answerEl.textContent = labels[idx][0];
  if (subEl) subEl.textContent = labels[idx][1];
  if (noteEl) noteEl.textContent = "The LLM emits one label at a time, so changes appear as answer flips rather than smooth uncertainty shifts.";

  pushLLMHistory(idx);
  drawLLMTimelineChart();
}

function updateActions(frame, vals) {
  const actionList = document.getElementById("actionList");
  if (!actionList) return;

  const ranked = frame?.decision_state?.ranked_actions;
  if (Array.isArray(ranked) && ranked.length) {
    actionList.innerHTML = ranked.slice(0, 3).map((r, i) => `
      <div class="action-row">
        <div class="action-left"><span class="action-rank">${i + 1}.</span> ${r.action}</div>
        <div class="action-score">${Number(r.score).toFixed(2)}</div>
      </div>
    `).join("");
    return;
  }

  const [trend, vol, mean] = vals;
  const rows = [
    { name: trend >= mean ? "Hold / wait for confirmation" : "Fade extension", score: (0.55 + trend / 200).toFixed(2) },
    { name: trend > vol ? "Long breakout" : "Hedge / volatility defense", score: (0.40 + Math.max(trend, vol) / 250).toFixed(2) },
    { name: mean > trend ? "Mean reversion entry" : "Reduce exposure", score: (0.28 + mean / 300).toFixed(2) }
  ];

  actionList.innerHTML = rows.map((r, i) => `
    <div class="action-row">
      <div class="action-left"><span class="action-rank">${i + 1}.</span> ${r.name}</div>
      <div class="action-score">${r.score}</div>
    </div>
  `).join("");
}

function renderFrame(frame) {
  const vals = mapBackendHypotheses(frame);
  if (!vals) return;

  updateHypothesisBars(vals);
  pushHistory(vals);
  drawHistory();
  updateLLM(frame, vals);
  updateActions(frame, vals);
}

function connectStream() {
  if (es) es.close();
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }

  setStreamStatus("reconnecting", "connecting");
  hasReceivedFrame = false;

  es = new EventSource(STREAM_URL);

  es.onopen = () => {
    setStreamStatus("reconnecting", "stream open");
  };

  es.onmessage = (event) => {
    try {
      const frame = JSON.parse(event.data);

      if (frame.stage === "config") {
        return;
      }

      if (frame.stage === "end") {
        setStreamStatus("disconnected", "stream ended");
        es.close();
        return;
      }

      renderFrame(frame);

      if (!hasReceivedFrame) {
        hasReceivedFrame = true;
        setStreamStatus("live");
      }
    } catch (err) {
      console.error("Bad frame", err);
    }
  };

  es.onerror = () => {
    console.error("SSE disconnected; retrying soon");

    try { es.close(); } catch (_) {}

    setStreamStatus(
      hasReceivedFrame ? "reconnecting" : "disconnected",
      hasReceivedFrame ? "retrying" : "no data"
    );

    reconnectTimer = setTimeout(() => {
      connectStream();
    }, 2000);
  };
}

/* ─────────────────────────────
   LLM ANSWER TIMELINE
───────────────────────────── */
const llmTimeline = {
  maxPoints: 36,
  labels: [0]
};

function pushLLMHistory(labelIdx) {
  llmTimeline.labels.push(labelIdx);
  if (llmTimeline.labels.length > llmTimeline.maxPoints) {
    llmTimeline.labels.shift();
  }
}

function drawLLMTimelineChart() {
  const canvas = document.getElementById("llmTimelineChart");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;

  if (canvas.width !== Math.round(rect.width * dpr) || canvas.height !== Math.round(rect.height * dpr)) {
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
  }

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);

  const pad = { top: 14, right: 12, bottom: 24, left: 108 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  const levels = [
    { y: pad.top + plotH * 0.15, label: "Bullish trend" },
    { y: pad.top + plotH * 0.50, label: "Volatility event" },
    { y: pad.top + plotH * 0.85, label: "Mean reversion" }
  ];

  ctx.strokeStyle = "#e8ebee";
  ctx.lineWidth = 1;
  ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "#6b7785";

  levels.forEach((level) => {
    ctx.beginPath();
    ctx.moveTo(pad.left, level.y);
    ctx.lineTo(w - pad.right, level.y);
    ctx.stroke();
    ctx.fillText(level.label, pad.left - 8, level.y);
  });

  ctx.strokeStyle = "#cfd6dc";
  ctx.beginPath();
  ctx.moveTo(pad.left, h - pad.bottom);
  ctx.lineTo(w - pad.right, h - pad.bottom);
  ctx.stroke();

  const colors = ["#ff8f00", "#c0392b", "#7b61ff"];
  const labels = llmTimeline.labels;
  if (labels.length < 2) return;

  ctx.lineWidth = 2.5;
  ctx.strokeStyle = "#5d6b78";
  ctx.beginPath();

  labels.forEach((labelIdx, i) => {
    const x = pad.left + (i / (llmTimeline.maxPoints - 1)) * plotW;
    const y = levels[labelIdx].y;

    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, levels[labels[i - 1]].y);
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();

  labels.forEach((labelIdx, i) => {
    const x = pad.left + (i / (llmTimeline.maxPoints - 1)) * plotW;
    const y = levels[labelIdx].y;
    ctx.beginPath();
    ctx.arc(x, y, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = colors[labelIdx];
    ctx.fill();
  });

  for (let i = 1; i < labels.length; i++) {
    if (labels[i] !== labels[i - 1]) {
      const x = pad.left + (i / (llmTimeline.maxPoints - 1)) * plotW;
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, h - pad.bottom);
      ctx.strokeStyle = "rgba(192,57,43,0.18)";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }
}

window.addEventListener("resize", () => {
  drawHistory();
  drawLLMTimelineChart();
});

pushLLMHistory(0);
drawLLMTimelineChart();
drawHistory();
setStreamStatus("reconnecting", "starting");
connectStream();