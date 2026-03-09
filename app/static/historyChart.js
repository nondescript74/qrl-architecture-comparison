export const history = {
  max: 36,
  series: [
    [42],
    [31],
    [27]
  ]
};

export function pushHistory(vals) {
  vals.forEach((v, i) => {
    history.series[i].push(v);
    if (history.series[i].length > history.max) {
      history.series[i].shift();
    }
  });
}

export function drawHistory() {
  const canvas = document.getElementById("hypHistoryChart");
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

  const pad = { top: 14, right: 12, bottom: 22, left: 34 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  ctx.strokeStyle = "#e8ebee";
  ctx.lineWidth = 1;

  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (plotH / 4) * i;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();
  }

  ctx.fillStyle = "#6b7785";
  ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";

  [100, 75, 50, 25, 0].forEach((val, i) => {
    const y = pad.top + (plotH / 4) * i;
    ctx.fillText(`${val}`, pad.left - 6, y);
  });

  ctx.strokeStyle = "#cfd6dc";
  ctx.beginPath();
  ctx.moveTo(pad.left, h - pad.bottom);
  ctx.lineTo(w - pad.right, h - pad.bottom);
  ctx.stroke();

  const colors = ["#0072ff", "#00875a", "#ff8f00"];

  history.series.forEach((series, i) => {
    ctx.strokeStyle = colors[i];
    ctx.lineWidth = 2;
    ctx.beginPath();

    series.forEach((v, x) => {
      const px = pad.left + (x / (history.max - 1)) * plotW;
      const py = pad.top + ((100 - v) / 100) * plotH;
      if (x === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });

    ctx.stroke();

    const last = series.length - 1;
    const px = pad.left + (last / (history.max - 1)) * plotW;
    const py = pad.top + ((100 - series[last]) / 100) * plotH;
    ctx.beginPath();
    ctx.arc(px, py, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = colors[i];
    ctx.fill();
  });
}