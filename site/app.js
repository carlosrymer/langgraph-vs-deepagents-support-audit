// Reads data/summary.json (built from artifacts/runs.jsonl by support_audit.report)
// and renders every chart and table. No framework, no build step.

const ARM_LABEL = {
  langgraph: "LangGraph (plain ReAct loop)",
  deepagents: "Deep Agents (defaults)",
  deepagents_no_offload: "Deep Agents, offload off",
};
const ARM_SHORT = { langgraph: "LangGraph", deepagents: "Deep Agents", deepagents_no_offload: "DA, no offload" };
const ARM_VAR = { langgraph: "--series-1", deepagents: "--series-2", deepagents_no_offload: "--series-3" };
const ARM_SHAPE = { langgraph: "circle", deepagents: "square", deepagents_no_offload: "diamond" };
const OUTCOME_LABEL = {
  pass: "pass", wrong_answer: "wrong answer", no_answer: "no answer",
  context_ceiling: "context ceiling", step_limit: "step limit", harness_error: "harness error",
};

const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs = {}, parent) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
};
const fmtInt = (n) => n.toLocaleString("en-US");
const fmtK = (n) => (n >= 1000 ? `${(n / 1000).toFixed(n >= 100000 ? 0 : 1)}k` : `${n}`);
const pct = (x) => (x == null ? "n/a" : `${Math.round(x * 100)}%`);
const usd = (x, d = 4) => (x == null ? "n/a" : `$${x.toFixed(d)}`);

const tip = document.getElementById("tip");
function showTip(evt, html) {
  tip.innerHTML = html;
  tip.style.display = "block";
  const pad = 14;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  const r = tip.getBoundingClientRect();
  if (x + r.width > window.innerWidth - 8) x = evt.clientX - r.width - pad;
  if (y + r.height > window.innerHeight - 8) y = evt.clientY - r.height - pad;
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}
const hideTip = () => (tip.style.display = "none");

function marker(g, shape, x, y, color) {
  const ring = css("--surface-1");
  if (shape === "circle") return el("circle", { cx: x, cy: y, r: 5, fill: color, stroke: ring, "stroke-width": 2 }, g);
  if (shape === "square") return el("rect", { x: x - 5, y: y - 5, width: 10, height: 10, rx: 2, fill: color, stroke: ring, "stroke-width": 2 }, g);
  return el("path", { d: `M${x} ${y - 6.5} L${x + 6.5} ${y} L${x} ${y + 6.5} L${x - 6.5} ${y} Z`, fill: color, stroke: ring, "stroke-width": 2 }, g);
}

function legend(target, arms) {
  target.innerHTML = arms
    .map((a) => `<span><svg width="18" height="12" style="width:18px;display:inline"><line x1="0" y1="6" x2="18" y2="6" stroke="${css(ARM_VAR[a])}" stroke-width="2"/></svg>${ARM_LABEL[a]}</span>`)
    .join("");
}

// Line chart over the ordinal size dial: one y-scale, CI whiskers, band for evicting sizes.
function lineChart(svgId, S, { value, lo, hi, yMax, yFmt, yTitle, tipFor, refLine, ticks = 4 }) {
  const svg = document.getElementById(svgId);
  svg.innerHTML = "";
  const W = 900, H = 340, m = { l: 56, r: 130, t: 16, b: 44 };
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const sizes = S.meta.sizes, arms = S.meta.arms;
  const xs = (i) => m.l + ((W - m.l - m.r) * (i + 0.5)) / sizes.length;
  const ys = (v) => H - m.b - ((H - m.t - m.b) * v) / yMax;
  const g = el("g", {}, svg);
  // evicting band (L, XL)
  const bandX = (xs(1) + xs(2)) / 2;
  el("rect", { x: bandX, y: m.t, width: W - m.r - bandX, height: H - m.t - m.b, fill: css("--band") }, g);
  const bt = el("text", { x: bandX + 8, y: m.t + 14, class: "axis-label" }, g);
  bt.textContent = "above Deep Agents' 20k-token offload threshold";
  // grid + y ticks
  for (let i = 0; i <= ticks; i++) {
    const v = (yMax * i) / ticks, y = ys(v);
    el("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, stroke: css("--grid"), "stroke-width": 1 }, g);
    const t = el("text", { x: m.l - 8, y: y + 4, "text-anchor": "end" }, g);
    t.textContent = yFmt(v);
  }
  const yt = el("text", { x: 14, y: (m.t + H - m.b) / 2, class: "axis-label", transform: `rotate(-90 14 ${(m.t + H - m.b) / 2})`, "text-anchor": "middle" }, g);
  yt.textContent = yTitle;
  sizes.forEach((s, i) => {
    const t = el("text", { x: xs(i), y: H - m.b + 18, "text-anchor": "middle", class: "axis-label" }, g);
    t.textContent = `${s} · ${fmtK(S.meta.payload_tokens_o200k[s])} tok/export`;
  });
  if (refLine) {
    const y = ys(refLine.value);
    el("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, stroke: css("--text-muted"), "stroke-width": 1 }, g);
    const t = el("text", { x: m.l + 4, y: y - 6 }, g);
    t.textContent = refLine.label;
  }
  const endLabels = [];
  arms.forEach((a, k) => {
    const color = css(ARM_VAR[a]);
    const off = (k - 1) * 9; // small dodge so whiskers don't overlap
    const pts = sizes.map((s, i) => [xs(i) + off, value(a, s)]).filter((p) => p[1] != null);
    el("path", { d: pts.map((p, j) => `${j ? "L" : "M"}${p[0]} ${ys(p[1])}`).join(" "), fill: "none", stroke: color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, g);
    sizes.forEach((s, i) => {
      const v = value(a, s);
      if (v == null) return;
      const x = xs(i) + off;
      if (lo) {
        el("line", { x1: x, x2: x, y1: ys(lo(a, s)), y2: ys(hi(a, s)), stroke: color, "stroke-width": 1.5, opacity: 0.55 }, g);
      }
      const mk = marker(g, ARM_SHAPE[a], x, ys(v), color);
      const hit = el("circle", { cx: x, cy: ys(v), r: 14, fill: "transparent" }, g);
      hit.addEventListener("mousemove", (e) => showTip(e, tipFor(a, s)));
      hit.addEventListener("mouseleave", hideTip);
      mk.style.pointerEvents = "none";
    });
    const last = pts[pts.length - 1];
    if (last) endLabels.push({ a, y: ys(last[1]), x: last[0], color });
  });
  // end labels with leader lines when they would collide
  endLabels.sort((p, q) => p.y - q.y);
  for (let i = 1; i < endLabels.length; i++) if (endLabels[i].y - endLabels[i - 1].y < 15) endLabels[i].ly = endLabels[i - 1].ly + 15 || endLabels[i - 1].y + 15;
  endLabels.forEach((p) => {
    const ly = p.ly || p.y;
    if (ly !== p.y) el("line", { x1: p.x + 8, y1: p.y, x2: W - m.r + 6, y2: ly, stroke: css("--grid"), "stroke-width": 1 }, g);
    const t = el("text", { x: W - m.r + 10, y: ly + 4, class: "axis-label" }, g);
    t.textContent = ARM_SHORT[p.a];
  });
}

function outcomeTable(S) {
  const outcomes = ["pass", "wrong_answer", "no_answer", "context_ceiling", "step_limit", "harness_error"];
  let h = `<table><thead><tr><th>Scaffold</th><th>Size</th><th class="num">Accuracy</th><th class="num">95% CI</th>${outcomes.map((o) => `<th class="num">${OUTCOME_LABEL[o]}</th>`).join("")}<th class="num">Mean input tok</th><th class="num">Mean $/run</th><th class="num">$ / pass</th></tr></thead><tbody>`;
  for (const a of S.meta.arms)
    for (const s of S.meta.sizes) {
      const c = S.grid[a][s];
      h += `<tr><td>${ARM_LABEL[a]}</td><td>${s}</td><td class="num">${pct(c.accuracy)}</td><td class="num">${pct(c.ci95[0])}–${pct(c.ci95[1])}</td>${outcomes.map((o) => `<td class="num">${c.outcomes[o]}</td>`).join("")}<td class="num">${fmtInt(c.mean_input_tokens)}</td><td class="num">${usd(c.mean_cost_usd)}</td><td class="num">${usd(c.cost_per_pass_usd)}</td></tr>`;
    }
  document.getElementById("outcome-table").innerHTML = h + "</tbody></table>";
}

function accountsTable(S) {
  let h = `<table><thead><tr><th>Scaffold</th>${S.meta.sizes.map((s) => [1, 2, 3].map((n) => `<th class="num">${s} · ${n} acct</th>`).join("")).join("")}</tr></thead><tbody>`;
  for (const a of S.meta.arms) {
    h += `<tr><td>${ARM_SHORT[a]}</td>`;
    for (const s of S.meta.sizes) for (const n of [1, 2, 3]) {
      const c = S.by_accounts[a][s][n];
      h += `<td class="num" title="${c.passed}/${c.scored}">${pct(c.accuracy)}</td>`;
    }
    h += "</tr>";
  }
  document.getElementById("accounts-table").innerHTML = h + "</tbody></table>";
}

function runsExplorer(S) {
  const f = { arm: document.getElementById("f-arm"), size: document.getElementById("f-size"), task: document.getElementById("f-task"), outcome: document.getElementById("f-outcome") };
  const fill = (sel, vals, lab = (v) => v) => (sel.innerHTML = `<option value="">all</option>` + vals.map((v) => `<option value="${v}">${lab(v)}</option>`).join(""));
  fill(f.arm, S.meta.arms, (a) => ARM_SHORT[a]);
  fill(f.size, S.meta.sizes);
  fill(f.task, S.meta.tasks.map((t) => t.task_id));
  fill(f.outcome, Object.keys(OUTCOME_LABEL), (o) => OUTCOME_LABEL[o]);
  const render = () => {
    const rows = S.runs.filter((r) => (!f.arm.value || r.arm === f.arm.value) && (!f.size.value || r.size === f.size.value) && (!f.task.value || r.task_id === f.task.value) && (!f.outcome.value || r.outcome === f.outcome.value));
    let h = `<table><thead><tr><th>Scaffold</th><th>Size</th><th>Task</th><th class="num">Trial</th><th>Outcome</th><th>Answer</th><th>Gold</th><th class="num">Calls</th><th class="num">Input tok</th><th class="num">Peak request</th><th class="num">Offloaded</th><th>Tools</th><th class="num">$</th></tr></thead><tbody>`;
    for (const r of rows)
      h += `<tr><td>${ARM_SHORT[r.arm]}</td><td>${r.size}</td><td>${r.task_id}</td><td class="num">${r.trial}</td><td><span class="pill">${OUTCOME_LABEL[r.outcome]}</span></td><td>${r.answer ?? "—"}</td><td>${r.gold}</td><td class="num">${r.model_calls}</td><td class="num">${fmtInt(r.input_tokens)}</td><td class="num">${fmtInt(r.attempted_peak_tokens)}</td><td class="num">${r.exports_evicted_in_main_context}</td><td>${Object.entries(r.tool_calls).map(([k, v]) => `${k}×${v}`).join(", ")}</td><td class="num">${r.cost_usd.toFixed(4)}</td></tr>`;
    document.getElementById("runs-table").innerHTML = h + "</tbody></table>";
    document.getElementById("runs-count").textContent = `${rows.length} of ${S.runs.length} runs`;
  };
  Object.values(f).forEach((s) => s.addEventListener("change", render));
  render();
}

function tasksList(S) {
  document.getElementById("tasks").innerHTML = S.meta.tasks.map((t) => `<li><strong>${t.task_id}</strong> (${t.accounts.length} account${t.accounts.length > 1 ? "s" : ""}) — ${t.question} <em>Gold: ${t.gold}</em></li>`).join("");
}

function render(S) {
  document.querySelectorAll("[data-h]").forEach((n) => (n.textContent = S.headline[n.dataset.h]));
  document.querySelectorAll("[data-t]").forEach((n) => (n.textContent = fmtInt(S.totals[n.dataset.t])));
  legend(document.getElementById("legend-acc"), S.meta.arms);
  legend(document.getElementById("legend-peak"), S.meta.arms);
  legend(document.getElementById("legend-cost"), S.meta.arms);
  const cell = (a, s) => S.grid[a][s];
  lineChart("chart-acc", S, {
    value: (a, s) => cell(a, s).accuracy, lo: (a, s) => cell(a, s).ci95[0], hi: (a, s) => cell(a, s).ci95[1],
    yMax: 1, yFmt: (v) => `${Math.round(v * 100)}%`, yTitle: "accuracy (pass / scored runs)",
    tipFor: (a, s) => { const c = cell(a, s); return `<strong>${ARM_LABEL[a]} · ${s}</strong><div class="row">accuracy ${pct(c.accuracy)} (${c.passed}/${c.scored})</div><div class="row">95% CI ${pct(c.ci95[0])}–${pct(c.ci95[1])}</div><div class="row">context ceiling ${c.outcomes.context_ceiling} · wrong ${c.outcomes.wrong_answer} · no answer ${c.outcomes.no_answer}</div>`; },
  });
  const peakMax = Math.max(220000, ...S.meta.arms.flatMap((a) => S.meta.sizes.map((s) => cell(a, s).max_attempted_peak_tokens)));
  lineChart("chart-peak", S, {
    value: (a, s) => (cell(a, s).runs ? cell(a, s).max_attempted_peak_tokens : null), yMax: Math.ceil(peakMax / 50000) * 50000, ticks: Math.ceil(peakMax / 50000),
    yFmt: (v) => fmtK(v), yTitle: "largest request in any run (tokens)",
    refLine: { value: 200000, label: "200k: this account's per-request limit" },
    tipFor: (a, s) => { const c = cell(a, s); return `<strong>${ARM_LABEL[a]} · ${s}</strong><div class="row">largest request ${fmtInt(c.max_attempted_peak_tokens)} tok</div><div class="row">mean of each run's largest ${fmtInt(c.mean_attempted_peak_tokens)} tok</div><div class="row">rejected at the ceiling: ${c.outcomes.context_ceiling}</div>`; },
  });
  const costMax = Math.max(...S.meta.arms.flatMap((a) => S.meta.sizes.map((s) => cell(a, s).mean_cost_usd)));
  lineChart("chart-cost", S, {
    value: (a, s) => (cell(a, s).runs ? cell(a, s).mean_cost_usd : null), yMax: Math.ceil(costMax * 100 * 1.15) / 100,
    yFmt: (v) => `$${v.toFixed(2)}`, yTitle: "mean metered cost per run (USD)",
    tipFor: (a, s) => { const c = cell(a, s); return `<strong>${ARM_LABEL[a]} · ${s}</strong><div class="row">mean ${usd(c.mean_cost_usd)} / run</div><div class="row">${usd(c.cost_per_pass_usd)} per passing run</div><div class="row">${c.mean_model_calls} model calls / run</div>`; },
  });
  outcomeTable(S);
  accountsTable(S);
  runsExplorer(S);
  tasksList(S);
}

let SUMMARY;
fetch("data/summary.json").then((r) => r.json()).then((S) => { SUMMARY = S; render(S); });
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => SUMMARY && render(SUMMARY));
