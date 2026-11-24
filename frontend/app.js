// AML Monitor — vanilla JS, no framework, no build step.
// All DOM is built with createElement/textContent (never innerHTML) so data and LLM text can't inject markup.

const PAGE_SIZE = 50;
const state = {
  items: [],
  total: 0,
  status: "pending",
  minScore: 0,
  selectedId: null,
  detail: null,
  historyTab: "sender",
  detailToken: 0,
  loadingQueue: false,
};

const $ = (sel) => document.querySelector(sel);

// ---------- helpers ----------

/** Create an element: h("div", {class: "x", onclick: fn}, child, "text", ...) */
function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (k === "class") el.className = v;
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

/** fetch wrapper: JSON in/out, throws Error with the API's detail message. */
async function api(path, options = {}) {
  const res = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  let body = null;
  try { body = await res.json(); } catch { /* empty body */ }
  if (!res.ok) {
    const detail = body && body.detail;
    const msg = Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : detail || res.statusText;
    throw new Error(`${res.status}: ${msg}`);
  }
  return body;
}

const fmtMoney = (x) => Number(x).toLocaleString("en-GB", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const fmtPct = (x, d = 1) => `${(Number(x) * 100).toFixed(d)}%`;
const fmtScore = (x) => Number(x).toFixed(4);
const fmtValue = (x) => (Number.isInteger(x) ? x.toLocaleString("en-GB") : Number(x).toLocaleString("en-GB", { maximumFractionDigits: 2 }));
const STATUS_LABEL = { pending: "Pending", approved: "Approved · escalate", dismissed: "Dismissed" };

const badge = (status) => h("span", { class: `badge badge-${status}` }, STATUS_LABEL[status] || status);
const spinner = () => h("span", { class: "spinner", "aria-hidden": "true" });

function toast(message) {
  const t = $("#toast");
  t.textContent = message;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { t.hidden = true; }, 2600);
}

// ---------- metrics bar ----------

async function loadMetrics() {
  const bar = $("#metrics");
  try {
    const m = await api("/api/metrics");
    const set = (key, value) => { bar.querySelector(`[data-metric="${key}"]`).textContent = value; };
    if (m.available) {
      const x = m.model.xgboost;
      set("pr_auc", x.pr_auc.toFixed(4));
      set("recall", fmtPct(x.recall_at_budget));
      set("precision", fmtPct(x.precision_at_budget));
      const note = $("#dataset-note");
      note.textContent = m.model.dataset === "synthetic-demo"
        ? "Demo mode: tiny generated dataset — metrics are meaningless."
        : `Metrics: XGBoost on the held-out test period of ${m.model.dataset} (${m.model.rows.toLocaleString("en-GB")} transactions, synthetic data). ` +
          "Scores are inflated by a dataset artifact — see README limitations. Not production performance.";
      note.hidden = false;
    }
    set("reviewed_today", m.queue.reviewed_today.toLocaleString("en-GB"));
    set("pending", m.queue.pending.toLocaleString("en-GB"));
  } catch (err) {
    bar.title = `Metrics unavailable: ${err.message}`;
  } finally {
    bar.setAttribute("aria-busy", "false");
  }
}

// ---------- queue ----------

function queueUrl(offset) {
  const p = new URLSearchParams({ limit: PAGE_SIZE, offset, min_score: state.minScore });
  if (state.status) p.set("status", state.status);
  return `/api/alerts?${p}`;
}

async function loadQueue({ reset = false } = {}) {
  if (state.loadingQueue) return;
  state.loadingQueue = true;
  const list = $("#alert-list");
  if (reset) {
    state.items = [];
    list.replaceChildren(h("li", { class: "queue-message" }, spinner(), " Loading alerts…"));
  }
  $("#load-more").disabled = true;
  try {
    const page = await api(queueUrl(state.items.length));
    state.items.push(...page.items);
    state.total = page.total;
    renderQueue();
    if (reset && state.items.length && !state.items.some((a) => a.alert_id === state.selectedId)) {
      selectAlert(state.items[0].alert_id, { focusList: false });
    }
  } catch (err) {
    list.replaceChildren(h("li", { class: "queue-message" },
      h("p", {}, `Could not load alerts (${err.message}).`),
      h("button", { type: "button", class: "btn", onclick: () => loadQueue({ reset: true }) }, "Retry")));
  } finally {
    state.loadingQueue = false;
    $("#load-more").disabled = false;
  }
}

function renderQueue() {
  const list = $("#alert-list");
  if (!state.items.length) {
    list.replaceChildren(h("li", { class: "queue-message" }, "No alerts match these filters."));
  } else {
    list.replaceChildren(...state.items.map(alertItem));
  }
  $("#queue-count").textContent = `Showing ${state.items.length.toLocaleString("en-GB")} of ${state.total.toLocaleString("en-GB")} alerts`;
  $("#load-more").hidden = state.items.length >= state.total;
}

function alertItem(a) {
  return h("li", {
    class: "alert-item", role: "option", id: `alert-${a.alert_id}`,
    "aria-selected": String(a.alert_id === state.selectedId),
    dataset: { id: a.alert_id },
    onclick: () => selectAlert(a.alert_id),
  },
  h("span", { class: "amount" }, `${fmtMoney(a.amount)} `, h("span", { class: "muted" }, a.currency)),
  h("span", { class: "score", title: "Model risk score" }, `#${a.rank} · ${fmtScore(a.score)}`),
  h("span", { class: "accounts" }, `${a.src} → ${a.dst}`),
  badge(a.status),
  h("span", { class: "meta" }, `${a.ts} · ${a.payment_format}`));
}

function selectAlert(id, { focusList = true } = {}) {
  state.selectedId = id;
  for (const li of document.querySelectorAll(".alert-item")) {
    li.setAttribute("aria-selected", String(Number(li.dataset.id) === id));
  }
  const el = document.getElementById(`alert-${id}`);
  const list = $("#alert-list");
  if (el) {
    el.scrollIntoView({ block: "nearest" });
    list.setAttribute("aria-activedescendant", el.id);
  }
  if (focusList) list.focus({ preventScroll: true });
  loadDetail(id);
}

async function move(step) {
  if (!state.items.length) return;
  let i = state.items.findIndex((a) => a.alert_id === state.selectedId);
  i = i < 0 ? 0 : i + step;
  if (i >= state.items.length && state.items.length < state.total) {
    await loadQueue();
  }
  i = Math.max(0, Math.min(i, state.items.length - 1));
  selectAlert(state.items[i].alert_id);
}

// ---------- detail ----------

async function loadDetail(id) {
  const token = ++state.detailToken;
  const pane = $("#detail");
  pane.setAttribute("aria-busy", "true");
  pane.replaceChildren(h("div", { class: "card" },
    h("p", {}, spinner(), ` Loading alert ${id}…`),
    ...Array.from({ length: 6 }, () => h("div", { class: "skeleton" }))));
  try {
    const d = await api(`/api/alerts/${id}`);
    if (token !== state.detailToken) return; // a newer selection won
    state.detail = d;
    renderDetail();
  } catch (err) {
    if (token !== state.detailToken) return;
    pane.replaceChildren(h("div", { class: "error-state", role: "alert" },
      h("h2", {}, "Could not load this alert"),
      h("p", {}, err.message),
      h("button", { type: "button", class: "btn", onclick: () => loadDetail(id) }, "Retry")));
  } finally {
    if (token === state.detailToken) pane.setAttribute("aria-busy", "false");
  }
}

function renderDetail() {
  const d = state.detail;
  $("#detail").replaceChildren(
    h("div", { class: "detail-head" },
      h("h2", {}, `Alert ${d.alert_id}`),
      badge(d.status),
      h("span", { class: "score-big" }, `rank #${d.rank.toLocaleString("en-GB")} · score ${fmtScore(d.score)}`)),
    h("div", { class: "grid-2" }, transactionCard(d), shapCard(d)),
    h("div", { class: "grid-2" }, accountCard("Sender", d.sender), accountCard("Receiver", d.receiver)),
    historyCard(d),
    narrativeCard(d),
    decisionCard(d),
  );
}

function facts(pairs) {
  return h("dl", { class: "facts" }, ...pairs.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)]));
}

function transactionCard(d) {
  const currency = d.pay_currency === d.recv_currency ? d.pay_currency : `${d.pay_currency} → ${d.recv_currency}`;
  return h("section", { class: "card", "aria-labelledby": "tx-title" },
    h("h3", { id: "tx-title" }, "Transaction"),
    facts([
      ["Timestamp", d.ts],
      ["Amount", `${fmtMoney(d.amount)} ${d.pay_currency}`],
      ["Currency", currency],
      ["Format", d.payment_format],
      ["Sender", h("span", { class: "mono" }, `${d.src} (${d.src_bank})`)],
      ["Receiver", h("span", { class: "mono" }, `${d.dst} (${d.dst_bank})`)],
    ]));
}

const BOOLEAN_FEATURES = new Set(["cross_currency", "cross_border", "same_account", "is_round_100", "is_round_1000",
  "below_10k_band", "src_in_cycle", "dst_in_cycle"]);

/** Flag features read as yes/no; everything else as a number. */
function displayValue(f) {
  if (f.feature.startsWith("fmt_") || BOOLEAN_FEATURES.has(f.feature)) return f.value >= 0.5 ? "yes" : "no";
  return fmtValue(f.value);
}

function shapCard(d) {
  const max = Math.max(...d.factors.map((f) => f.magnitude), 1e-9);
  const rows = d.factors.map((f) => {
    const width = `${(f.magnitude / max) * 34}%`; // half-track minus room for the number label
    const up = f.shap > 0;
    const bar = h("span", { class: `shap-bar ${up ? "up" : "down"}`, style: `width:${width}` });
    const num = h("span", { class: "shap-num", style: up ? `left:calc(50% + ${width} + 4px)` : `right:calc(50% + ${width} + 4px)` },
      `${f.shap > 0 ? "+" : ""}${f.shap.toFixed(3)}`);
    return h("li", { class: "shap-row" },
      h("span", { class: "shap-label" }, f.label, h("span", { class: "value" }, `value ${displayValue(f)} · ${f.direction}`)),
      h("span", { class: "shap-track", role: "img", "aria-label": `${f.label}: SHAP ${f.shap.toFixed(3)}, ${f.direction}` }, bar, num));
  });
  return h("section", { class: "card", "aria-labelledby": "shap-title" },
    h("h3", { id: "shap-title" }, "Why the model flagged it (top 5 SHAP factors)"),
    h("ul", { class: "shap-list" }, rows),
    h("div", { class: "shap-legend" },
      h("span", {}, h("span", { class: "swatch", style: "background:var(--risk-up)" }), "increases risk"),
      h("span", {}, h("span", { class: "swatch", style: "background:var(--risk-down)" }), "decreases risk"),
      h("span", {}, "units: log-odds")));
}

function accountCard(title, a) {
  return h("section", { class: "card" },
    h("h3", {}, `${title} account`),
    facts([
      ["Account", h("span", { class: "mono" }, `${a.account} (${a.bank})`)],
      ["First seen", a.first_seen || "–"],
      ["Sent", `${a.sent_count.toLocaleString("en-GB")} txns · ${fmtMoney(a.sent_amount)}`],
      ["Received", `${a.received_count.toLocaleString("en-GB")} txns · ${fmtMoney(a.received_amount)}`],
      ["Counterparties", a.unique_counterparties.toLocaleString("en-GB")],
      ["Last 7 days", `${a.last_7d_count.toLocaleString("en-GB")} txns · ${fmtMoney(a.last_7d_amount)}`],
    ]));
}

function historyCard(d) {
  const tab = (key, label) => h("button", {
    type: "button", class: "tab", role: "tab", "aria-selected": String(state.historyTab === key),
    onclick: () => { state.historyTab = key; renderDetail(); },
  }, label);
  const account = state.historyTab === "sender" ? d.src : d.dst;
  const rows = (state.historyTab === "sender" ? d.sender_history : d.receiver_history).slice().reverse();
  const table = h("table", {},
    h("thead", {}, h("tr", {}, h("th", {}, "Timestamp"), h("th", {}, "Direction"), h("th", {}, "Counterparty"),
      h("th", { class: "num" }, "Amount"), h("th", {}, "Format"))),
    h("tbody", {}, rows.map((r) => {
      const out = r.src === account;
      return h("tr", { class: r.txn_id === d.alert_id ? "is-alert" : "" },
        h("td", {}, r.timestamp),
        h("td", { class: out ? "dir-out" : "dir-in" }, out ? "↑ sent" : "↓ received"),
        h("td", { class: "mono" }, out ? r.dst : r.src),
        h("td", { class: "num" }, fmtMoney(r.amount)),
        h("td", {}, r.payment_format + (r.txn_id === d.alert_id ? " (this alert)" : "")));
    })));
  return h("section", { class: "card", "aria-labelledby": "hist-title" },
    h("h3", { id: "hist-title" }, "Account history (up to alert time, latest 20)"),
    h("div", { class: "tabs", role: "tablist" }, tab("sender", `Sender ${d.src}`), tab("receiver", `Receiver ${d.dst}`)),
    rows.length ? h("div", { class: "table-wrap" }, table) : h("p", { class: "muted" }, "No history."));
}

function provenance(n) {
  const v = n.validation;
  let label = "validated";
  let cls = "badge-ok";
  if (v.fell_back) { label = "LLM drafts rejected → template"; cls = "badge-warn"; }
  else if (n.provider === "template") { label = "template (no LLM key)"; }
  else if (v.attempts > 1) { label = `validated after ${v.attempts} attempts`; cls = "badge-warn"; }
  return h("span", { class: "provenance" },
    h("span", { class: `badge ${cls}` }, v.passed ? `✓ ${label}` : `✗ ${label}`),
    ` ${n.provider}${n.model ? ` · ${n.model}` : ""} · ${n.created_at.replace("T", " ").replace("+00:00", " UTC")}`);
}

function narrativeCard(d) {
  const n = d.narrative;
  const btn = h("button", { type: "button", class: "btn btn-primary", id: "gen-btn", onclick: generateNarrative },
    n ? "Regenerate draft" : "Generate draft");
  const issues = n && (n.validation.llm_issues.length || n.validation.issues.length)
    ? h("details", { class: "issues" },
      h("summary", {}, "Validator findings"),
      h("ul", {}, [...n.validation.llm_issues, ...n.validation.issues].map((i) => h("li", {}, i))))
    : null;
  const text = (d.decision && d.decision.narrative_text) || (n ? n.text : "");
  return h("section", { class: "card", "aria-labelledby": "narr-title" },
    h("div", { class: "narrative-head" }, h("h3", { id: "narr-title" }, "SAR narrative draft"), n ? provenance(n) : null, btn),
    issues,
    h("label", { for: "narrative-text", class: "muted" }, "Editable draft — your edits are saved with your decision"),
    h("textarea", { id: "narrative-text", spellcheck: "true", placeholder: "No draft yet. Click “Generate draft”." }, text),
    h("p", { class: "boundary" }, "The LLM only drafts text from the facts above; every number, account, date and time is checked against them. It never makes the decision — you do."));
}

function decisionCard(d) {
  const last = d.decision
    ? h("p", { class: "last-decision" }, "Last decision: ", badge(d.decision.status), ` on ${d.decision.created_at.replace("T", " ").replace("+00:00", " UTC")} — “${d.decision.note}”`)
    : null;
  return h("section", { class: "card", "aria-labelledby": "dec-title" },
    h("h3", { id: "dec-title" }, "Analyst decision"),
    last,
    h("form", { class: "decision-form", onsubmit: (e) => e.preventDefault() },
      h("label", { for: "decision-note" }, "Note (required — document your reasoning)"),
      h("textarea", { id: "decision-note", rows: "3", maxlength: "2000", required: true, "aria-describedby": "note-error" }),
      h("p", { id: "note-error", class: "issues", role: "alert", hidden: true }, "Please add a note before deciding."),
      h("div", { class: "decision-actions" },
        h("button", { type: "button", class: "btn btn-escalate", onclick: () => submitDecision("approve") }, "Approve · escalate for SAR"),
        h("button", { type: "button", class: "btn", onclick: () => submitDecision("dismiss") }, "Dismiss · false positive"))));
}

// ---------- actions ----------

async function generateNarrative() {
  const d = state.detail;
  const btn = $("#gen-btn");
  btn.disabled = true;
  btn.replaceChildren(spinner(), " Drafting…");
  try {
    const n = await api(`/api/alerts/${d.alert_id}/narrative`, { method: "POST", body: "{}" });
    if (state.detail && state.detail.alert_id === d.alert_id) {
      state.detail.narrative = n;
      if (state.detail.decision) state.detail.decision = { ...state.detail.decision, narrative_text: null };
      renderDetail();
      toast(n.validation.fell_back ? "LLM draft failed validation — template used" : "Draft ready");
    }
  } catch (err) {
    toast(`Draft failed: ${err.message}`);
    btn.disabled = false;
    btn.textContent = "Retry draft";
  }
}

async function submitDecision(decision) {
  const d = state.detail;
  const note = $("#decision-note");
  const err = $("#note-error");
  if (!note.value.trim()) {
    err.hidden = false;
    note.focus();
    return;
  }
  err.hidden = true;
  const buttons = document.querySelectorAll(".decision-actions .btn");
  buttons.forEach((b) => { b.disabled = true; });
  try {
    const narrative = $("#narrative-text").value.trim();
    const saved = await api(`/api/alerts/${d.alert_id}/decision`, {
      method: "POST",
      body: JSON.stringify({ decision, note: note.value.trim(), narrative_text: narrative || null }),
    });
    toast(`Alert ${d.alert_id} ${saved.status}`);
    loadMetrics();
    const idx = state.items.findIndex((a) => a.alert_id === d.alert_id);
    if (state.status && state.status !== saved.status && idx >= 0) {
      // No longer matches the filter: drop it and advance to the next alert.
      state.items.splice(idx, 1);
      state.total -= 1;
      renderQueue();
      const next = state.items[Math.min(idx, state.items.length - 1)];
      if (next) selectAlert(next.alert_id);
      else $("#detail").replaceChildren(h("div", { class: "empty-state" }, h("h2", {}, "Queue cleared"), h("p", {}, "No more alerts match these filters.")));
    } else {
      if (idx >= 0) state.items[idx].status = saved.status;
      renderQueue();
      state.detail = { ...d, status: saved.status, decision: saved };
      renderDetail();
    }
  } catch (e) {
    toast(`Could not save decision: ${e.message}`);
    buttons.forEach((b) => { b.disabled = false; });
  }
}

// ---------- events ----------

function isTyping(el) {
  return el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.isContentEditable);
}

document.addEventListener("keydown", (e) => {
  if (e.altKey || e.ctrlKey || e.metaKey || isTyping(e.target)) return;
  const inList = e.target === $("#alert-list");
  if (e.key === "j" || (inList && e.key === "ArrowDown")) { e.preventDefault(); move(1); }
  else if (e.key === "k" || (inList && e.key === "ArrowUp")) { e.preventDefault(); move(-1); }
});

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

function init() {
  $("#filter-status").addEventListener("change", (e) => { state.status = e.target.value; loadQueue({ reset: true }); });
  $("#filter-score").addEventListener("input", debounce((e) => {
    const v = Math.min(1, Math.max(0, Number(e.target.value) || 0));
    state.minScore = v;
    loadQueue({ reset: true });
  }, 300));
  $("#load-more").addEventListener("click", () => loadQueue());
  loadMetrics();
  loadQueue({ reset: true });
}

init();
