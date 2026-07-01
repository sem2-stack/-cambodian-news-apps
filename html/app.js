"use strict";

/* ------------------------------------------------------------------ *
 * Front-end for the Cambodian News Classifier.
 *
 * Classification is performed by the real fine-tuned checkpoints through
 * the backend API (see app/server.py). Model metadata (labels, available
 * models, accuracy/F1) is fetched from /api/meta on load.
 * ------------------------------------------------------------------ */

const API = {
  meta: "/api/meta",
  classify: "/api/classify",
  extractPdf: "/api/extract-pdf",
};

const CATEGORY_COLORS = {
  politics: "#7c3aed",
  technology: "#10b981",
  economics: "#3b82f6",
  health: "#0ea5e9",
  sports: "#f59e0b",
  environment: "#16a34a",
};
const DEFAULT_COLOR = "#64748b";

// Sensible defaults; overwritten by /api/meta once it resolves.
let MIN_WORDS = 50;
let LABELS = ["economics", "health", "politics", "sports", "technology"];

let MODEL_INFO = {
  distilbert: { display: "DistilBERT", accuracy: 0.9515, macro_f1: 0.9476, available: false },
  roberta: { display: "RoBERTa", accuracy: 0.9432, macro_f1: 0.9384, available: false },
  electra: { display: "ELECTRA", accuracy: 0.9297, macro_f1: 0.9233, available: false },
  bert: { display: "BERT", accuracy: 0.8893, macro_f1: 0.8725, available: false },
};
let DEFAULT_MODEL = "distilbert";

const STORAGE_KEY = "cnc_history";

const state = {
  page: "classifier",
  history: loadHistory(),
  lastResult: null,
  modelKey: DEFAULT_MODEL,
};

/* ----------------------------- helpers ----------------------------- */
function $(id) {
  return document.getElementById(id);
}

function colorFor(cat) {
  return CATEGORY_COLORS[cat] || DEFAULT_COLOR;
}

function wordCount(text) {
  const t = text.trim();
  return t ? t.split(/\s+/).length : 0;
}

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) || [];
  } catch (_) {
    return [];
  }
}

function saveHistory() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state.history));
  } catch (_) {
    /* storage unavailable — keep in-memory only */
  }
}

/* ------------------------- classification -------------------------- */
async function classifyRemote(text, model) {
  const res = await fetch(API.classify, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, model }),
  });
  if (!res.ok) {
    let detail = "Classification failed.";
    try {
      detail = (await res.json()).error || detail;
    } catch (_) {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json();
}

async function loadMeta() {
  try {
    const res = await fetch(API.meta);
    if (!res.ok) return;
    const meta = await res.json();
    LABELS = meta.labels && meta.labels.length ? meta.labels : LABELS;
    MIN_WORDS = meta.min_words || MIN_WORDS;
    if (meta.default_model) {
      DEFAULT_MODEL = meta.default_model;
      state.modelKey = DEFAULT_MODEL;
    }
    if (Array.isArray(meta.models)) {
      const next = {};
      for (const m of meta.models) {
        next[m.key] = {
          display: m.display,
          accuracy: m.accuracy,
          macro_f1: m.macro_f1,
          available: m.available,
        };
      }
      MODEL_INFO = next;
    }
  } catch (_) {
    /* backend unreachable — keep defaults */
  }
}

/* ----------------------------- routing ----------------------------- */
function setPage(page) {
  state.page = page;
  ["classifier", "history", "about"].forEach((p) => {
    $("page-" + p).classList.toggle("hidden", p !== page);
  });
  document.querySelectorAll(".nav-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.page === page);
  });
  if (page === "history") renderHistory();
}

/* -------------------------- input handling ------------------------- */
function updateCounts() {
  const text = $("input-text").value;
  $("char-count").textContent = text.length.toLocaleString();
  $("word-count").textContent = wordCount(text).toLocaleString();
  $("analyze-btn").disabled = wordCount(text) === 0;
}

/* ----------------------------- tabs -------------------------------- */
function setTab(tab) {
  ["text", "pdf"].forEach((t) => {
    $("tab-" + t).classList.toggle("hidden", t !== tab);
  });
  document.querySelectorAll(".tab").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === tab);
  });
}

/* -------------------------- PDF upload ----------------------------- */
async function handlePdf(file) {
  if (!file) return;
  const status = $("pdf-status");
  status.className = "pdf-status loading";
  status.textContent = "Extracting text from “" + file.name + "”…";

  try {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch(API.extractPdf, { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Extraction failed.");

    $("input-text").value = data.text;
    updateCounts();
    status.className = "pdf-status ok";
    status.textContent =
      "\u2713 Extracted " +
      (data.words || wordCount(data.text)).toLocaleString() +
      " words. Switched to the text editor — review and click Analyze Text.";
    setTab("text");
  } catch (err) {
    status.className = "pdf-status err";
    status.textContent = "\u26A0 " + (err.message || "Could not read PDF.");
  } finally {
    $("pdf-file").value = "";
  }
}

/* --------------------------- rendering ----------------------------- */
function renderScores(scores, container) {
  const ordered = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  container.innerHTML = "";
  for (const [cat, prob] of ordered) {
    const pct = (prob * 100).toFixed(1);
    const color = colorFor(cat);
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML =
      `<div class="bar-name">${cat}</div>` +
      `<div class="bar-track"><div class="bar-fill" style="background:${color};"></div></div>` +
      `<div class="bar-val" style="color:${color};">${pct}%</div>`;
    container.appendChild(row);
    // Animate width on next frame.
    const fill = row.querySelector(".bar-fill");
    requestAnimationFrame(() => {
      fill.style.width = pct + "%";
    });
  }
}

function renderResult(result) {
  $("result-empty").classList.add("hidden");
  const errBox = $("result-error");
  if (errBox) errBox.classList.add("hidden");
  $("result-body").classList.remove("hidden");

  const cat = result.category;
  const conf = (result.confidence * 100).toFixed(1);
  const color = colorFor(cat);

  $("result-cat").textContent = cat;
  $("result-cat").style.color = color;
  $("conf-pill").textContent = conf + "% confidence";

  $("stat-chars").textContent = result.chars.toLocaleString();
  $("stat-words").textContent = result.words.toLocaleString();

  const note = $("length-note");
  if (result.words >= MIN_WORDS) {
    note.className = "ok-note";
    note.innerHTML = "\u2713 Text length is optimal for classification";
  } else {
    note.className = "warn-note";
    note.innerHTML = "\u26A0 Short text \u2014 prediction may be less reliable";
  }

  renderScores(result.scores, $("scores-list"));
  $("model-caption").textContent = "Model: " + result.model;
}

/* --------------------------- analyze ------------------------------- */
async function analyze() {
  const text = $("input-text").value;
  const words = wordCount(text);
  if (words === 0) return;

  const btn = $("analyze-btn");
  const original = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Analyzing…';

  try {
    const data = await classifyRemote(text, state.modelKey);
    const result = {
      timestamp: new Date().toISOString().slice(0, 19),
      model: data.model || MODEL_INFO[state.modelKey].display,
      category: data.category,
      confidence: data.confidence,
      scores: data.scores,
      chars: text.length,
      words,
      preview: text.trim().replace(/\s+/g, " ").slice(0, 160),
    };

    state.lastResult = result;
    state.history.unshift(result);
    saveHistory();
    renderResult(result);
  } catch (err) {
    renderError(err.message || "Classification failed.");
  } finally {
    btn.innerHTML = original;
    btn.disabled = wordCount($("input-text").value) === 0;
  }
}

function renderError(message) {
  $("result-empty").classList.add("hidden");
  $("result-body").classList.add("hidden");
  let box = $("result-error");
  if (!box) {
    box = document.createElement("div");
    box.id = "result-error";
    box.className = "error-box";
    $("result-card").appendChild(box);
  }
  box.classList.remove("hidden");
  box.textContent = "⚠ " + message;
}

/* ------------------------- export helpers -------------------------- */
function download(filename, content, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function exportResult() {
  if (!state.lastResult) return;
  const r = state.lastResult;
  download(
    `classification_${r.timestamp.replace(/:/g, "-")}.json`,
    JSON.stringify(r, null, 2),
    "application/json"
  );
}

function exportCsv() {
  const headers = ["timestamp", "model", "category", "confidence", "words", "chars", "preview"];
  const rows = state.history.map((h) =>
    [
      h.timestamp,
      h.model,
      h.category,
      h.confidence.toFixed(4),
      h.words,
      h.chars,
      `"${h.preview.replace(/"/g, '""')}"`,
    ].join(",")
  );
  download("session_history.csv", [headers.join(","), ...rows].join("\n"), "text/csv");
}

/* --------------------------- history ------------------------------- */
function renderHistory() {
  const history = state.history;
  const hasItems = history.length > 0;
  $("history-empty").classList.toggle("hidden", hasItems);
  $("history-content").classList.toggle("hidden", !hasItems);
  if (!hasItems) return;

  const confidences = history.map((h) => h.confidence);
  const cats = history.map((h) => h.category);
  const avg = (confidences.reduce((a, b) => a + b, 0) / confidences.length) * 100;
  const topCat = cats
    .sort((a, b) => cats.filter((c) => c === a).length - cats.filter((c) => c === b).length)
    .pop();

  $("m-articles").textContent = history.length;
  $("m-cats").textContent = new Set(cats).size;
  $("m-avg").textContent = avg.toFixed(1) + "%";
  $("m-top").textContent = topCat.charAt(0).toUpperCase() + topCat.slice(1);

  // Category filter options.
  const sel = $("history-cat");
  const current = sel.value || "All";
  const uniqueCats = ["All", ...Array.from(new Set(cats)).sort()];
  sel.innerHTML = uniqueCats
    .map((c) => `<option value="${c}">${c === "All" ? "All categories" : c}</option>`)
    .join("");
  sel.value = uniqueCats.includes(current) ? current : "All";

  renderHistoryList();
}

function renderHistoryList() {
  const query = $("history-search").value.toLowerCase();
  const catFilter = $("history-cat").value;
  const list = $("history-list");
  list.innerHTML = "";

  for (const h of state.history) {
    if (query && !h.preview.toLowerCase().includes(query)) continue;
    if (catFilter !== "All" && h.category !== catFilter) continue;

    const color = colorFor(h.category);
    const item = document.createElement("div");
    item.className = "history-item";
    item.innerHTML =
      `<span class="badge" style="background:${color};">${h.category}</span>` +
      `<span class="history-meta">${(h.confidence * 100).toFixed(1)}% &middot; ${h.model} &middot; ${h.timestamp}</span>` +
      `<div class="history-preview">${escapeHtml(h.preview)}&hellip;</div>`;
    list.appendChild(item);
  }
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

/* ----------------------------- about ------------------------------- */
function renderAbout() {
  $("about-categories").textContent = LABELS.map(
    (c) => c.charAt(0).toUpperCase() + c.slice(1)
  ).join(", ");

  const body = $("model-table-body");
  body.innerHTML = Object.values(MODEL_INFO)
    .map(
      (info) =>
        `<tr><td>${info.display}</td>` +
        `<td>${(info.accuracy * 100).toFixed(2)}%</td>` +
        `<td>${(info.macro_f1 * 100).toFixed(2)}%</td>` +
        `<td>${info.available ? "&#10003;" : "&#10007;"}</td></tr>`
    )
    .join("");
}

/* ------------------------- model selector -------------------------- */
function initModelSelect() {
  const sel = $("model-select");
  const entries = Object.entries(MODEL_INFO);
  const availableEntries = entries.filter(([, info]) => info.available);
  const usable = availableEntries.length ? availableEntries : entries;

  sel.innerHTML = usable
    .map(
      ([key, info]) =>
        `<option value="${key}">${info.display}  \u00b7  Acc ${(info.accuracy * 100).toFixed(
          1
        )}% / F1 ${(info.macro_f1 * 100).toFixed(1)}%</option>`
    )
    .join("");

  if (!MODEL_INFO[state.modelKey] || !MODEL_INFO[state.modelKey].available) {
    state.modelKey = usable[0] ? usable[0][0] : state.modelKey;
  }
  sel.value = state.modelKey;
  sel.addEventListener("change", () => {
    state.modelKey = sel.value;
  });
}

/* ------------------------------ init ------------------------------- */
async function init() {
  await loadMeta();

  // Navigation.
  document.querySelectorAll(".nav-btn").forEach((b) => {
    b.addEventListener("click", () => setPage(b.dataset.page));
  });

  // Input tabs.
  document.querySelectorAll(".tab").forEach((b) => {
    b.addEventListener("click", () => setTab(b.dataset.tab));
  });

  // Input.
  $("input-text").addEventListener("input", updateCounts);
  $("clear-input").addEventListener("click", () => {
    $("input-text").value = "";
    updateCounts();
  });
  $("analyze-btn").addEventListener("click", analyze);

  // PDF upload.
  $("pdf-file").addEventListener("change", (e) => handlePdf(e.target.files[0]));

  // Result actions.
  $("export-btn").addEventListener("click", exportResult);
  $("clear-result-btn").addEventListener("click", () => {
    state.lastResult = null;
    $("input-text").value = "";
    updateCounts();
    $("result-body").classList.add("hidden");
    const errBox = $("result-error");
    if (errBox) errBox.classList.add("hidden");
    $("result-empty").classList.remove("hidden");
  });

  // History actions.
  $("history-search").addEventListener("input", renderHistoryList);
  $("history-cat").addEventListener("change", renderHistoryList);
  $("export-csv-btn").addEventListener("click", exportCsv);
  $("clear-history-btn").addEventListener("click", () => {
    state.history = [];
    saveHistory();
    renderHistory();
  });

  initModelSelect();
  renderAbout();
  updateCounts();
}

document.addEventListener("DOMContentLoaded", init);
