"""Cambodian News Classifier - Streamlit dashboard.

Run from the project root:

    streamlit run app/streamlit_app.py

Serves the four fine-tuned ``full_no_environment`` encoders (BERT, DistilBERT,
RoBERTa, ELECTRA) behind a single classifier UI with session history and an
about/model-card page.
"""

from __future__ import annotations

import io
import json
from datetime import datetime

import pandas as pd
import streamlit as st

from inference.predictor import (
    DEFAULT_MODEL,
    MODEL_INFO,
    available_models,
    classify,
    get_labels,
)

# --------------------------------------------------------------------------- #
# Page config + theme
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Cambodian News Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Per-category accent colours (consistent across the whole app).
CATEGORY_COLORS = {
    "politics": "#7c3aed",
    "technology": "#10b981",
    "economics": "#3b82f6",
    "health": "#0ea5e9",
    "sports": "#f59e0b",
}
DEFAULT_COLOR = "#64748b"

MIN_WORDS = 50

CSS = """
<style>
    /* hide default chrome */
    #MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden;}
    .block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1180px;}

    .app-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 14px;
        padding: 18px 26px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 22px;
    }
    .brand {display: flex; align-items: center; gap: 14px;}
    .brand-logo {
        width: 42px; height: 42px; border-radius: 11px;
        background: linear-gradient(135deg,#6366f1,#06b6d4);
        display:flex; align-items:center; justify-content:center;
        font-size: 20px; color:white; font-weight:700;
    }
    .brand-title {color:#f8fafc; font-size:20px; font-weight:700; line-height:1.1;}
    .brand-sub {color:#94a3b8; font-size:11px; letter-spacing:2px; font-weight:600;}

    .card {
        background:#ffffff; border:1px solid #e5e7eb; border-radius:14px;
        padding:22px 24px; box-shadow:0 1px 3px rgba(0,0,0,0.04);
    }
    .card-title {font-size:20px; font-weight:700; color:#111827; margin:0;}
    .card-sub {font-size:13px; color:#6b7280; margin-top:2px;}

    .result-head {
        background:linear-gradient(135deg,#f5f3ff,#faf5ff);
        border:1px solid #ede9fe; border-radius:14px; padding:20px 24px;
        margin-bottom:16px;
    }
    .result-kicker {font-size:11px; letter-spacing:1.5px; color:#6b7280; font-weight:700;}
    .result-cat {font-size:34px; font-weight:800; margin:2px 0 0 0; text-transform:uppercase;}
    .conf-pill {
        float:right; background:#ede9fe; color:#6d28d9; font-weight:700;
        font-size:12px; padding:5px 12px; border-radius:999px;
    }

    .stat-box {
        background:#f8fafc; border:1px solid #eef2f7; border-radius:12px;
        padding:14px; text-align:center;
    }
    .stat-num {font-size:26px; font-weight:800; color:#111827;}
    .stat-lab {font-size:12px; color:#6b7280; margin-top:2px;}

    .bar-row {display:flex; align-items:center; margin:9px 0; font-size:13px;}
    .bar-name {width:110px; color:#374151; text-transform:capitalize;}
    .bar-track {flex:1; background:#eef2f7; border-radius:6px; height:9px; overflow:hidden; margin:0 12px;}
    .bar-fill {height:100%; border-radius:6px;}
    .bar-val {width:54px; text-align:right; font-weight:700;}

    .ok-note {color:#059669; font-size:13px; font-weight:600;}
    .warn-note {color:#d97706; font-size:13px; font-weight:600;}

    .badge {
        display:inline-block; padding:3px 10px; border-radius:999px;
        font-size:11px; font-weight:700; color:white; text-transform:capitalize;
    }

    div.stButton > button {border-radius:10px; font-weight:600;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def _init_state() -> None:
    st.session_state.setdefault("page", "Classifier")
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("input_text", "")
    st.session_state.setdefault("model_key", DEFAULT_MODEL)


_init_state()


def _color(cat: str) -> str:
    return CATEGORY_COLORS.get(cat, DEFAULT_COLOR)


def html_block(html: str) -> None:
    """Render raw HTML, collapsing per-line indentation.

    Indented multi-line strings are otherwise treated as markdown code blocks
    and shown verbatim, so we strip leading whitespace from every line first.
    """
    cleaned = "".join(line.strip() for line in html.splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Header / navigation
# --------------------------------------------------------------------------- #
def render_header() -> None:
    html_block(
        """
        <div class="app-header">
          <div class="brand">
            <div class="brand-logo">📰</div>
            <div>
              <div class="brand-title">Cambodian News Classifier</div>
              <div class="brand-sub">MULTI-CLASS AI ANALYSIS</div>
            </div>
          </div>
        </div>
        """
    )
    nav = st.columns([1, 1, 1, 4])
    pages = ["Classifier", "Session History", "About"]
    for col, name in zip(nav, pages):
        with col:
            kind = "primary" if st.session_state.page == name else "secondary"
            if st.button(name, key=f"nav_{name}", use_container_width=True, type=kind):
                st.session_state.page = name
                st.rerun()


# --------------------------------------------------------------------------- #
# Confidence bars
# --------------------------------------------------------------------------- #
def render_scores(scores: dict[str, float]) -> None:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    parts = [
        '<div style="font-weight:700;color:#374151;margin:6px 0 4px;">'
        "📊 Confidence Scores</div>"
    ]
    for cat, prob in ordered:
        pct = prob * 100
        color = _color(cat)
        parts.append(
            '<div class="bar-row">'
            f'<div class="bar-name">{cat}</div>'
            '<div class="bar-track">'
            f'<div class="bar-fill" style="width:{pct:.1f}%;background:{color};"></div>'
            "</div>"
            f'<div class="bar-val" style="color:{color};">{pct:.1f}%</div>'
            "</div>"
        )
    st.markdown("".join(parts), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Classifier page
# --------------------------------------------------------------------------- #
def page_classifier() -> None:
    left, right = st.columns([1, 1], gap="large")

    with left:
        text = st.session_state.input_text
        chars, words = len(text), len(text.split())
        html_block(
            f"""
            <div style="display:flex;justify-content:space-between;align-items:flex-end;">
              <div>
                <p class="card-title">Input Section</p>
                <p class="card-sub">Paste news text for classification</p>
              </div>
              <div style="color:#6b7280;font-size:13px;">{chars:,} chars &nbsp;|&nbsp; {words:,} words</div>
            </div>
            """
        )

        model_selector()

        tab_text, tab_pdf = st.tabs(["Direct Text Entry", "PDF Upload"])
        with tab_text:
            text = st.text_area(
                "Article body",
                value=st.session_state.input_text,
                height=320,
                label_visibility="collapsed",
                placeholder="Paste the news article body here (English). "
                "At least 50 words is recommended for a reliable classification.",
                key="text_area_input",
            )
            st.session_state.input_text = text

        with tab_pdf:
            pdf = st.file_uploader("Upload a PDF article", type=["pdf"])
            if pdf is not None:
                extracted = _read_pdf(pdf)
                if extracted:
                    st.session_state.input_text = extracted
                    text = extracted
                    st.success(f"Extracted {len(extracted.split()):,} words from PDF.")
                    st.text_area("Extracted text", value=extracted, height=180)

        words = len(text.split())
        analyze = st.button(
            "✨ Analyze Text",
            type="primary",
            use_container_width=True,
            disabled=words == 0,
        )
        if analyze:
            if words < MIN_WORDS:
                st.warning(
                    f"Only {words} words detected. {MIN_WORDS}+ words give more "
                    "reliable results, but classifying anyway."
                )
            with st.spinner("Running inference…"):
                scores = classify(text, st.session_state.model_key)
            top_cat = max(scores, key=scores.get)
            result = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": MODEL_INFO[st.session_state.model_key]["display"],
                "category": top_cat,
                "confidence": scores[top_cat],
                "scores": scores,
                "chars": len(text),
                "words": words,
                "preview": text.strip().replace("\n", " ")[:160],
            }
            st.session_state.last_result = result
            st.session_state.history.insert(0, result)

    with right:
        result = st.session_state.last_result
        if result is None:
            st.markdown(
                '<div class="card" style="text-align:center;color:#9ca3af;padding:60px 24px;">'
                "🔎<br><br>Enter article text and click <b>Analyze Text</b> to see the "
                "predicted category and confidence scores.</div>",
                unsafe_allow_html=True,
            )
            return

        cat = result["category"]
        conf = result["confidence"] * 100
        html_block(
            f"""
            <div class="result-head">
              <span class="conf-pill">{conf:.1f}% confidence</span>
              <div class="result-kicker">🏆 TOP CLASSIFICATION</div>
              <div class="result-cat" style="color:{_color(cat)};">{cat}</div>
            </div>
            """
        )

        c1, c2 = st.columns(2)
        c1.markdown(
            f'<div class="stat-box"><div class="stat-num">{result["chars"]:,}</div>'
            '<div class="stat-lab">Characters</div></div>',
            unsafe_allow_html=True,
        )
        c2.markdown(
            f'<div class="stat-box"><div class="stat-num">{result["words"]:,}</div>'
            '<div class="stat-lab">Words</div></div>',
            unsafe_allow_html=True,
        )

        if result["words"] >= MIN_WORDS:
            st.markdown(
                '<p class="ok-note">✓ Text length is optimal for classification</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p class="warn-note">⚠ Short text — prediction may be less reliable</p>',
                unsafe_allow_html=True,
            )

        render_scores(result["scores"])
        st.caption(f"Model: {result['model']}")

        e1, e2 = st.columns(2)
        with e1:
            st.download_button(
                "⬇ Export",
                data=json.dumps(result, indent=2),
                file_name=f"classification_{result['timestamp'].replace(':', '-')}.json",
                mime="application/json",
                use_container_width=True,
            )
        with e2:
            if st.button("🗑 Clear", use_container_width=True):
                st.session_state.last_result = None
                st.session_state.input_text = ""
                st.rerun()


def _read_pdf(uploaded) -> str:
    try:
        import pdfplumber
    except ImportError:
        st.error("PDF support needs `pdfplumber`. Install it with: pip install pdfplumber")
        return ""
    try:
        text_parts: list[str] = []
        with pdfplumber.open(io.BytesIO(uploaded.read())) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts).strip()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read PDF: {exc}")
        return ""


# --------------------------------------------------------------------------- #
# Session history page
# --------------------------------------------------------------------------- #
def page_history() -> None:
    history = st.session_state.history
    st.markdown('<p class="card-title">Session History</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="card-sub">Classifications from this browser session (resets on refresh).</p>',
        unsafe_allow_html=True,
    )

    if not history:
        st.info("No classifications yet. Analyze an article to populate the history.")
        return

    confidences = [h["confidence"] for h in history]
    cats = [h["category"] for h in history]
    top_cat = max(set(cats), key=cats.count)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Articles", len(history))
    m2.metric("Categories used", len(set(cats)))
    m3.metric("Avg confidence", f"{(sum(confidences) / len(confidences)) * 100:.1f}%")
    m4.metric("Top category", top_cat.capitalize())

    st.markdown("---")
    f1, f2 = st.columns([3, 1])
    query = f1.text_input("Search text", placeholder="Filter by preview text…")
    cat_filter = f2.selectbox("Category", ["All"] + sorted(set(cats)))

    rows = []
    for h in history:
        if query and query.lower() not in h["preview"].lower():
            continue
        if cat_filter != "All" and h["category"] != cat_filter:
            continue
        rows.append(h)

    for h in rows:
        color = _color(h["category"])
        html_block(
            f"""
            <div class="card" style="margin-bottom:10px;padding:14px 18px;">
              <span class="badge" style="background:{color};">{h['category']}</span>
              <span style="float:right;color:#6b7280;font-size:12px;">
                {h['confidence']*100:.1f}% &middot; {h['model']} &middot; {h['timestamp']}
              </span>
              <div style="margin-top:8px;color:#374151;font-size:14px;">{h['preview']}&hellip;</div>
            </div>
            """
        )

    st.markdown("---")
    df = pd.DataFrame(
        [
            {
                "timestamp": h["timestamp"],
                "model": h["model"],
                "category": h["category"],
                "confidence": round(h["confidence"], 4),
                "words": h["words"],
                "chars": h["chars"],
                "preview": h["preview"],
            }
            for h in history
        ]
    )
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇ Export All (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="session_history.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        if st.button("🗑 Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()


# --------------------------------------------------------------------------- #
# About page
# --------------------------------------------------------------------------- #
def page_about() -> None:
    labels = get_labels()
    st.markdown('<p class="card-title">About</p>', unsafe_allow_html=True)
    st.markdown(
        """
        This dashboard classifies English-language Cambodian news articles into one of
        five categories using transformer encoders fine-tuned on a custom corpus scraped
        from Cambodian news outlets. It is the deployment deliverable (Part 3) of the
        thesis project.
        """
    )

    st.subheader("Categories")
    st.write(", ".join(c.capitalize() for c in labels))
    st.caption(
        "This is the *no-environment* variant — the Environment class was excluded "
        "from the corpus, leaving five balanced-enough categories."
    )

    st.subheader("Model card — test-set performance")
    rank = pd.DataFrame(
        [
            {
                "Model": info["display"],
                "Accuracy": f"{info['accuracy']*100:.2f}%",
                "Macro F1": f"{info['macro_f1']*100:.2f}%",
                "Available": "✓" if key in available_models() else "✗",
            }
            for key, info in MODEL_INFO.items()
        ]
    )
    st.table(rank)
    st.success(
        "**DistilBERT** is the recommended default — best macro-F1 (94.76%) and the "
        "smallest checkpoint, comfortably under the 5-second latency target."
    )

    st.subheader("Pipeline")
    st.markdown(
        """
        1. **Preprocess** — lowercase, strip HTML / URLs / emails / digits, drop
           stop-words (identical to training, via `preprocessing.clean.preprocess`).
        2. **Tokenize** — model-specific HuggingFace tokenizer, `max_length=512`,
           body text only.
        3. **Classify** — `TransformerClassifier` ([CLS] → 512 → LogSoftmax over 5 classes).
        4. **Report** — `exp()` of log-probabilities gives the confidence scores shown.
        """
    )

    st.subheader("Known limitations")
    st.markdown(
        """
        - Trained only on a handful of Cambodian English-language outlets; may
          underperform on other regions or styles.
        - English only — Khmer-language or heavily code-switched text is out of scope.
        - History is per-session and clears on browser refresh.
        """
    )


# --------------------------------------------------------------------------- #
# Model selector (on-page) + sidebar + routing
# --------------------------------------------------------------------------- #
def model_selector() -> None:
    """On-page transformer picker bound to ``st.session_state.model_key``."""
    models = available_models()
    if not models:
        st.error("No checkpoints found in models/checkpoints/full_no_environment/")
        return
    if st.session_state.model_key not in models:
        st.session_state.model_key = models[0]
    default_idx = models.index(st.session_state.model_key)
    choice = st.selectbox(
        "🤖 Classification model",
        models,
        index=default_idx,
        format_func=lambda k: (
            f"{MODEL_INFO[k]['display']}  ·  "
            f"Acc {MODEL_INFO[k]['accuracy']*100:.1f}% / F1 {MODEL_INFO[k]['macro_f1']*100:.1f}%"
        ),
        key="model_select",
        help="Pick which fine-tuned encoder runs the classification. "
        "DistilBERT is the best performer.",
    )
    st.session_state.model_key = choice


def render_sidebar() -> None:
    with st.sidebar:
        st.header("⚙️ Settings")
        current = st.session_state.model_key
        if current in MODEL_INFO:
            st.markdown(f"**Active model:** {MODEL_INFO[current]['display']}")
            st.caption(
                f"Accuracy {MODEL_INFO[current]['accuracy']*100:.2f}% · "
                f"Macro-F1 {MODEL_INFO[current]['macro_f1']*100:.2f}% (test set)"
            )
        st.caption("Switch models from the dropdown on the Classifier page.")
        st.divider()
        st.caption("Corpus: full_no_environment · 5 classes · max_length 512")


def main() -> None:
    render_header()
    render_sidebar()
    page = st.session_state.page
    if page == "Classifier":
        page_classifier()
    elif page == "Session History":
        page_history()
    else:
        page_about()


main()
