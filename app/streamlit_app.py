"""
app/streamlit_app.py
─────────────────────
Healthcare Clinical Severity RAG — Streamlit Dashboard

4-step workflow:
  Step 1 → Upload Patient Report
  Step 2 → Extract Clinical Findings
  Step 3 → Retrieve Relevant Medical Knowledge
  Step 4 → Generate Clinical Severity Assessment
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
# Ensure the project root is on sys.path so all package imports work
# regardless of how streamlit is invoked.
_APP_DIR    = Path(__file__).resolve().parent
_PROJ_ROOT  = _APP_DIR.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

# ── Internal imports ──────────────────────────────────────────────────────────
from utils.config import UPLOADS_PATH

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedSeverity AI | Clinical RAG",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ─────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ──────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Dark background ─────────────────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%);
    color: #e2e8f0;
}

/* ── Main header ─────────────────────────────────────────────────── */
.main-header {
    text-align: center;
    padding: 2rem 0 1.5rem 0;
    border-bottom: 1px solid rgba(100, 200, 255, 0.15);
    margin-bottom: 2rem;
}
.main-header h1 {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.main-header p {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-top: 0.5rem;
}

/* ── Step card ───────────────────────────────────────────────────── */
.step-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(100, 200, 255, 0.12);
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.5rem;
    transition: border-color 0.3s ease;
}
.step-card:hover {
    border-color: rgba(56, 189, 248, 0.35);
}
.step-label {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1rem;
}
.step-number {
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    color: white;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.9rem;
    flex-shrink: 0;
}
.step-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
}

/* ── Severity gauge ──────────────────────────────────────────────── */
.severity-gauge {
    text-align: center;
    padding: 2rem;
    border-radius: 20px;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.severity-score-num {
    font-size: 6rem;
    font-weight: 800;
    line-height: 1;
    text-shadow: 0 0 40px currentColor;
}
.severity-score-label {
    font-size: 1rem;
    opacity: 0.7;
    margin-top: 0.25rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
}
.severity-level-badge {
    display: inline-block;
    padding: 0.5rem 2rem;
    border-radius: 50px;
    font-size: 1.4rem;
    font-weight: 700;
    margin-top: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ── Finding card ────────────────────────────────────────────────── */
.finding-item {
    background: rgba(56, 189, 248, 0.07);
    border-left: 3px solid #38bdf8;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 1rem;
    margin: 0.4rem 0;
    color: #e2e8f0;
    font-size: 0.95rem;
}

/* ── Evidence card ───────────────────────────────────────────────── */
.evidence-item {
    background: rgba(99, 102, 241, 0.07);
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 1rem;
    margin: 0.4rem 0;
    color: #c7d2fe;
    font-size: 0.92rem;
    font-style: italic;
}

/* ── Reference card ──────────────────────────────────────────────── */
.reference-card {
    background: rgba(52, 211, 153, 0.06);
    border: 1px solid rgba(52, 211, 153, 0.2);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.6rem 0;
}
.reference-source {
    color: #34d399;
    font-weight: 600;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}
.reference-text {
    color: #94a3b8;
    font-size: 0.88rem;
    line-height: 1.6;
}

/* ── Summary box ─────────────────────────────────────────────────── */
.summary-box {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    color: #cbd5e1;
    font-size: 0.97rem;
    line-height: 1.75;
}

/* ── Metric tile ─────────────────────────────────────────────────── */
.metric-tile {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(100, 200, 255, 0.15);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #38bdf8;
}
.metric-label {
    font-size: 0.78rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.2rem;
}

/* ── Sidebar ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(10, 15, 30, 0.95);
    border-right: 1px solid rgba(100, 200, 255, 0.1);
}
.sidebar-logo {
    text-align: center;
    padding: 1rem 0;
}

/* ── Status pills ────────────────────────────────────────────────── */
.status-ready {
    background: rgba(52, 211, 153, 0.15);
    border: 1px solid rgba(52, 211, 153, 0.4);
    color: #34d399;
    padding: 0.3rem 0.8rem;
    border-radius: 50px;
    font-size: 0.82rem;
    font-weight: 600;
}
.status-not-ready {
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.3);
    color: #f87171;
    padding: 0.3rem 0.8rem;
    border-radius: 50px;
    font-size: 0.82rem;
    font-weight: 600;
}

/* ── Divider ─────────────────────────────────────────────────────── */
hr { border-color: rgba(100, 200, 255, 0.1); }
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "extracted_text":       "",
        "retrieved_chunks":     [],
        "severity_result":      None,
        "pipeline_ready":       False,
        "rag_pipeline":         None,
        "severity_analyzer":    None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Lazy-load heavy objects (cached across reruns) ────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_pipeline():
    """Load RAG pipeline singleton — runs once per Streamlit process."""
    from rag.rag_pipeline import RAGPipeline
    return RAGPipeline()

@st.cache_resource(show_spinner=False)
def _load_analyzer():
    """Load Gemini analyzer singleton."""
    from llm.severity_analyzer import SeverityAnalyzer
    return SeverityAnalyzer()

def _get_pipeline():
    if st.session_state.rag_pipeline is None:
        st.session_state.rag_pipeline = _load_pipeline()
    return st.session_state.rag_pipeline

def _get_analyzer():
    if st.session_state.severity_analyzer is None:
        st.session_state.severity_analyzer = _load_analyzer()
    return st.session_state.severity_analyzer


# ── Helpers ───────────────────────────────────────────────────────────────────

def _severity_colors(level: str) -> tuple[str, str]:
    """Return (background_css, text_color) for a severity level."""
    return {
        "Low":      ("linear-gradient(135deg,#052e16,#14532d)", "#22c55e"),
        "Moderate": ("linear-gradient(135deg,#1c1400,#422006)", "#eab308"),
        "High":     ("linear-gradient(135deg,#1c0a00,#431407)", "#f97316"),
        "Critical": ("linear-gradient(135deg,#1c0000,#450a0a)", "#ef4444"),
    }.get(level, ("rgba(255,255,255,0.05)", "#94a3b8"))

def _save_upload(uploaded_file) -> Path:
    """Persist uploaded PDF to the uploads directory."""
    UPLOADS_PATH.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = UPLOADS_PATH / f"{ts}_{uploaded_file.name}"
    dest.write_bytes(uploaded_file.getvalue())
    return dest


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div style="font-size:3rem">🏥</div>
        <div style="font-size:1.1rem;font-weight:700;color:#38bdf8">MedSeverity AI</div>
        <div style="font-size:0.78rem;color:#64748b">Clinical RAG System</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Knowledge base status
    st.markdown("##### 📚 Knowledge Base")
    pipeline = _get_pipeline()
    kb_ready = pipeline.is_knowledge_base_ready()
    chunk_count = pipeline.get_document_count()

    if kb_ready:
        st.markdown(f'<span class="status-ready">✅ Ready — {chunk_count:,} chunks</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-not-ready">⚠️ Not built yet</span>',
                    unsafe_allow_html=True)

    st.markdown("")
    force_rebuild = st.checkbox("Force rebuild", value=False,
                                help="Delete existing index and re-ingest all guidelines.")

    if st.button("🔨 Build Knowledge Base", use_container_width=True, type="primary"):
        log_area = st.empty()
        progress_bar = st.progress(0)
        messages: list[str] = []

        def _progress(msg: str):
            messages.append(msg)
            log_area.text("\n".join(messages[-6:]))

        with st.spinner("Ingesting medical guidelines…"):
            total = pipeline.build_knowledge_base(
                progress_callback=_progress,
                force_rebuild=force_rebuild,
            )
        progress_bar.progress(1.0)
        if total > 0:
            st.success(f"✅ Ingested {total:,} chunks!")
            st.cache_resource.clear()
            st.rerun()
        else:
            st.error("No documents found. Add .txt or .pdf files to knowledge_base/")

    st.divider()

    # Settings panel
    st.markdown("##### ⚙️ Settings")
    top_k = st.slider("Retrieved chunks (top-K)", min_value=3, max_value=10, value=5,
                      help="Number of guideline passages to retrieve per report.")

    st.divider()
    st.markdown("""
    <div style="font-size:0.75rem;color:#475569;text-align:center">
        Powered by Gemini 2.5 Flash<br>
        ChromaDB · sentence-transformers<br>
        PyMuPDF · Streamlit
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏥 MedSeverity AI</h1>
    <p>Retrieval-Augmented Clinical Severity Assessment — Grounded in Medical Guidelines</p>
</div>
""", unsafe_allow_html=True)

# Knowledge base warning banner
if not kb_ready:
    st.warning(
        "⚠️ **Knowledge base is not ready.** "
        "Click **Build Knowledge Base** in the sidebar to ingest medical guidelines before uploading a report.",
        icon="⚠️",
    )

# ── Step 1: Upload or Enter Patient Data ──────────────────────────────────────
st.markdown("""
<div class="step-card">
    <div class="step-label">
        <div class="step-number">1</div>
        <div class="step-title">Patient Data Input</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Upload-only patient data input
uploaded_file = None
uploaded_file = st.file_uploader(
    "Upload a PDF lab report (blood test, metabolic panel, urine test, etc.)",
    type=["pdf"],
    key="pdf_uploader",
    label_visibility="collapsed",
)

st.caption("Lab reports, metabolic panels, CBCs, urinalyses, LFTs, and more.")


# ── Step 2: Extract ───────────────────────────────────────────────────────────
# Input is driven by uploaded PDF only
has_pdf = uploaded_file is not None

if has_pdf:
    st.markdown("""
    <div class="step-card">
        <div class="step-label">
            <div class="step-number">2</div>
            <div class="step-title">Clinical Data Preparation</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.extracted_text or st.button("🔄 Re-extract from PDF"):
        with st.spinner("Extracting text from PDF…"):
            from rag.pdf_parser import extract_text_from_pdf
            try:
                raw_bytes = uploaded_file.getvalue()
                text = extract_text_from_pdf(raw_bytes)
                st.session_state.extracted_text = text
                # Save to disk
                _save_upload(uploaded_file)
            except Exception as e:
                st.error(f"PDF extraction error: {e}")
                st.stop()

    text = st.session_state.extracted_text
    word_count = len(text.split())
    char_count = len(text)

    # Metrics row
    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-value">{word_count:,}</div>
            <div class="metric-label">Words Extracted</div>
        </div>""", unsafe_allow_html=True)
    with mc2:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-value">{char_count:,}</div>
            <div class="metric-label">Characters</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📄 View Extracted Report Text", expanded=False):
        st.text_area(
            label="extracted",
            value=text,
            height=300,
            disabled=True,
            label_visibility="collapsed",
        )

    # ── Step 3-4: Analyze & Generate Severity ─────────────────────────────────
    st.markdown("""
    <div class="step-card">
        <div class="step-label">
            <div class="step-number">3</div>
            <div class="step-title">Analysis & Clinical Severity Assessment</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Single prominent action button
    analyze_btn = st.button(
        "🚀 Analyze Report & Generate Assessment",
        use_container_width=True,
        disabled=not kb_ready,
        type="primary",
        help="Retrieves relevant medical guidelines and generates severity assessment",
    )

    if analyze_btn:
        # Step 3: Retrieve
        with st.spinner("🔍 Searching medical knowledge base for relevant guidelines…"):
            pipeline = _get_pipeline()
            chunks = pipeline.run_query(text, top_k=top_k)
            st.session_state.retrieved_chunks = chunks

        if not st.session_state.retrieved_chunks:
            st.warning("⚠️ No relevant chunks found. Ensure the knowledge base is built.")
        else:
            st.success(f"✅ Retrieved **{len(st.session_state.retrieved_chunks)}** relevant medical guideline passages.")

            with st.expander("📚 View Retrieved Medical References"):
                for i, chunk in enumerate(st.session_state.retrieved_chunks, start=1):
                    similarity_pct = f"{chunk.similarity:.0%}"
                    st.markdown(f"""
                    <div class="reference-card">
                        <div class="reference-source">
                            📖 Guideline {i} — {chunk.source.replace("_", " ").title()}
                            &nbsp;·&nbsp; Relevance: {similarity_pct}
                        </div>
                        <div class="reference-text">{chunk.text[:500]}{'…' if len(chunk.text) > 500 else ''}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Step 4: Generate Assessment
            with st.spinner("🧠 Gemini is analyzing patient findings against medical guidelines…"):
                analyzer = _get_analyzer()
                result = analyzer.analyze(text, st.session_state.retrieved_chunks)
                st.session_state.severity_result = result

    # ── Results display ────────────────────────────────────────────────────────
    result = st.session_state.severity_result
    if result is not None:
        st.divider()
        st.markdown("## 📊 Clinical Severity Assessment")

        # Parse error warning
        if result.parse_error and result.severity_score == 0:
            st.error(f"⚠️ Analysis error: {result.parse_error}")
            with st.expander("Raw model response"):
                st.text(result.raw_response)

        else:
            bg_css, text_color = _severity_colors(result.severity_level)

            # ── Severity Gauge ───────────────────────────────────────────────
            gauge_col, meta_col = st.columns([1, 1])

            with gauge_col:
                st.markdown(f"""
                <div class="severity-gauge" style="background:{bg_css}; color:{text_color};">
                    <div class="severity-score-num">{result.severity_score}</div>
                    <div class="severity-score-label">Severity Score (out of 10)</div>
                    <div class="severity-level-badge"
                         style="background:rgba(0,0,0,0.3); color:{text_color};
                                border: 2px solid {text_color};">
                        {result.severity_level}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Score bar ────────────────────────────────────────────────────
            with meta_col:
                st.markdown("**Score Distribution**")
                score_pct = result.severity_score / 10
                st.progress(score_pct)

                # Severity legend
                for lvl, (rng, col) in {
                    "🟢 Low (0–4)":       ("0–4",  "#22c55e"),
                    "🟡 Medium (5–8)":    ("5–8",  "#eab308"),
                    "🔴 Critical (9–10)": ("9–10", "#ef4444"),
                }.items():
                    marker = "◀ Current" if (
                        (rng == "0–4" and result.severity_score <= 4) or
                        (rng == "5–8" and 5 <= result.severity_score <= 8) or
                        (rng == "9–10" and result.severity_score >= 9)
                    ) else ""
                    st.markdown(
                        f'<span style="color:{col};font-size:0.88rem">{lvl}</span> '
                        f'<span style="color:#38bdf8;font-size:0.78rem">{marker}</span>',
                        unsafe_allow_html=True,
                    )

            st.divider()

            # ── Three-column layout ──────────────────────────────────────────
            col_find, col_evid = st.columns([1, 1])

            with col_find:
                st.markdown("#### 🔬 Key Clinical Findings")
                if result.key_findings:
                    for finding in result.key_findings:
                        st.markdown(
                            f'<div class="finding-item">⚡ {finding}</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No key findings extracted.")

            with col_evid:
                st.markdown("#### 📖 Medical Evidence")
                if result.evidence:
                    for ev in result.evidence:
                        st.markdown(
                            f'<div class="evidence-item">📌 {ev}</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No evidence citations provided.")

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Clinical Summary ─────────────────────────────────────────────
            st.markdown("#### 📝 Clinical Summary")
            st.markdown(
                f'<div class="summary-box">{result.summary}</div>',
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Retrieved guideline references ───────────────────────────────
            if st.session_state.retrieved_chunks:
                with st.expander("📚 Medical Guidelines Used in This Assessment"):
                    for i, chunk in enumerate(st.session_state.retrieved_chunks, 1):
                        st.markdown(f"""
                        <div class="reference-card">
                            <div class="reference-source">
                                Reference {i} — {chunk.source.replace("_", " ").title()}
                                &nbsp;·&nbsp; Relevance {chunk.similarity:.0%}
                            </div>
                            <div class="reference-text">{chunk.text}</div>
                        </div>
                        """, unsafe_allow_html=True)

            # ── Download JSON ────────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            json_output = json.dumps(result.to_dict(), indent=2)
            st.download_button(
                label="⬇️ Download Assessment (JSON)",
                data=json_output,
                file_name=f"severity_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )

            # ── Raw response toggle ──────────────────────────────────────────
            with st.expander("🔍 Raw Gemini Response (debug)"):
                st.code(result.raw_response, language="json")

else:
    # Idle state — show instructions
    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "👆 **Upload a patient laboratory report PDF** to begin the analysis pipeline.",
        icon="ℹ️",
    )
