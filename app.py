"""
AI-Powered Document Question Answering System using RAG
Streamlit + PyMuPDF + LangChain + SentenceTransformers + FAISS + Groq
"""

import os
import hashlib
import time
import streamlit as st
from dotenv import load_dotenv

# Load env (GROQ_API_KEY, GROQ_MODEL) — .env not committed
load_dotenv()

from utils.document_processor import process_uploaded_documents, SUPPORTED_EXTENSIONS
# Keep old import for backward compat (PDF flow unchanged)
from utils.pdf_processor import process_uploaded_pdfs  # noqa: F401
from utils.embeddings import load_embedding_model, get_embedding_dimension, embed_texts
from utils.vector_store import build_faiss_index, get_index_stats
from utils.rag_pipeline import answer_question, get_groq_model, get_groq_client, ANSWER_TYPES, normalize_answer_type
import numpy as np
import re
import uuid
import streamlit.components.v1 as components

# --- Page Config ---
st.set_page_config(
    page_title="AI Document QA — RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS (fixed visibility — high contrast, light+dark compatible) ---
st.markdown("""
<style>
/* Base readable text — force dark text on light cards, works in both light/dark themes */
.small { font-size: 0.9rem; color: #334155 !important; }

/* Metric card remains subtle but readable */
.metric-card { background: #f8fafc !important; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0 !important; color: #0f172a !important; }

/* Chat cards — explicit foreground/background for contrast, no white-on-white */
.chat-user {
    background: #e0f2fe !important;  /* light blue */
    color: #0f172a !important;        /* slate-900 dark text */
    padding: 12px 16px !important;
    border-radius: 10px !important;
    margin: 8px 0 !important;
    border: 1px solid #bae6fd !important;
    line-height: 1.6 !important;
}
.chat-user b { color: #0c4a6e !important; }
.chat-ai {
    background: #dcfce7 !important;  /* light green */
    color: #14532d !important;        /* very dark green for readability */
    padding: 12px 16px !important;
    border-radius: 10px !important;
    margin: 8px 0 !important;
    border: 1px solid #86efac !important;
    line-height: 1.6 !important;
}
.chat-ai b { color: #14532d !important; }

/* Sources badge — high contrast dark text on light gray */
.source-badge {
    display: inline-block !important;
    background: #f1f5f9 !important;
    color: #1e293b !important;
    padding: 6px 12px !important;
    border-radius: 999px !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    margin: 4px 6px 4px 0 !important;
    border: 1px solid #cbd5e1 !important;
}

/* Ensure container around chat history has readable border */
div[data-testid="stContainer"] {
    border: 1px solid #e2e8f0 !important;
}

/* Dark-mode override: keep same high-contrast cards (light card + dark text is still readable on dark page) */
@media (prefers-color-scheme: dark) {
    .chat-user { background: #e0f2fe !important; color: #0f172a !important; border-color: #7dd3fc !important; }
    .chat-ai { background: #dcfce7 !important; color: #14532d !important; border-color: #86efac !important; }
    .source-badge { background: #f1f5f9 !important; color: #1e293b !important; border-color: #94a3b8 !important; }
    .small { color: #cbd5e1 !important; }
}

/* Remove any accidental white-on-white: force markdown text inside cards to inherit card color */
.chat-user *, .chat-ai *, .source-badge * { color: inherit !important; }
</style>
""", unsafe_allow_html=True)

# --- Helpers: Mermaid rendering ---
def render_answer_with_mermaid(answer: str):
    """
    Render answer markdown, handling ```mermaid blocks with visual diagram via components.html.
    Falls back to code block + caption if JS unavailable.
    """
    if not answer:
        st.markdown(answer)
        return
    # Check for mermaid blocks
    mermaid_pattern = r"```mermaid\s*\n(.*?)\n```"
    if "```mermaid" not in answer:
        st.markdown(answer)
        return
    # Split into text / mermaid alternations
    parts = re.split(mermaid_pattern, answer, flags=re.DOTALL)
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Text part — may still contain other markdown
            if part.strip():
                st.markdown(part)
        else:
            # Mermaid code
            mcode = part.strip()
            # Render via mermaid.js inside iframe
            mermaid_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
mermaid.initialize({{ startOnLoad: true, theme: 'default', securityLevel: 'loose' }});
</script>
<style>
  body {{ margin:0; padding:12px; background:white; display:flex; justify-content:center; }}
  .mermaid {{ background:white; padding:8px; border-radius:8px; }}
</style>
</head>
<body>
<pre class="mermaid">
{mcode}
</pre>
</body>
</html>
"""
            try:
                # Estimate height based on lines (min 250, max 600)
                h = min(600, max(250, 40 * len(mcode.splitlines()) + 80))
                components.html(mermaid_html, height=h, scrolling=True)
            except Exception:
                st.code(mcode, language="mermaid")
            # Also show copyable code in expander
            with st.expander("📊 Diagram code (Mermaid) — copy to https://mermaid.live if not rendered above", expanded=False):
                st.code(mcode, language="mermaid")
                st.caption("Diagram is generated ONLY from retrieved context. If diagram is not relevant, this section is omitted by the model.")

# --- Helper: format source for multi-format citations ---
def _format_source_display(c: dict) -> str:
    """Return citation string: PDF->Page N, PPTX->Slide N, others->FILETYPE."""
    fn = c.get("filename", "unknown")
    ft = (c.get("file_type") or "").lower()
    if not ft:
        import os as _os
        ft = _os.path.splitext(fn)[1].lower().lstrip(".")
    pn = c.get("page_number", 1)
    if ft == "pptx":
        return f"{fn} | Slide {pn}"
    if ft == "pdf":
        return f"{fn} | Page {pn}"
    if ft in ("docx", "txt", "csv"):
        return f"{fn} | {ft.upper()}"
    try:
        return f"{fn} | Page {int(pn)}"
    except Exception:
        return fn


def _source_badge_html(s: dict) -> str:
    """HTML badge for a source dict."""
    label = _format_source_display(s)
    return f"<span class='source-badge'>📄 {label}</span>"

# --- Session State Init ---
# Need total_slides for PPTX; keep total_pages for PDF backward compat; per-file pagination for dashboard
for key, default in [
    ("chunks", []),
    ("faiss_index", None),
    ("embeddings", None),  # stored to allow per-doc filtered search without recompute
    ("embedding_dim", 0),
    ("file_names", []),
    ("file_hash", None),
    ("chat_history", []),
    ("processed_messages", []),
    ("vectors", 0),
    ("total_pages", 0),
    ("total_slides", 0),
    ("per_file_pages", {}),
    ("per_file_slides", {}),
    ("processing_time", 0.0),
    ("embedding_time", 0.0),  # isolated embedding generation time for before/after comparison
    ("faiss_time", 0.0),
    ("selected_doc", "All Documents"),
    ("doc_chunk_counts", {}),
    ("answer_type", "10 Marks"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- Marks-selection sync (callback pattern — the ONLY writer of widget keys) ---
# Root-cause fix: the old code overwrote `sidebar_answer_type` / `main_answer_type`
# in the script body BEFORE each widget was instantiated, using the stale canonical
# `answer_type`. That reverted the user's just-made choice back to the previous value
# (default "10 Marks") on the very same run, so 2/5 could never stick.
# Rule: never assign a widget's session-state key in the script body. Writes to
# another widget's key are ONLY allowed inside on_change callbacks (which run before
# the next script run, i.e. before instantiation).
def _on_sidebar_answer_type_change():
    """Sidebar selectbox changed -> update canonical + mirror main widget."""
    val = normalize_answer_type(st.session_state.get("sidebar_answer_type", "10 Marks"))
    st.session_state.answer_type = val
    st.session_state.main_answer_type = val


def _on_main_answer_type_change():
    """Main-panel selectbox changed -> update canonical + mirror sidebar widget."""
    val = normalize_answer_type(st.session_state.get("main_answer_type", "10 Marks"))
    st.session_state.answer_type = val
    st.session_state.sidebar_answer_type = val


# Seed widget keys ONCE from canonical (only when absent — never overwrite existing).
if "sidebar_answer_type" not in st.session_state:
    st.session_state.sidebar_answer_type = st.session_state.answer_type
if "main_answer_type" not in st.session_state:
    st.session_state.main_answer_type = st.session_state.answer_type

# --- Header ---
st.title("📄 AI-Powered Document Question Answering System")
st.markdown("**Ask questions from your uploaded documents using Retrieval-Augmented Generation.**")
st.caption("Upload PDFs, DOCX, TXT, PPTX, CSV — ask questions, get grounded answers with citations (filename + page/slide/file).")
st.caption("Tech: Streamlit • PyMuPDF • python-docx • python-pptx • pandas • LangChain • all-MiniLM-L6-v2 • FAISS • Groq LLM")

with st.expander("ℹ️ How RAG Works — Architecture", expanded=False):
    st.markdown("""
```
Documents (PDF, DOCX, TXT, PPTX, CSV)
  ↓
Text Extraction (PyMuPDF / python-docx / python-pptx / pandas / Python file reading — with filename + file_type + page/slide metadata)
  ↓
Text Chunking (RecursiveCharacterTextSplitter — 500 chars, 100 overlap — preserves metadata)
  ↓
Embedding Generation (SentenceTransformer all-MiniLM-L6-v2 — 384-dim) — ONE call for ALL docs
  ↓
FAISS Vector Database (IndexFlatL2 — float32) — single shared index across all formats
  ↓
Question Embedding (same model)
  ↓
Similarity Search (top-k = 3–5, semantic across all docs)
  ↓
Relevant Context + Question → Groq LLM
  ↓
Answer + Sources (📄 filename | Page N / Slide N / DOCX / TXT / CSV)
```
**Flow:** Upload (mix formats) → extract per format → chunk (with metadata) → embed ONE shot for ALL chunks → FAISS → *cached in session state* → per question: embed query → search → Groq → answer.

**Optimizations:** Model loads once (`@st.cache_resource` — `load_embedding_model()` never in loop), embeddings generated ONCE for all chunks together with `model.encode(batch_size=64, show_progress_bar=True)` fallback 32, single FAISS index, no regen on question/mark-change/rerun.
    """)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_model_env = get_groq_model()
    st.text_input("Groq Model (from .env)", value=groq_model_env, disabled=True, help="Set GROQ_MODEL in .env to change. Fallback is openai/gpt-oss-120b (verified 2026-08-29)")
    api_ready = get_groq_client() is not None
    if api_ready:
        st.success("GROQ_API_KEY: configured ✓")
    else:
        st.error("GROQ_API_KEY: missing ✗")
        st.caption("Add `GROQ_API_KEY=your_key` to `.env` or Streamlit secrets. Get key at console.groq.com/keys")

    st.divider()
    st.subheader("Chunking & Retrieval")
    chunk_size = st.slider("Chunk size", 300, 1000, 500, 50, help="Characters per chunk")
    chunk_overlap = st.slider("Chunk overlap", 0, 200, 100, 10)
    top_k = st.slider("Top-k chunks to retrieve", 1, 10, 4, help="Number of relevant chunks sent to LLM (3–5 recommended)")
    st.caption("Recommended: chunk_size=500, overlap=100, top_k=4")

    st.divider()
    # ── EXAM ANSWER MODE ── Sidebar selector (persistent)
    st.subheader("📝 Exam Answer Mode")
    # Canonical value lives in st.session_state.answer_type. Each selectbox owns
    # ONLY its own key; cross-widget sync happens in on_change callbacks above.
    # No index override, no key assignment, no st.rerun() here.
    st.selectbox(
        "Answer Type",
        options=ANSWER_TYPES,
        key="sidebar_answer_type",
        on_change=_on_sidebar_answer_type_change,
        help="2 Marks: concise (30-50w) | 5 Marks: medium (100-150w) | 10 Marks: detailed exam answer (200-350w) with headings + diagram when relevant | Detailed Answer: comprehensive",
    )
    sidebar_answer_type = normalize_answer_type(st.session_state.sidebar_answer_type)
    # Show description for selected mode
    _desc = {
        "2 Marks": "Concise: definition + 1-2 points. ~30-50 words.",
        "5 Marks": "Medium: intro + explanation + points + example. ~100-150 words.",
        "10 Marks": "📚 **Most detailed** — B.Tech exam format with headings, points & diagram when relevant. ~200–350 words.",
        "Detailed Answer": "Comprehensive explanation with headings & diagram if helpful.",
    }
    st.caption(_desc.get(sidebar_answer_type, ""))
    if sidebar_answer_type == "10 Marks":
        st.info("10-Mark mode: includes Introduction, Explanation, Components, Working, Steps, Advantages, Applications, Limitations (if in context), Diagram, Conclusion — only sections supported by PDFs.", icon="🎓")
    st.divider()
    # Stats
    st.subheader("📊 Index Stats")
    # Show embedding dim even before processing (from cached model)
    try:
        _model = load_embedding_model()
        _dim = get_embedding_dimension(_model)
        st.metric("Embedding dimension", _dim)
    except Exception:
        st.metric("Embedding dimension", st.session_state.get("embedding_dim", 0) or "—")

    stats = get_index_stats(st.session_state.faiss_index, st.session_state.chunks) if st.session_state.faiss_index else {"vectors": 0, "dimension": st.session_state.embedding_dim, "chunks": len(st.session_state.chunks)}
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Files", len(st.session_state.file_names))
        st.metric("📄 Pages (PDF)", st.session_state.total_pages or 0)
        st.metric("Slides (PPTX)", st.session_state.get("total_slides", 0) or 0)
    with col2:
        st.metric("Chunks", len(st.session_state.chunks))
        st.metric("Vectors", stats.get("vectors", 0) or len(st.session_state.chunks))
        st.metric("Dimension", stats.get("dimension", 0) or st.session_state.embedding_dim or "—")
    # Keep total time in full width
    st.metric("⏱️ Time", f"{st.session_state.processing_time}s" if st.session_state.processing_time else "—")
    if st.session_state.processing_time:
        st.caption(f"Last processing: {st.session_state.processing_time}s (embeddings {st.session_state.embedding_time}s)")
        if st.session_state.embedding_time:
            st.caption(f"Embedding time: {st.session_state.embedding_time}s — compare before/after optimization")

    st.divider()
    if st.button("🗑️ Clear All — Documents + Chat", use_container_width=True):
        st.session_state.chunks = []
        st.session_state.faiss_index = None
        st.session_state.embeddings = None
        st.session_state.file_names = []
        st.session_state.file_hash = None
        st.session_state.chat_history = []
        st.session_state.processed_messages = []
        st.session_state.vectors = 0
        st.session_state.total_pages = 0
        st.session_state.total_slides = 0
        st.session_state.per_file_pages = {}
        st.session_state.per_file_slides = {}
        st.session_state.processing_time = 0.0
        st.session_state.embedding_time = 0.0
        st.session_state.faiss_time = 0.0
        st.session_state.selected_doc = "All Documents"
        st.session_state.doc_chunk_counts = {}
        st.success("Cleared documents and chat history.")
        st.rerun()

    st.caption("Tip: Documents are processed only when files change. Asking a question does NOT re-embed documents.")

# --- Main Column Layout ---
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.subheader("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload one or multiple documents (mix formats allowed)",
        type=["pdf", "docx", "txt", "pptx", "csv"],
        accept_multiple_files=True,
        help="Supported: .pdf (PyMuPDF), .docx (python-docx), .txt (Python), .pptx (python-pptx), .csv (pandas). Mix formats: e.g., PDF+DOCX+TXT+PPTX. Each chunk keeps filename + file type + page/slide metadata. Empty/scanned files will show a warning."
    )

    if uploaded_files:
        st.caption(f"Selected {len(uploaded_files)} file(s): " + ", ".join([f"`{f.name}`" for f in uploaded_files]))

        # Compute hash to detect change — avoid reprocessing if same files
        # Optimization: read each file's bytes ONCE and reuse (avoid duplicate getvalue/ extraction)
        file_data = [(f.name, f.getvalue()) for f in uploaded_files]
        file_bytes_concat = b"".join([b for _, b in file_data])
        meta_str = "|".join([f"{name}:{len(b)}" for name, b in file_data])
        new_hash = hashlib.md5(file_bytes_concat + meta_str.encode()).hexdigest()

        needs_processing = (st.session_state.file_hash != new_hash) or (st.session_state.faiss_index is None and len(st.session_state.chunks) == 0)

        if needs_processing:
            if st.button("⚡ Process Documents & Build Vector Index", type="primary", use_container_width=True):
                # Optimized processing pipeline per spec (multi-format):
                # Extract per format -> text splitting (preserving metadata) -> embedding generation (once) -> FAISS index creation
                # Displays: "Extracting text..." "Creating chunks..." "Generating embeddings..." "Building FAISS index..." "Ready!"
                # Model loaded ONCE via @st.cache_resource, embeddings batched 64 (fallback 32), FAISS cached, no regen on rerun
                progress = st.progress(0, text="Starting processing...")
                status_text = st.empty()
                t_start = time.time()
                try:
                    # ── Step 1: Multi-format extraction ──
                    status_text.info("Extracting text from all documents...")
                    progress.progress(15, text="Extracting text...")
                    # Single call processes ALL documents together (PDF/DOCX/TXT/PPTX/CSV); preserves filename + file_type + page/slide
                    all_chunks, messages = process_uploaded_documents(uploaded_files, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    st.session_state.processed_messages = messages
                    for msg in messages:
                        if msg.startswith("✓"):
                            st.toast(msg, icon="✅")
                        elif "Skipped" in msg or "No extractable" in msg or "unsupported" in msg.lower():
                            st.toast(msg, icon="⚠️")
                    # Compute pages/slides directly from chunk metadata (robust, not regex on messages)
                    # This ensures PDF page numbers and PPTX slide numbers are always accurate in dashboard
                    from collections import defaultdict as _dd
                    _pdf_pages = _dd(set)
                    _pptx_slides = _dd(set)
                    total_pages = 0
                    total_slides = 0
                    if all_chunks:
                        for _c in all_chunks:
                            _ft = (_c.get("file_type") or "").lower()
                            _fn = _c.get("filename", "")
                            _pn = _c.get("page_number", 1)
                            if _ft == "pdf":
                                _pdf_pages[_fn].add(_pn)
                            elif _ft == "pptx":
                                _pptx_slides[_fn].add(_pn)
                        total_pages = sum(len(v) for v in _pdf_pages.values())
                        total_slides = sum(len(v) for v in _pptx_slides.values())
                    # Fallback to regex on messages if chunk metadata missing (backward compat)
                    if total_pages == 0 and total_slides == 0:
                        import re as _re
                        for msg in messages:
                            m = _re.search(r"(\d+)\s+pages", msg)
                            if m:
                                total_pages += int(m.group(1))
                            m2 = _re.search(r"(\d+)\s+slides", msg)
                            if m2:
                                total_slides += int(m2.group(1))
                    st.session_state.total_pages = total_pages
                    st.session_state.total_slides = total_slides
                    st.session_state.per_file_pages = {k: len(v) for k, v in _pdf_pages.items()}
                    st.session_state.per_file_slides = {k: len(v) for k, v in _pptx_slides.items()}
                    total_units = total_pages + total_slides if (total_pages or total_slides) else len(all_chunks)
                    # ── Step 2: Creating chunks ──
                    progress.progress(40, text="Creating chunks...")
                    status_text.info(f"Creating chunks... {len(all_chunks)} chunks from {len(uploaded_files)} files")
                    if not all_chunks:
                        st.error("No chunks created. No searchable content extracted. Check messages below.")
                        st.session_state.chunks = []
                        st.session_state.faiss_index = None
                        st.session_state.embeddings = None
                        st.session_state.file_names = []
                        st.session_state.total_pages = 0
                        st.session_state.total_slides = 0
                        st.session_state.per_file_pages = {}
                        st.session_state.per_file_slides = {}
                        st.session_state.doc_chunk_counts = {}
                        progress.empty()
                        status_text.empty()
                    else:
                        # ── Step 3: Generating embeddings ──
                        progress.progress(55, text="Generating embeddings...")
                        status_text.info(f"Generating embeddings... {len(all_chunks)} chunks (batch_size=64, fallback 32)")
                        # Load model ONCE outside any loop (cached via @st.cache_resource)
                        model = load_embedding_model()
                        dim = get_embedding_dimension(model)
                        st.session_state.embedding_dim = dim
                        texts = [c["text"] for c in all_chunks]
                        # Spec: batch_size=64 with show_progress_bar=True, fallback to 32 on memory error
                        # Generate embeddings ONCE for ALL chunks together (not per-PDF)
                        effective_batch = 64
                        n = len(texts)
                        num_batches = (n + effective_batch - 1) // effective_batch
                        emb_progress = st.progress(0, text=f"Generating embeddings... 0/{n} chunks")
                        t_embed_start = time.time()
                        embeddings_list = []
                        batch_failed_fallback = False
                        for batch_idx, start in enumerate(range(0, n, effective_batch)):
                            end = min(start + effective_batch, n)
                            batch_texts = texts[start:end]
                            try:
                                batch_emb = model.encode(
                                    batch_texts,
                                    batch_size=effective_batch,
                                    show_progress_bar=True,
                                    convert_to_numpy=True
                                )
                            except RuntimeError as e:
                                # Memory pressure -> fallback to 32 per spec
                                if effective_batch == 64 and ("memory" in str(e).lower() or "oom" in str(e).lower()):
                                    batch_failed_fallback = True
                                    effective_batch = 32
                                    num_batches = (n + effective_batch - 1) // effective_batch
                                    # Retry this batch with 32
                                    batch_emb = model.encode(
                                        batch_texts,
                                        batch_size=32,
                                        show_progress_bar=True,
                                        convert_to_numpy=True
                                    )
                                else:
                                    raise
                            if batch_emb.dtype != np.float32:
                                batch_emb = batch_emb.astype(np.float32)
                            embeddings_list.append(batch_emb)
                            frac = end / n
                            try:
                                emb_progress.progress(frac, text=f"Generating embeddings... {end}/{n} chunks (batch {batch_idx+1}/{num_batches})")
                            except TypeError:
                                emb_progress.progress(frac)
                            try:
                                main_frac = 55 + int(frac * 30)
                                progress.progress(main_frac, text=f"Generating embeddings... {end}/{n} chunks")
                            except Exception:
                                pass
                        if len(embeddings_list) == 1:
                            embeddings = embeddings_list[0]
                        else:
                            embeddings = np.vstack(embeddings_list).astype(np.float32)
                        t_embed = round(time.time() - t_embed_start, 2)
                        st.session_state.embedding_time = t_embed
                        emb_progress.progress(1.0, text=f"Embeddings done: {embeddings.shape[0]} vectors × {dim} dim in {t_embed}s (batch {effective_batch})")
                        if batch_failed_fallback:
                            st.toast("Batch size 64 caused memory pressure — automatically fell back to 32", icon="⚠️")
                        # ── Step 4: Building FAISS index ──
                        progress.progress(85, text="Building FAISS index...")
                        status_text.info("Building FAISS index...")
                        t_faiss_start = time.time()
                        with st.status("Building FAISS index…", expanded=False) as s:
                            index = build_faiss_index(embeddings)
                            s.update(label=f"FAISS ready: {index.ntotal} vectors (IndexFlatL2, float32)", state="complete")
                        t_faiss = round(time.time() - t_faiss_start, 2)
                        st.session_state.faiss_time = t_faiss
                        # ── Step 5: Cache for reuse (avoid regen on question / mark change / rerun) ──
                        # ONE shared FAISS index containing ALL documents; position -> chunk -> filename/file_type/page_number
                        st.session_state.chunks = all_chunks
                        st.session_state.embeddings = embeddings
                        st.session_state.faiss_index = index
                        st.session_state.vectors = index.ntotal
                        st.session_state.file_names = [f.name for f in uploaded_files]
                        st.session_state.file_hash = new_hash
                        st.session_state.processing_time = round(time.time() - t_start, 2)
                        from collections import Counter as _Counter
                        st.session_state.doc_chunk_counts = dict(_Counter([c["filename"] for c in all_chunks]))
                        st.session_state.selected_doc = "All Documents"
                        progress.progress(100, text="Ready!")
                        status_text.success(f"Ready! Total {st.session_state.processing_time}s (embeddings {t_embed}s, FAISS {t_faiss}s)")
                        emb_progress.empty()
                        st.success(f"✅ Indexed {len(all_chunks)} chunks from {len(uploaded_files)} file(s) in {st.session_state.processing_time}s — embeddings {t_embed}s (batch {effective_batch})  •  PDF pages: {total_pages}  •  PPTX slides: {total_slides}")
                        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
                        c1.metric("📄 Files", len(uploaded_files))
                        c2.metric("📑 Pages", total_pages or 0)
                        c3.metric("📊 Slides", total_slides or 0)
                        c4.metric("🧩 Chunks", len(all_chunks))
                        c5.metric("🔢 Vectors", index.ntotal)
                        c6.metric("📐 Dim", dim)
                        c7.metric("⏱️ Embed", f"{t_embed}s")
                        st.caption(f"Pipeline: {total_pages} pages + {total_slides} slides → {len(all_chunks)} chunks → {index.ntotal} vectors (batch {effective_batch}, cached model) → FAISS {t_faiss}s — total {st.session_state.processing_time}s")
                        # Detail per-format counts + per-file
                        from collections import Counter as _Cnt2
                        ft_counts = _Cnt2([c.get("file_type","unknown") for c in all_chunks])
                        st.caption(f"Per-format chunks: {dict(ft_counts)}  •  PDF pages: {total_pages} | PPTX slides: {total_slides}")
                        # Per-file explicit for dashboard
                        try:
                            _details2 = []
                            for _fn in [f.name for f in uploaded_files]:
                                _cnt = sum(1 for _cc in all_chunks if _cc["filename"]==_fn)
                                # find file_type
                                _ft2 = next(( _cc.get("file_type","") for _cc in all_chunks if _cc["filename"]==_fn), "")
                                _ft2 = (_ft2 or "").lower()
                                if _ft2=="pdf":
                                    _pg = len(set(_cc["page_number"] for _cc in all_chunks if _cc["filename"]==_fn))
                                    _details2.append(f"{_fn}: {_pg} pages")
                                elif _ft2=="pptx":
                                    _sl = len(set(_cc["page_number"] for _cc in all_chunks if _cc["filename"]==_fn))
                                    _details2.append(f"{_fn}: {_sl} slides")
                                else:
                                    _details2.append(f"{_fn}: {_cnt} chunks ({_ft2.upper()})")
                            st.caption("Per-file: " + " | ".join(_details2))
                        except Exception:
                            pass
                        st.rerun()
                except Exception as e:
                    st.error(f"Processing failed: {e}")
                    if 'all_chunks' not in locals() or not all_chunks:
                        st.session_state.chunks = []
                        st.session_state.faiss_index = None
                    progress.empty()
                    status_text.empty()
        else:
            st.info(f"✓ Already processed {len(st.session_state.chunks)} chunks from {len(st.session_state.file_names)} file(s). Ask a question below. (Re-upload or change files to reprocess)")
            if st.session_state.processed_messages:
                with st.expander("Processing log"):
                    for m in st.session_state.processed_messages:
                        if m.startswith("✓"):
                            st.success(m)
                        elif "Skipped" in m or "No " in m:
                            st.warning(m)
                        else:
                            st.write(m)

    else:
        if st.session_state.chunks:
            st.info(f"Documents in memory: {len(st.session_state.file_names)} file(s), {len(st.session_state.chunks)} chunks. Upload new files to replace, or ask questions below.")
        else:
            st.warning("No documents uploaded yet. Upload at least one text-based document (PDF/DOCX/TXT/PPTX/CSV) to begin.")

    # Processing log expander always visible if messages exist
    if st.session_state.processed_messages and not uploaded_files:
        with st.expander("Last processing log"):
            for m in st.session_state.processed_messages:
                st.write(m)

    st.divider()
    # Document Processing Status section per spec — includes all required stats + embedding time
    st.subheader("📊 Document Processing Status")
    if st.session_state.chunks:
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("📄 Files", len(st.session_state.file_names))
        c2.metric("📑 Pages", st.session_state.total_pages or 0)
        c3.metric("📊 Slides", st.session_state.get("total_slides", 0) or 0)
        c4.metric("🧩 Chunks", len(st.session_state.chunks))
        c5.metric("🔢 Vectors", st.session_state.vectors or len(st.session_state.chunks))
        c6.metric("📐 Dim", st.session_state.embedding_dim or 384)
        c7.metric("⏱️ Embed", f"{st.session_state.embedding_time}s" if st.session_state.embedding_time else f"{st.session_state.processing_time}s")
        st.caption(f"Total {st.session_state.processing_time}s — embeddings {st.session_state.embedding_time}s — FAISS {st.session_state.faiss_time}s")
        # Per-format breakdown + per-file page/slide counts
        try:
            from collections import Counter as _CC
            _ftc = _CC([c.get("file_type","unk") for c in st.session_state.chunks])
            st.caption(f"Per-format chunks: {dict(_ftc)}  •  PDF pages: {st.session_state.total_pages}  •  PPTX slides: {st.session_state.get('total_slides',0)}")
        except Exception:
            pass
        # Show per-file pages/slides table for dashboard verification
        try:
            _pfp = st.session_state.get("per_file_pages", {})
            _pfs = st.session_state.get("per_file_slides", {})
            if _pfp or _pfs:
                _details = []
                for _fn in st.session_state.file_names:
                    _ft = None
                    for _cc in st.session_state.chunks:
                        if _cc["filename"] == _fn:
                            _ft = _cc.get("file_type","")
                            break
                    _ft = (_ft or "").lower()
                    if _ft == "pdf":
                        _details.append(f"`{_fn}`: { _pfp.get(_fn,0)} pages")
                    elif _ft == "pptx":
                        _details.append(f"`{_fn}`: { _pfs.get(_fn,0)} slides")
                    else:
                        _details.append(f"`{_fn}`: {st.session_state.doc_chunk_counts.get(_fn,0)} chunks ({_ft.upper() if _ft else 'UNK'})")
                st.caption("Per-file: " + " | ".join(_details))
        except Exception:
            pass
    else:
        st.caption("No documents processed yet. Upload documents (PDF/DOCX/TXT/PPTX/CSV) and click Process.")

    # Debug verification: Documents in index with per-file chunk counts (collapsible) — explicit page/slide numbers
    if st.session_state.chunks and st.session_state.doc_chunk_counts:
        with st.expander("🔍 Documents in index (debug — verification)", expanded=False):
            st.markdown("**Documents in index (with page/slide counts):**")
            _pfp_dbg = st.session_state.get("per_file_pages", {})
            _pfs_dbg = st.session_state.get("per_file_slides", {})
            for fname, cnt in st.session_state.doc_chunk_counts.items():
                _ft_dbg = next((c.get("file_type","") for c in st.session_state.chunks if c["filename"]==fname), "")
                _ft_dbg = (_ft_dbg or "").lower()
                if _ft_dbg == "pdf":
                    _pg_dbg = _pfp_dbg.get(fname, 0)
                    st.markdown(f"- **{fname}** — **{cnt} chunks** — **{_pg_dbg} pages** ({_ft_dbg})")
                elif _ft_dbg == "pptx":
                    _sl_dbg = _pfs_dbg.get(fname, 0)
                    st.markdown(f"- **{fname}** — **{cnt} chunks** — **{_sl_dbg} slides** ({_ft_dbg})")
                else:
                    # DOCX/TXT/CSV — no pagination, show chunks
                    # Find original message for reference
                    pages_info = ""
                    for msg in st.session_state.processed_messages:
                        if fname in msg:
                            pages_info = msg.replace("✓","").strip()
                            break
                    st.markdown(f"- **{fname}** — **{cnt} chunks** — {pages_info} ({_ft_dbg.upper() if _ft_dbg else 'UNK'})")
            st.caption(f"Dashboard totals: **{st.session_state.total_pages} PDF pages** + **{st.session_state.get('total_slides',0)} PPTX slides** = {st.session_state.total_pages + st.session_state.get('total_slides',0)} paginated units  •  FAISS: {st.session_state.vectors} vectors × {st.session_state.embedding_dim} dim (IndexFlatL2, float32)")
            # Show shared index verification (single FAISS across all formats)
            if st.session_state.vectors == len(st.session_state.chunks):
                st.success(f"✓ Shared index verified: {st.session_state.vectors} vectors correspond to {len(st.session_state.chunks)} chunks across {len(st.session_state.file_names)} files (all formats). Page numbers preserved via PDF page_number & PPTX slide_number.")
            else:
                st.warning(f"Index/chunk mismatch: {st.session_state.vectors} vectors vs {len(st.session_state.chunks)} chunks.")
            # Show sample metadata for verification (preserve file_type + page/slide)
            st.markdown("**Sample chunk metadata (first 2):**")
            for c in st.session_state.chunks[:2]:
                st.code(f"filename: {c['filename']}\nfile_type: {c.get('file_type','unknown')}\npage_number: {c['page_number']}\ntext: {c['text'][:120]}...", language="text")
            # Explicit table for page numbers (dashboard visibility)
            st.markdown("**Per-file pagination table:**")
            try:
                import pandas as _pd
                _rows = []
                for _fn, _cnt in st.session_state.doc_chunk_counts.items():
                    _ft4 = next((c.get("file_type","") for c in st.session_state.chunks if c["filename"]==_fn), "")
                    _ft4 = (_ft4 or "").lower()
                    _pg4 = _pfp_dbg.get(_fn, 0) if _ft4=="pdf" else "-"
                    _sl4 = _pfs_dbg.get(_fn, 0) if _ft4=="pptx" else "-"
                    _rows.append({"File": _fn, "Type": _ft4.upper(), "Pages": _pg4, "Slides": _sl4, "Chunks": _cnt})
                st.dataframe(_pd.DataFrame(_rows), use_container_width=True, hide_index=True)
            except Exception:
                pass

    st.divider()
    st.subheader("💬 Ask Questions")
    # Document selector/filter for multi-document retrieval (clean filter — all formats)
    if st.session_state.file_names and st.session_state.chunks:
        doc_options = ["All Documents"] + st.session_state.file_names
        # Ensure selected value is still valid (after Clear All or reprocessing)
        if st.session_state.selected_doc not in doc_options:
            st.session_state.selected_doc = "All Documents"
        selected = st.selectbox(
            "📂 Search in:",
            options=doc_options,
            index=doc_options.index(st.session_state.selected_doc),
            help="All Documents searches across every uploaded document (all formats). Select a single file to filter retrieval to that document only."
        )
        st.session_state.selected_doc = selected
        if selected != "All Documents":
            cnt = st.session_state.doc_chunk_counts.get(selected, 0)
            st.caption(f"Filtering retrieval to **{selected}** ({cnt} chunks)")
        else:
            st.caption(f"Searching across all {len(st.session_state.file_names)} document(s) — {len(st.session_state.chunks)} chunks total")
    else:
        # Default when no docs yet
        st.session_state.selected_doc = "All Documents"

    # ── Answer Type selector in main panel (mirrors sidebar via callbacks) ──
    st.markdown("#### 📝 Answer Type")
    # Canonical value is maintained by the on_change callbacks; just render.
    # No validation reset to "10 Marks" here — normalize for display only.
    st.selectbox(
        "Choose answer length & style",
        options=ANSWER_TYPES,
        key="main_answer_type",
        on_change=_on_main_answer_type_change,
        help="2 Marks = short (30-50w) | 5 Marks = medium (100-150w) | 10 Marks = detailed B.Tech exam format (200-350w) | Detailed Answer = comprehensive",
        label_visibility="collapsed"
    )
    main_answer_type = normalize_answer_type(st.session_state.main_answer_type)
    # Show hint for selected mode
    _hints = {
        "2 Marks": "✏️ **2 Marks** — concise: definition + 1-2 points (~30-50 words). No diagram.",
        "5 Marks": "📄 **5 Marks** — medium answer with intro, points, example & conclusion (~100-150 words).",
        "10 Marks": "🎓 **10 Marks** — detailed B.Tech exam answer with headings, points, advantages, applications, limitations & **Mermaid diagram when relevant** (~200-350 words).",
        "Detailed Answer": "📚 **Detailed Answer** — comprehensive explanation with headings & diagram if helpful.",
    }
    st.caption(_hints.get(st.session_state.answer_type, ""))
    # Small info about diagram
    if st.session_state.answer_type in ("10 Marks", "Detailed Answer"):
        st.caption("💡 Tip: For diagrams, the model generates ` ```mermaid flowchart TD ... ``` ` based **only** on retrieved PDF context (e.g., architecture, workflow, ML pipeline). Diagram is omitted if not relevant.")

    # Question input — keep history
    question = st.text_input(
        "Your question about the uploaded documents",
        placeholder="e.g., What is the leave policy? or Summarize the RAG chunking strategy...",
        label_visibility="collapsed"
    )
    col_q1, col_q2, col_q3 = st.columns([0.65, 0.15, 0.20])
    with col_q2:
        ask_clicked = st.button("Ask", type="primary", use_container_width=True)
    with col_q3:
        clear_chat = st.button("Clear Chat History", use_container_width=True)

    if clear_chat:
        st.session_state.chat_history = []
        st.success("Chat history cleared.")
        st.rerun()

    # Handle question — retrieval respects document filter (All Documents vs single PDF)
    if ask_clicked:
        if not question or not question.strip():
            st.warning("Please enter a question.")
        elif not st.session_state.chunks or st.session_state.faiss_index is None:
            st.error("No documents indexed. Upload documents and click 'Process Documents' first.")
        else:
            # Check too few chunks for top_k (use filtered count if filtered)
            _active_chunks = st.session_state.chunks
            _filter_doc = st.session_state.get("selected_doc", "All Documents")
            if _filter_doc != "All Documents":
                _active_chunks = [c for c in st.session_state.chunks if c["filename"] == _filter_doc]
            if len(_active_chunks) < top_k:
                st.toast(f"Only {len(_active_chunks)} chunks available for '{_filter_doc}' — retrieving {len(_active_chunks)} instead of {top_k}.", icon="ℹ️")
            with st.spinner("Retrieving relevant chunks & generating answer…"):
                try:
                    model = load_embedding_model()
                    # Normalize + persist: guarantees the CURRENT UI selection drives this
                    # call (handles missing/invalid/stale values; no cross-marks reuse).
                    _active_answer_type = normalize_answer_type(st.session_state.get("answer_type", "Detailed Answer"))
                    st.session_state.answer_type = _active_answer_type
                    # If filter is All Documents: use shared FAISS index (all PDFs together)
                    # If filter is specific PDF: search only that PDF's chunks using stored embeddings subset
                    if _filter_doc == "All Documents" or st.session_state.embeddings is None:
                        # Shared knowledge base: search across ALL documents
                        result = answer_question(
                            query=question.strip(),
                            chunks=st.session_state.chunks,
                            index=st.session_state.faiss_index,
                            embedding_model=model,
                            top_k=top_k,
                            answer_type=_active_answer_type
                        )
                    else:
                        # Filtered retrieval: ONE shared index was built, now create temporary filtered view
                        # Collect indices for selected doc to keep FAISS position -> chunk -> metadata correct
                        filtered_indices = [i for i, c in enumerate(st.session_state.chunks) if c["filename"] == _filter_doc]
                        filtered_chunks = [st.session_state.chunks[i] for i in filtered_indices]
                        if not filtered_chunks:
                            result = {"answer": f"No chunks found for document '{_filter_doc}'.", "sources": [], "contexts": [], "via": "error:no-filter-chunks"}
                        else:
                            # Build temporary FAISS index from stored embeddings subset (no recompute)
                            try:
                                filt_embs = st.session_state.embeddings[filtered_indices]
                                # Ensure contiguous float32 for FAISS
                                filt_embs = np.ascontiguousarray(filt_embs.astype(np.float32))
                                filt_index = build_faiss_index(filt_embs)
                                result = answer_question(
                                    query=question.strip(),
                                    chunks=filtered_chunks,
                                    index=filt_index,
                                    embedding_model=model,
                                    top_k=top_k,
                                    answer_type=_active_answer_type
                                )
                            except Exception as fe:
                                # Fallback: filter after global search if filtered index fails
                                result = answer_question(
                                    query=question.strip(),
                                    chunks=st.session_state.chunks,
                                    index=st.session_state.faiss_index,
                                    embedding_model=model,
                                    top_k=top_k * 3,
                                    answer_type=_active_answer_type
                                )
                                # Post-filter to selected doc
                                result["contexts"] = [c for c in result.get("contexts", []) if c["filename"] == _filter_doc][:top_k]
                                from utils.rag_pipeline import deduplicate_sources as _dedup
                                result["sources"] = _dedup(result["contexts"])
                                if not result["contexts"]:
                                    result["answer"] = "I couldn't find this information in the uploaded documents."
                except Exception as e:
                    result = {"answer": f"Error during QA: {e}", "sources": [], "contexts": [], "via": "error"}

            # Save to chat history (session-based) — include answer_type
            st.session_state.chat_history.append({
                "question": question.strip(),
                "answer": result["answer"],
                "sources": result.get("sources", []),
                "contexts": result.get("contexts", []),
                "via": result.get("via", ""),
                "answer_type": result.get("answer_type", st.session_state.answer_type),
            })
            # Rerun to show latest at top (or we just display)
            st.rerun()

    # --- Answer / Chat History Display ---
    if st.session_state.chat_history:
        st.markdown("### 💬 Chat History")
        # Show most recent first using st.chat_message per spec, plus custom styling fallback
        for i, turn in enumerate(reversed(st.session_state.chat_history)):
            idx = len(st.session_state.chat_history) - i
            with st.container(border=True):
                with st.chat_message("user"):
                    st.markdown(f"**Q{idx}:** {turn['question']}")
                with st.chat_message("assistant"):
                    # Answer type badge
                    _at = turn.get("answer_type", "")
                    if _at:
                        st.caption(f"📝 Answer Type: **{_at}**")
                    # Render answer with Mermaid support
                    if "```mermaid" in turn['answer']:
                        render_answer_with_mermaid(turn['answer'])
                    else:
                        st.markdown(turn['answer'])
                    # Sources as badges (multi-format: Page vs Slide vs File)
                    if turn.get("sources"):
                        st.markdown("**📚 Sources:**")
                        badges = " ".join([_source_badge_html(s) for s in turn["sources"]])
                        st.markdown(badges, unsafe_allow_html=True)
                    else:
                        st.caption("No sources (answer not grounded in retrieved chunks)")
                with st.expander(f"View retrieved chunks for Q{idx} (top {len(turn.get('contexts', []))})"):
                    if turn.get("contexts"):
                        for c in turn["contexts"]:
                            st.markdown(f"**📄 {_format_source_display(c)}** — score {c.get('score','')} | distance {c.get('distance','')}")
                            st.write(c["text"][:800] + ("…" if len(c["text"]) > 800 else ""))
                            st.divider()
                    else:
                        st.write("No contexts retrieved.")
                st.caption(f"via: `{turn.get('via','')}`")
        # Also show latest AI Answer separately per spec
        st.divider()
        st.subheader("🤖 AI Answer (latest)")
        latest = st.session_state.chat_history[-1]
        st.caption(f"📝 Answer Type: **{latest.get('answer_type', st.session_state.answer_type)}**  •  via: `{latest.get('via','')}`")
        with st.chat_message("assistant"):
            if "```mermaid" in latest["answer"]:
                render_answer_with_mermaid(latest["answer"])
            else:
                st.markdown(latest["answer"])
        st.subheader("📚 Sources (latest)")
        if latest.get("sources"):
            badges = " ".join([_source_badge_html(s) for s in latest["sources"]])
            st.markdown(badges, unsafe_allow_html=True)
        else:
            st.info("I couldn't find this information in the uploaded documents." if "not available" in latest["answer"].lower() or "no relevant" in latest["answer"].lower() else "No sources for this answer.")

with right:
    st.subheader("📊 Document Statistics")
    # Metrics — multi-format: Files, Pages, Slides, Chunks, Vectors, Embedding dim
    m1, m2 = st.columns(2)
    with m1:
        st.metric("📄 Files", len(st.session_state.file_names))
        if st.session_state.file_names:
            st.caption(", ".join(st.session_state.file_names))
        st.metric("📑 Pages (PDF)", st.session_state.total_pages or 0)
        st.metric("📊 Slides (PPTX)", st.session_state.get("total_slides", 0) or 0)
        st.metric("🧩 Chunks", len(st.session_state.chunks))
        if st.session_state.processing_time:
            st.metric("⏱️ Total", f"{st.session_state.processing_time}s")
            if st.session_state.embedding_time:
                st.caption(f"Embeddings: {st.session_state.embedding_time}s (batch 64→32)")
    with m2:
        st.metric("🔢 Vectors", st.session_state.vectors or (st.session_state.faiss_index.ntotal if st.session_state.faiss_index else 0))
        st.metric("📐 Embedding dim", st.session_state.embedding_dim or 384)
        st.metric("📂 Chunks", len(st.session_state.chunks))
        if st.session_state.processing_time:
            st.caption(f"FAISS: {st.session_state.faiss_time}s" if st.session_state.faiss_time else f"Processed in {st.session_state.processing_time}s")
        # Per-file explicit for right dashboard
        if st.session_state.chunks and st.session_state.file_names:
            st.caption("— Per-file pages/slides —")
            for _fn in st.session_state.file_names:
                _cnt = st.session_state.doc_chunk_counts.get(_fn, 0)
                _ft3 = next((c.get("file_type","") for c in st.session_state.chunks if c["filename"]==_fn), "")
                _ft3 = (_ft3 or "").lower()
                if _ft3 == "pdf":
                    _pg3 = st.session_state.get("per_file_pages", {}).get(_fn, 0)
                    st.caption(f"{_fn}: {_pg3} pages")
                elif _ft3 == "pptx":
                    _sl3 = st.session_state.get("per_file_slides", {}).get(_fn, 0)
                    st.caption(f"{_fn}: {_sl3} slides")
                else:
                    st.caption(f"{_fn}: {_cnt} chunks ({_ft3.upper()})")

    st.divider()
    st.subheader("🔍 Sources (latest answer)")
    if st.session_state.chat_history:
        latest = st.session_state.chat_history[-1]
        if latest.get("sources"):
            for s in latest["sources"]:
                st.markdown(_source_badge_html(s), unsafe_allow_html=True)
        else:
            st.caption("No sources for last answer.")
        with st.expander("Retrieved context (latest)"):
            for c in latest.get("contexts", []):
                st.markdown(f"**{_format_source_display(c)}** (score {c.get('score')})")
                st.write(c["text"])
                st.divider()
    else:
        st.caption("Ask a question to see sources here.")

    st.divider()
    with st.expander("⚠️ Error Handling & Tips", expanded=False):
        st.markdown("""
- **No document uploaded** → Upload text-based PDFs/DOCX/TXT/PPTX/CSV first.
- **Empty / scanned PDF** → Shows: “No extractable text… may be scanned… try OCR”.
- **Empty / corrupted DOCX/PPTX/CSV** → Clear “Failed to open” or “No extractable text” message; file is skipped, other files still indexed.
- **Unsupported type** → Shows: “Skipped … unsupported file type … Supported: .pdf, .docx, .txt, .pptx, .csv”.
- **Empty question** → Warning to enter a question.
- **API key missing** → Sidebar shows ✗; answer falls back to extractive excerpts.
- **Invalid API key / 401** → Clear error (key not shown).
- **Model unavailable** → Explains how to set `GROQ_MODEL` in `.env` (e.g., `openai/gpt-oss-120b`).
- **Network / Groq error** → Fallback to relevant excerpts.
- **FAISS errors** → Shown without exposing internals.
- **Too few chunks** → Automatically clamps top_k.
- **Corrupted file** → “Failed to open … may be corrupted” — skipped gracefully.
- Secrets never exposed in UI.
        """)

    with st.expander("📦 Deployment Checklist", expanded=False):
        st.markdown("""
- `requirements.txt` present with pinned compatible versions
- `.env` is in `.gitignore` (never committed)
- `.env.example` with placeholders added
- No Windows hard-coded paths
- Use Streamlit secrets for `GROQ_API_KEY` on Community Cloud: Settings → Secrets → `GROQ_API_KEY = "gsk_..."`
        """)

st.divider()
st.caption("Built for B.Tech Major Project — AI-Powered Document QA using RAG • Streamlit + PyMuPDF + LangChain + MiniLM + FAISS + Groq • Ready for Streamlit Community Cloud")
