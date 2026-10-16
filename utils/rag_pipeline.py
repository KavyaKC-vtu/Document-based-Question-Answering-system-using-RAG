"""
RAG pipeline: retrieve → context → Groq LLM → answer + sources.

- Converts question to embedding
- Searches FAISS (top_k 3-5)
- Builds context with metadata
- Calls Groq Chat Completions
- Enforces "only use provided context" prompt
- Returns answer + deduplicated sources
"""

import os
from typing import List, Dict, Tuple
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Groq client will be lazily loaded and cached


def _get_api_key() -> str | None:
    """Get GROQ_API_KEY from Streamlit secrets (Cloud) with env fallback (local)."""
    # Try Streamlit secrets first (Community Cloud)
    try:
        # st.secrets may not exist locally or may be empty
        if hasattr(st, "secrets"):
            # Using .get with fallback; also handle case where secrets file missing
            try:
                val = st.secrets.get("GROQ_API_KEY", None)  # type: ignore
                if val and str(val).strip() and str(val).strip() != "your_groq_api_key_here":
                    return str(val).strip()
            except Exception:
                # Fallback to dict-like access
                try:
                    if "GROQ_API_KEY" in st.secrets:
                        val = st.secrets["GROQ_API_KEY"]  # type: ignore
                        if val and str(val).strip() and str(val).strip() != "your_groq_api_key_here":
                            return str(val).strip()
                except Exception:
                    pass
    except Exception:
        pass
    # Fallback to environment variable (local .env)
    api_key = os.getenv("GROQ_API_KEY")
    if api_key and api_key.strip() and api_key.strip() != "your_groq_api_key_here":
        return api_key.strip()
    return None


def _get_model_name() -> str:
    """Get GROQ_MODEL from secrets or env, with safe default."""
    try:
        if hasattr(st, "secrets"):
            try:
                val = st.secrets.get("GROQ_MODEL", None)  # type: ignore
                if val and str(val).strip():
                    return str(val).strip()
            except Exception:
                try:
                    if "GROQ_MODEL" in st.secrets:
                        val = st.secrets["GROQ_MODEL"]  # type: ignore
                        if val and str(val).strip():
                            return str(val).strip()
                except Exception:
                    pass
    except Exception:
        pass
    return os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


@st.cache_resource(show_spinner=False)
def get_groq_client():
    """
    Load Groq client once (cached). Returns None if no API key.
    Handles missing/invalid key gracefully.
    Uses st.secrets on Streamlit Cloud with fallback to env vars locally.
    Never exposes key in errors.
    """
    api_key = _get_api_key()
    if not api_key:
        return None
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        return client
    except Exception as e:
        # Don't expose key in error; log safely
        print(f"[Groq] client init failed: {type(e).__name__}")
        return None


def get_groq_model() -> str:
    """Return model from secrets/env or sensible default (verified available 2026)."""
    # Verified via Groq API models.list() on 2026-08-29:
    # openai/gpt-oss-120b, openai/gpt-oss-20b, allam-2-7b, groq/compound are live.
    # Older llama-3.1-8b-instant is now 404, so default must be current.
    return _get_model_name()


def _format_source(c: Dict) -> str:
    """Return citation string respecting file type: PDF->Page, PPTX->Slide, others->file reference."""
    fn = c.get("filename", "unknown")
    ft = (c.get("file_type") or "").lower()
    # Derive from extension if file_type missing (backward compat)
    if not ft:
        import os as _os
        ft = _os.path.splitext(fn)[1].lower().lstrip(".")
    pn = c.get("page_number", 1)
    if ft == "pptx":
        return f"{fn} | Slide {pn}"
    if ft == "pdf":
        return f"{fn} | Page {pn}"
    if ft in ("docx", "txt", "csv"):
        # Provide appropriate reference: no page, but keep type for clarity
        return f"{fn} | {ft.upper()}"
    # Fallback — preserve page if present
    try:
        pn_int = int(pn)
        return f"{fn} | Page {pn_int}"
    except Exception:
        return fn


def build_context(retrieved_chunks: List[Dict]) -> str:
    """Combine retrieved chunks into context string with source tags (multi-format aware)."""
    parts = []
    for i, c in enumerate(retrieved_chunks, 1):
        src = _format_source(c)
        parts.append(f"[Source {i}: {src}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def deduplicate_sources(chunks: List[Dict]) -> List[Dict]:
    """Deduplicate by (filename, page_number, file_type). Preserves page/slide semantics."""
    seen = set()
    uniq = []
    for c in chunks:
        fn = c.get("filename", "")
        pn = c.get("page_number", 1)
        ft = (c.get("file_type") or "").lower()
        if not ft:
            import os as _os
            ft = _os.path.splitext(fn)[1].lower().lstrip(".")
        key = (fn, pn, ft)
        if key not in seen:
            seen.add(key)
            uniq.append({"filename": fn, "page_number": pn, "file_type": ft})
    return uniq


SYSTEM_PROMPT = (
    "You are a helpful AI assistant for document question answering. "
    "Answer the user's question based ONLY on the provided context. "
    "Rules:\n"
    "1. Use only information from the context. Do not invent information.\n"
    "2. If the answer cannot be found in the provided context, clearly say: "
    "\"I couldn't find this information in the uploaded documents.\" Do not guess or hallucinate.\n"
    "3. Match the depth and length to the MARKS ALLOCATED given in the user message. "
    "Never expand a short-answer (2-mark) question into a long essay.\n"
    "4. If context is empty or irrelevant, state that no relevant information was found using the same phrase: "
    "\"I couldn't find this information in the uploaded documents.\"\n"
    "5. Preserve source information — mention filename and page when citing.\n"
)

# ── EXAM ANSWER MODE CONFIG ──────────────────────────────────────────────
ANSWER_TYPES = ["2 Marks", "5 Marks", "10 Marks", "Detailed Answer"]

# Single source of truth: marks -> expected depth/length (spec: guidelines, not rigid limits)
MARKS_CONFIG = {
    "2 Marks": {"marks": 2, "target_words": "30-50", "max_tokens": 150},
    "5 Marks": {"marks": 5, "target_words": "100-150", "max_tokens": 450},
    "10 Marks": {"marks": 10, "target_words": "200-350", "max_tokens": 900},
    "Detailed Answer": {"marks": None, "target_words": "300-500", "max_tokens": 1200},
}

# Max tokens per answer type (Groq) — kept tight so the model cannot ramble
# far beyond the target word range. Roughly: tokens ≈ words × 1.5 + overhead.
ANSWER_TYPE_MAX_TOKENS = {k: v["max_tokens"] for k, v in MARKS_CONFIG.items()}

# Adaptive top_k boost for exam modes (ensures enough context without overriding user's slider too much)
ANSWER_TYPE_TOP_K_BOOST = {
    "2 Marks": 0,      # concise → use exactly slider value
    "5 Marks": 1,      # +1 chunk for medium answer
    "10 Marks": 2,     # +2 chunks — needs broader context for full 10-mark structure
    "Detailed Answer": 2,
}

ANSWER_TYPE_INSTRUCTIONS = {
    "2 Marks": (
        "MARKS ALLOCATED: 2.\n"
        "Give a SHORT, direct, exam-oriented answer for a 2-mark question.\n"
        "Expected length: approximately 30-50 words (guideline — adjust slightly for question complexity).\n"
        "Required structure: the definition or main concept, plus one or two essential points only.\n"
        "Rules: stay relevant to the question; avoid introductions, examples, elaboration, "
        "extra sections (Advantages/Applications/Conclusion) and diagrams unless directly asked. "
        "STOP after ~50 words — do NOT expand into an essay.\n"
        "Grounding: use ONLY the retrieved document context; if insufficient, say "
        "\"I couldn't find this information in the uploaded documents.\" Do not invent facts."
    ),
    "5 Marks": (
        "MARKS ALLOCATED: 5.\n"
        "Give a MODERATELY detailed, well-structured, exam-oriented answer for a 5-mark question.\n"
        "Expected length: approximately 100-150 words (guideline — adjust slightly for question complexity).\n"
        "Required structure: definition/introduction, important points (headings or numbered points where helpful), "
        "and a relevant example only when useful and supported by context.\n"
        "Rules: stay relevant; do not inflate to a 10-mark essay and do not shrink to a 2-mark definition. "
        "Use headings/numbered points where appropriate.\n"
        "Grounding: use ONLY the retrieved document context; if insufficient, state the limitation instead of fabricating. "
        "A Mermaid diagram ONLY if it genuinely clarifies and is supported by context."
    ),
    "10 Marks": (
        "MARKS ALLOCATED: 10.\n"
        "Give a DETAILED, comprehensive, exam-oriented answer for a 10-mark question.\n"
        "Expected length: approximately 200-350 words (guideline — adjust slightly for question complexity).\n"
        "Required structure (use headings and numbered points for readability; omit any section not supported by context):\n"
        "1. ### Introduction / Definition\n"
        "2. ### Explanation (core idea, components/types)\n"
        "3. ### Important Points / Working / Process / Steps (as applicable)\n"
        "4. ### Relevant Examples / Applications (only from context)\n"
        "5. ### Advantages / Limitations (ONLY if mentioned in context; do NOT invent)\n"
        "6. ### Diagram — ONLY if the question benefits from a visual AND context supports it, emit:\n"
        "```mermaid\n"
        "flowchart TD\n"
        "    A[User] --> B[Input]\n"
        "    B --> C[Processing]\n"
        "    C --> D[Output]\n"
        "```\n"
        "Keep diagrams simple (4-8 nodes, labels strictly from context); otherwise omit entirely.\n"
        "7. ### Conclusion (2-3 line synthesis when appropriate)\n"
        "Rules: stay relevant to the question; highlight key terms with **bold**; avoid repetition. "
        "Grounding: use ONLY retrieved context; never invent facts."
    ),
    "Detailed Answer": (
        "You are in **DETAILED ANSWER MODE**.\n"
        "Give a comprehensive, well-structured explanation based ONLY on the retrieved context (~400–600 words).\n"
        "- Cover definition, explanation, key points, process/working, examples, applications as supported by context.\n"
        "- Use clear headings (###), bullet/numbered lists, bold key terms.\n"
        "- If a diagram would help understanding (architecture, workflow, block diagram, etc.), include a Mermaid flowchart/block diagram "
        "based ONLY on context using:\n"
        "```mermaid\n"
        "flowchart TD\n"
        "    A[...] --> B[...]\n"
        "```\n"
        "- Be thorough but do NOT hallucinate. Omit sections lacking context support."
    ),
}


def normalize_answer_type(answer_type) -> str:
    """Canonicalize user-selected marks to one of ANSWER_TYPES.

    Accepts variants like '2', '2 marks', '2-marks', lowercase, extra spaces.
    Unknown/empty values safely default to 'Detailed Answer'.
    """
    if answer_type is None:
        return "Detailed Answer"
    s = str(answer_type).strip().lower().replace("-", " ").replace("_", " ")
    s = " ".join(s.split())
    if s in ("2", "2 mark", "2 marks"):
        return "2 Marks"
    if s in ("5", "5 mark", "5 marks"):
        return "5 Marks"
    if s in ("10", "10 mark", "10 marks"):
        return "10 Marks"
    if s in ("detailed", "detailed answer", "detail", "long"):
        return "Detailed Answer"
    # Already-canonical (case-insensitive) check
    for valid in ANSWER_TYPES:
        if s == valid.lower():
            return valid
    return "Detailed Answer"


def get_answer_type_instructions(answer_type: str) -> str:
    """Return exam-mode instructions for the given answer type (normalized)."""
    return ANSWER_TYPE_INSTRUCTIONS.get(
        normalize_answer_type(answer_type), ANSWER_TYPE_INSTRUCTIONS["Detailed Answer"]
    )


def get_max_tokens_for_answer_type(answer_type: str) -> int:
    """Return max_tokens tuned for the answer type (normalized)."""
    return ANSWER_TYPE_MAX_TOKENS.get(normalize_answer_type(answer_type), 900)


def generate_answer(
    query: str,
    retrieved_chunks: List[Dict],
    stream: bool = False,
    answer_type: str = "Detailed Answer"
) -> Tuple[str, str]:
    """
    Generate answer using Groq LLM with retrieved context.

    Returns:
        answer: str
        via: str (model name or error flag)
    """
    if not retrieved_chunks:
        return (
            "I couldn't find this information in the uploaded documents.",
            "no-context"
        )

    context = build_context(retrieved_chunks)
    client = get_groq_client()
    model = get_groq_model()

    # If no Groq client, fallback to extractive (show retrieved chunks)
    if client is None:
        fallback = (
            "⚠️ Groq API key not configured. Showing most relevant excerpts instead.\n\n"
            + "\n\n".join([f"**{_format_source(c)}** (score {c.get('score','')})\n{c['text'][:600]}" for c in retrieved_chunks])
            + "\n\n---\n*Configure GROQ_API_KEY in .env or Streamlit secrets to enable LLM generation.*"
        )
        return fallback, "extractive-no-key"

    # Normalize first so UI variants / stale values can never trigger a wrong length
    answer_type = normalize_answer_type(answer_type)
    # Build answer-type specific instructions
    exam_instructions = get_answer_type_instructions(answer_type)
    max_tokens = get_max_tokens_for_answer_type(answer_type)
    cfg = MARKS_CONFIG[answer_type]

    # Effective system prompt combines base rules + exam-mode instructions
    effective_system_prompt = SYSTEM_PROMPT + "\n\n" + exam_instructions

    # Build messages — marks value is explicit so the LLM cannot ignore it
    marks_label = f"{cfg['marks']} marks" if cfg["marks"] else "detailed answer (no fixed marks)"
    user_prompt = (
        f"Context (retrieved from documents — use ONLY this):\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Marks allocated: {marks_label}. Answer Type: {answer_type}. "
        f"Target length: ~{cfg['target_words']} words (guideline).\n"
        f"Instructions: {effective_system_prompt}\n\n"
        "Provide the answer now. Match the marks exactly: do not expand a short "
        "answer into a long essay, and do not shrink a long answer into a definition. "
        "If you use information from context, you may mention the source file/page/slide."
    )

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": effective_system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=max_tokens,
            top_p=0.9,
        )
        answer = completion.choices[0].message.content or ""
        answer = answer.strip()
        if not answer:
            raise ValueError("Empty response from LLM")
        # Strip <think> tags if any (some Groq models)
        import re
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
        return answer, f"groq:{model}"

    except Exception as e:
        err_msg = str(e)
        # Detect model not found
        if "model" in err_msg.lower() and ("not found" in err_msg.lower() or "does not exist" in err_msg.lower() or "decommissioned" in err_msg.lower()):
            return (
                f"❌ Groq model '{model}' is unavailable or decommissioned.\n\n"
                f"Details: {err_msg}\n\n"
                "Fix: Set a valid model in your `.env` file:\n"
                "`GROQ_MODEL=openai/gpt-oss-120b`\n"
                "Other valid options (verified 2026-08-29, check https://console.groq.com/docs/models): "
                "`openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `allam-2-7b`, `groq/compound`, `qwen/qwen3-32b`.\n\n"
                "After changing .env (or Streamlit secrets), restart the app. Run `python -c \"from groq import Groq; print([m.id for m in Groq().models.list().data])\"` to list live models.",
                f"error:model-unavailable:{model}"
            )
        # Invalid API key
        if "api_key" in err_msg.lower() or "401" in err_msg or "unauthorized" in err_msg.lower() or "invalid" in err_msg.lower():
            # Do not leak key
            return (
                "❌ Groq API key is missing or invalid. Please check your `GROQ_API_KEY` in `.env` or Streamlit secrets. "
                "Get a key at https://console.groq.com/keys",
                "error:invalid-key"
            )
        # Network / rate limit etc -> fallback extractive
        fallback = (
            f"⚠️ Groq API error ({err_msg[:200]}). Showing extracted relevant passages:\n\n"
            + "\n\n---\n\n".join([f"**{_format_source(c)}**\n{c['text'][:600]}" for c in retrieved_chunks])
        )
        return fallback, f"extractive-fallback:{model}"


def answer_question(
    query: str,
    chunks: List[Dict],
    index,
    embedding_model,
    top_k: int = 4,
    answer_type: str = "Detailed Answer"
) -> Dict:
    """
    Full RAG flow: embed query → search FAISS → generate answer → sources.
    This function is called per question; it does NOT rebuild index.
    answer_type controls exam-mode formatting (2/5/10 Marks, Detailed Answer).
    """
    from utils.embeddings import embed_query
    from utils.vector_store import search_faiss

    # Validate
    if not query or not query.strip():
        return {"answer": "Please enter a question.", "sources": [], "contexts": [], "via": "error:empty-query"}
    if not chunks or index is None or index.ntotal == 0:
        return {"answer": "No documents indexed. Please upload documents first and wait for processing.", "sources": [], "contexts": [], "via": "error:no-index"}
    if embedding_model is None:
        return {"answer": "Embedding model not loaded. Please refresh the app.", "sources": [], "contexts": [], "via": "error:no-embedding-model"}

    # Normalize marks selection FIRST — guarantees the whole pipeline uses one value
    answer_type = normalize_answer_type(answer_type)

    # Adaptive top_k for exam modes: 10 Marks / Detailed need broader context
    boost = ANSWER_TYPE_TOP_K_BOOST.get(answer_type, 0)
    effective_top_k = top_k + boost
    # Clamp to available chunks
    k = min(max(1, effective_top_k), len(chunks))
    if k < effective_top_k:
        pass

    # Embed query
    try:
        q_emb = embed_query(query, embedding_model)
    except Exception as e:
        return {"answer": f"Failed to embed query: {e}", "sources": [], "contexts": [], "via": "error:embed-failed"}

    # Retrieve
    try:
        contexts = search_faiss(q_emb, index, chunks, top_k=k)
    except Exception as e:
        return {"answer": f"Retrieval failed: {e}", "sources": [], "contexts": [], "via": "error:faiss"}

    if not contexts:
        return {
            "answer": "I couldn't find this information in the uploaded documents.",
            "sources": [],
            "contexts": [],
            "via": "no-results"
        }

    # Generate — pass answer_type for exam-mode formatting
    answer, via = generate_answer(query, contexts, answer_type=answer_type)

    sources = deduplicate_sources(contexts)

    return {
        "answer": answer,
        "sources": sources,
        "contexts": contexts,
        "via": via,
        "answer_type": answer_type,
    }
