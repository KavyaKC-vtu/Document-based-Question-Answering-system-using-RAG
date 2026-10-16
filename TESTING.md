# Testing — AI Document QA RAG

Manual test checklist. Perform each case after `streamlit run app.py`.

## Prerequisites

- `pip install -r requirements.txt`
- `.env` with `GROQ_API_KEY` and `GROQ_MODEL=openai/gpt-oss-120b` (fallback `openai/gpt-oss-120b`, see .env.example)
- Sample PDFs: use `sample_docs/*.txt` converted to PDF, or any text-based PDF

---

## Test Cases

### 1) Single PDF Upload
- **Steps:** Upload 1 PDF (e.g., `ai_overview.txt` as PDF or any course PDF) → click Process
- **Expected:** Toast `✓ 'file.pdf': N pages, M chunks`, stats show `PDFs=1`, `Chunks>0`, `Vectors>0`, `Dim=384`, success message, processed log expander

### 2) Multiple PDF Upload
- **Steps:** Upload 3 PDFs at once → Process
- **Expected:** Each file logged `✓`, total chunks = sum, vectors = chunks, sidebar shows `PDFs=3`

### 3) Valid Question — Answer Present
- **Steps:** With `ai_overview.pdf` indexed, ask: “What is Machine Learning?”
- **Expected:** Grounded answer from doc, cites `📄 ai_overview.pdf | Page 1`, expander shows retrieved chunks containing “Machine Learning” text

### 4) Question with Answer Absent
- **Steps:** Ask unrelated: “Who won the 3025 World Cup?”
- **Expected:** “I couldn't find this information in the uploaded documents.”, no hallucinated facts

### 5) Empty PDF
- **Steps:** Upload a PDF with 0 extractable pages (e.g., create blank PDF)
- **Expected:** Warning “No extractable text found… scanned/image-only… Try OCR”, no index built, error not crashing app

### 6) Invalid PDF (corrupted / .txt renamed)
- **Steps:** Upload a `.txt` renamed to `.pdf` or corrupted bytes
- **Expected:** “Failed to open PDF… may be corrupted…” or “Skipped: only PDF allowed”, graceful error

### 7) API Failure — Missing Key
- **Steps:** Rename `.env` temporarily, restart app, ask question
- **Expected:** Sidebar shows `GROQ_API_KEY: missing ✗`, answer fallback: “Groq API key not configured… Showing excerpts…”, app not crash, no key leaked in error

### 8) API Failure — Invalid Model
- **Steps:** Set `GROQ_MODEL=nonexistent-model-xyz` in `.env`, restart, ask question
- **Expected:** Clear error: “Groq model 'nonexistent-model-xyz' unavailable… Set GROQ_MODEL=openai/gpt-oss-120b…”, with link to docs, suggests valid models

### 9) Multiple Questions (Chat History)
- **Steps:** Ask Q1, then Q2, then Q3 without reprocessing
- **Expected:** Chat history shows 3 turns (most recent first), each with own answer + sources + via tag, `Clear Chat History` removes all but keeps index

### 10) Chat History & Clear Chat
- **Steps:** After 2 questions, click `Clear Chat` (main) → ask again, click `Clear All` (sidebar)
- **Expected:** `Clear Chat` clears history only, `Clear All` clears history + index + file names, stats reset to 0

### 11) Duplicate File Upload
- **Steps:** Upload `sample.pdf` → Process → upload same `sample.pdf` again (same bytes) → no Process click needed
- **Expected:** Hash detects no change → shows “Already processed…”, does NOT re-embed (performance check), vectors unchanged

### 12) Large PDF
- **Steps:** Upload a 30+ page PDF (>5k words)
- **Expected:** Chunking produces many chunks (e.g., 60+), embeddings batched (no hang), FAISS builds, stats reflect large vectors, top-k retrieval still fast (<1s)

---

## Automated Quick Sanity

```bash
# Imports
python -c "import fitz, faiss, sentence_transformers, streamlit, groq, langchain; print('imports OK')"

# PDF processor unit
python -c "
from utils.pdf_processor import chunk_pages
from langchain_text_splitters import RecursiveCharacterTextSplitter
pages=[{'text':'Hello world. '*50,'filename':'test.pdf','page_number':1}]
from utils.pdf_processor import chunk_pages
print(len(chunk_pages(pages,500,100)), 'chunks')
"

# FAISS
python -c "
import numpy as np, faiss
from utils.vector_store import build_faiss_index, search_faiss
emb=np.random.randn(5,384).astype('float32')
idx=build_faiss_index(emb)
ch=[{'text':f'chunk {i}','filename':'f.pdf','page_number':1} for i in range(5)]
print(search_faiss(emb[0:1], idx, ch, top_k=2))
"
```

## Pass Criteria

- No unhandled exceptions (check terminal)
- No API key in UI errors
- Sources deduplicated (ask question spanning same page → one badge per page)
- FAISS not rebuilt per question (watch logs/spinner only on Process, not on Ask)
- Model loads once (first Ask may be slow due to 80MB download, second Ask instant)
