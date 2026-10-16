"""
Multi-format document text extraction + chunking with metadata preservation.

Supported formats:
- .pdf  -> PyMuPDF (fitz) per page, page_number = page index+1, file_type=pdf
- .docx -> python-docx per paragraph/table, page_number = 1, file_type=docx
- .txt  -> standard Python file reading (utf-8 fallback latin-1), file_type=txt
- .pptx -> python-pptx per slide, page_number = slide number, file_type=pptx
- .csv  -> pandas per row as text, page_number = 1 (or row range), file_type=csv

Each chunk retains: text, filename, page_number, file_type, source_label
Existing pipeline (embeddings -> FAISS) unchanged — single shared index.
"""

import io
import os
from typing import List, Dict, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".pptx", ".csv"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_type_from_name(filename: str) -> str:
    _, ext = os.path.splitext(filename)
    return ext.lower().lstrip(".") if ext else "unknown"


def _make_source_label(filename: str, file_type: str, page_number: int) -> str:
    """Human-readable citation label."""
    ft = file_type.lower()
    if ft == "pdf":
        return f"{filename} | Page {page_number}"
    if ft == "pptx":
        return f"{filename} | Slide {page_number}"
    if ft in ("docx", "txt", "csv"):
        # For these, page/slide not applicable — but keep Page 1 for consistent dedup,
        # display as just filename or with marker
        # PPTX case above handles slide, pdf page, others show file reference
        # Use generic label without page for cleaner citations, but keep page_number for internal tracking
        return f"{filename} | {ft.upper()}"
    return f"{filename} | Page {page_number}"


# ---------------------------------------------------------------------------
# Extractors per format
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes, filename: str) -> Tuple[List[Dict], str]:
    import fitz  # PyMuPDF
    pages_data = []
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        return [], f"Failed to open PDF '{filename}': {e}. The file may be corrupted or not a valid PDF."
    if len(doc) == 0:
        return [], f"PDF '{filename}' has no pages."
    for page_num in range(len(doc)):
        try:
            page = doc[page_num]
            text = page.get_text("text")
            text = text.strip() if text else ""
            if text:
                pages_data.append({
                    "text": text,
                    "filename": filename,
                    "page_number": page_num + 1,
                    "file_type": "pdf",
                })
        except Exception:
            continue
    doc.close()
    if not pages_data:
        return [], (
            f"No extractable text found in '{filename}'. "
            "This may be a scanned/image-only PDF. "
            "Try uploading a text-based PDF or use OCR to convert it first."
        )
    return pages_data, ""


def extract_text_from_docx(file_bytes: bytes, filename: str) -> Tuple[List[Dict], str]:
    try:
        from docx import Document
    except ImportError:
        return [], f"python-docx not installed. Cannot process '{filename}'. Run: pip install python-docx"
    try:
        doc = Document(io.BytesIO(file_bytes))
    except Exception as e:
        return [], f"Failed to open DOCX '{filename}': {e}. The file may be corrupted or not a valid DOCX."

    texts = []
    # Paragraphs
    for para in doc.paragraphs:
        t = para.text.strip()
        if t:
            texts.append(t)
    # Tables
    for table in doc.tables:
        for row in table.rows:
            row_texts = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_texts:
                texts.append(" | ".join(row_texts))

    if not texts:
        return [], f"No extractable text found in '{filename}' (DOCX appears empty)."

    full_text = "\n\n".join(texts)
    return [{
        "text": full_text,
        "filename": filename,
        "page_number": 1,
        "file_type": "docx",
    }], ""


def extract_text_from_txt(file_bytes: bytes, filename: str) -> Tuple[List[Dict], str]:
    text = ""
    # Try utf-8 then fallback
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            text = file_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = ""
    text = text.strip()
    if not text:
        return [], f"No extractable text found in '{filename}' (TXT appears empty)."
    return [{
        "text": text,
        "filename": filename,
        "page_number": 1,
        "file_type": "txt",
    }], ""


def extract_text_from_pptx(file_bytes: bytes, filename: str) -> Tuple[List[Dict], str]:
    try:
        from pptx import Presentation
    except ImportError:
        return [], f"python-pptx not installed. Cannot process '{filename}'. Run: pip install python-pptx"
    try:
        prs = Presentation(io.BytesIO(file_bytes))
    except Exception as e:
        return [], f"Failed to open PPTX '{filename}': {e}. The file may be corrupted or not a valid PPTX."

    slides_data: List[Dict] = []
    for idx, slide in enumerate(prs.slides, start=1):
        texts = []
        for shape in slide.shapes:
            # Text frames
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    t = para.text.strip()
                    if t:
                        texts.append(t)
            # Tables inside shapes
            if shape.has_table:
                for row in shape.table.rows:
                    row_texts = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_texts:
                        texts.append(" | ".join(row_texts))
            # Group shapes: recurse one level
            if shape.shape_type == 6:  # group
                try:
                    for subshape in shape.shapes:
                        if subshape.has_text_frame:
                            for para in subshape.text_frame.paragraphs:
                                t = para.text.strip()
                                if t:
                                    texts.append(t)
                except Exception:
                    pass
        slide_text = "\n".join(texts).strip()
        if slide_text:
            slides_data.append({
                "text": slide_text,
                "filename": filename,
                "page_number": idx,  # slide number
                "file_type": "pptx",
            })
    if not slides_data:
        return [], f"No extractable text found in '{filename}' (PPTX appears empty or contains only images)."
    return slides_data, ""


def extract_text_from_csv(file_bytes: bytes, filename: str) -> Tuple[List[Dict], str]:
    try:
        import pandas as pd
    except ImportError:
        return [], f"pandas not installed. Cannot process '{filename}'. Run: pip install pandas"

    # Decode bytes for pandas
    # Try to handle empty
    if len(file_bytes) == 0:
        return [], f"Skipped '{filename}': file is empty (0 bytes)."

    # pandas read_csv from BytesIO; try different encodings via python engine
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8", engine="python", on_bad_lines="skip")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1", engine="python", on_bad_lines="skip")
        except Exception as e:
            return [], f"Failed to read CSV '{filename}': {e}."
    except pd.errors.EmptyDataError:
        return [], f"No extractable text found in '{filename}' (CSV is empty)."
    except Exception as e:
        return [], f"Failed to read CSV '{filename}': {e}. The file may be corrupted or not a valid CSV."

    if df.empty:
        return [], f"No extractable text found in '{filename}' (CSV has no data rows)."

    # Build text representation: header + rows as sentences
    # Keep within reasonable size; convert each row to "col: value, ..."
    texts = []
    # Include header
    header = ", ".join([str(c) for c in df.columns])
    texts.append(f"CSV File: {filename} | Columns: {header}")

    # Convert rows — limit to maybe first 1000 rows to avoid huge memory? But spec says preserve all
    # We'll include all rows, each row as a line
    for idx, row in df.iterrows():
        # Row as "col1: val1 | col2: val2 ..."
        row_parts = []
        for col in df.columns:
            val = row[col]
            # Handle NaN
            try:
                import pandas as pd2
                if pd2.isna(val):
                    continue
            except Exception:
                pass
            row_parts.append(f"{col}: {val}")
        if row_parts:
            texts.append(" | ".join(row_parts))

    if not texts:
        return [], f"No extractable text found in '{filename}' (CSV rows empty)."

    full_text = "\n".join(texts)
    return [{
        "text": full_text,
        "filename": filename,
        "page_number": 1,
        "file_type": "csv",
    }], ""


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_pages(
    pages_data: List[Dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[Dict]:
    """
    Chunk text while keeping metadata (filename, page_number, file_type).
    page_number is per-page for PDF, per-slide for PPTX, 1 for others.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks: List[Dict] = []
    for page in pages_data:
        split_texts = splitter.split_text(page["text"])
        for ct in split_texts:
            ct = ct.strip()
            if len(ct) < 30:
                continue
            chunks.append({
                "text": ct,
                "filename": page["filename"],
                "page_number": page.get("page_number", 1),
                "file_type": page.get("file_type", _file_type_from_name(page["filename"])),
            })
    return chunks


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def extract_text_from_file(file_bytes: bytes, filename: str) -> Tuple[List[Dict], str]:
    """Route to correct extractor based on extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes, filename)
    if ext == ".docx":
        return extract_text_from_docx(file_bytes, filename)
    if ext == ".txt":
        return extract_text_from_txt(file_bytes, filename)
    if ext == ".pptx":
        return extract_text_from_pptx(file_bytes, filename)
    if ext == ".csv":
        return extract_text_from_csv(file_bytes, filename)
    return [], f"Skipped '{filename}': unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."


def process_uploaded_documents(
    uploaded_files,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> Tuple[List[Dict], List[str]]:
    """
    Process multiple Streamlit UploadedFile objects of mixed formats.
    Returns combined chunks + messages.
    Single FAISS index will be built downstream — embeddings generated ONCE for all chunks.
    """
    all_chunks: List[Dict] = []
    messages: List[str] = []

    if not uploaded_files:
        return [], ["No files uploaded."]

    for uf in uploaded_files:
        filename = uf.name
        ext = os.path.splitext(filename)[1].lower()

        # Validate extension first
        if ext not in SUPPORTED_EXTENSIONS:
            messages.append(f"Skipped '{filename}': unsupported file type '{ext}'. Supported: .pdf, .docx, .txt, .pptx, .csv")
            continue

        try:
            file_bytes = uf.getvalue()
            if len(file_bytes) == 0:
                messages.append(f"Skipped '{filename}': file is empty (0 bytes).")
                continue
            pages_data, err = extract_text_from_file(file_bytes, filename)
            if err:
                messages.append(err)
                continue
            chunks = chunk_pages(pages_data, chunk_size, chunk_overlap)
            if not chunks:
                messages.append(f"No chunks created from '{filename}': text too short or unreadable.")
                continue
            all_chunks.extend(chunks)
            # Message preserving page/slide info
            ft = _file_type_from_name(filename)
            if ft == "pptx":
                slides = len(pages_data)
                messages.append(f"✓ '{filename}': {slides} slides, {len(chunks)} chunks.")
            elif ft == "pdf":
                pages = len(pages_data)
                messages.append(f"✓ '{filename}': {pages} pages, {len(chunks)} chunks.")
            else:
                messages.append(f"✓ '{filename}': {len(chunks)} chunks ({ft.upper()}).")
        except Exception as e:
            messages.append(f"Error processing '{filename}': {type(e).__name__}: {e}")

    return all_chunks, messages


# Alias for backward compatibility — existing app.py imported process_uploaded_pdfs
def process_uploaded_pdfs(
    uploaded_files,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> Tuple[List[Dict], List[str]]:
    """
    Backward-compatible wrapper. Previously PDF-only. Now delegates to
    process_uploaded_documents so PDF flow is preserved while allowing mixed formats
    if called with non-PDF files (previously would skip).
    """
    # Check if all files are PDFs — preserve strict PDF messages if so
    # Otherwise delegate fully (so old code won't break when adding .docx etc.)
    # For now simply delegate — PDF handling is identical.
    return process_uploaded_documents(uploaded_files, chunk_size, chunk_overlap)
