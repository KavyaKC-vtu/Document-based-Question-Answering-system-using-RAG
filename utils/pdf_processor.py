"""
PDF text extraction + chunking with metadata preservation.

Preserved + Extended for multi-format RAG:
- Original PDF extraction via PyMuPDF unchanged (backward compatible)
- Now also supports .docx, .txt, .pptx, .csv via document_processor dispatcher
- process_uploaded_pdfs remains for backward compat but now delegates to
  process_uploaded_documents (so mixed PDF+DOCX etc. works even via old import)

For new code, prefer: from utils.document_processor import process_uploaded_documents
"""

from utils.document_processor import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_txt,
    extract_text_from_pptx,
    extract_text_from_csv,
    extract_text_from_file,
    chunk_pages,
    process_uploaded_documents,
    SUPPORTED_EXTENSIONS,
)

# Re-export for backward compatibility
__all__ = [
    "extract_text_from_pdf",
    "extract_text_from_docx",
    "extract_text_from_txt",
    "extract_text_from_pptx",
    "extract_text_from_csv",
    "extract_text_from_file",
    "chunk_pages",
    "process_uploaded_documents",
    "process_uploaded_pdfs",
    "SUPPORTED_EXTENSIONS",
]


def process_uploaded_pdfs(
    uploaded_files,
    chunk_size: int = 500,
    chunk_overlap: int = 100
):
    """
    Backward-compatible wrapper — delegates to process_uploaded_documents.
    Previously PDF-only; now supports all formats while preserving PDF page-number citations.
    """
    return process_uploaded_documents(uploaded_files, chunk_size, chunk_overlap)
