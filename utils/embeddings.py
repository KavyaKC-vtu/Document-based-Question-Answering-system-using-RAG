"""
Embedding utilities with Streamlit caching.

- Uses SentenceTransformer("all-MiniLM-L6-v2")
- Caches model load with @st.cache_resource to load only once
- Provides batch embedding (float32) for FAISS
"""

import os
import streamlit as st
import numpy as np
from typing import List


EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Load SentenceTransformer model once and cache it.
    Uses Streamlit's cache_resource so it survives reruns.

    Optimization: Model is loaded ONLY ONCE and reused for every
    rerun / question / mark-change. Never call SentenceTransformer()
    inside a loop.

    Spec-compliant canonical form:
        @st.cache_resource
        def load_embedding_model():
            return SentenceTransformer("all-MiniLM-L6-v2")
    This implementation keeps an optional model_name param for
    backward compatibility but defaults to all-MiniLM-L6-v2.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers not installed. Run: pip install sentence-transformers"
        ) from e

    # This will download model on first run (~80MB) and cache thereafter
    model = SentenceTransformer(model_name)
    return model


def get_embedding_dimension(model) -> int:
    """Return embedding dimension."""
    try:
        return model.get_sentence_embedding_dimension()
    except Exception:
        # Fallback for MiniLM-L6-v2
        return 384


def embed_texts(texts: List[str], model, batch_size: int = 64) -> np.ndarray:
    """
    Batch embed texts to float32 numpy array.

    OPTIMIZED PIPELINE (spec 3 & 4):
    - Generates embeddings ONCE for ALL chunks from ALL PDFs together.
    - Uses batched model.encode with batch_size=64 and progress bar.
    - Falls back to 32 if 64 causes OOM.

    Returns: np.ndarray shape (n, dim) dtype float32
    """
    if not texts:
        dim = get_embedding_dimension(model)
        return np.zeros((0, dim), dtype=np.float32)

    # Try batch_size=64 first (spec), fallback to 32 on memory error
    for bs in (batch_size, 32) if batch_size == 64 else (batch_size,):
        try:
            embeddings = model.encode(
                texts,
                batch_size=bs,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)
            return embeddings
        except RuntimeError as e:
            # CUDA OOM or memory pressure — retry with smaller batch
            if "memory" in str(e).lower() or "oom" in str(e).lower():
                if bs != 32:
                    continue
            raise
        except Exception as e:
            # Generic fallback for memory-related errors
            if bs == 64 and ("memory" in str(e).lower() or "allocation" in str(e).lower()):
                continue
            raise
    # Fallback exhausted (should not reach here)
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    return embeddings


def embed_texts_with_progress(
    texts: List[str],
    model,
    batch_size: int = 64,
    progress_bar=None,
    status_text=None,
) -> np.ndarray:
    """
    Generate embeddings in batches with real progress updates.

    Spec: batch_size=64 with fallback to 32 + show_progress_bar=True.
    Processes ALL chunks together (not per-PDF) and updates
    st.progress() after every batch — required for 20MB / 94-page PDFs.

    Args:
        texts: list of chunk texts (ALL PDFs combined)
        model: cached SentenceTransformer instance (load once via load_embedding_model)
        batch_size: batch size for model.encode (default 64, fallback 32)
        progress_bar: st.progress object to update (optional)
        status_text: st.empty/status object for text updates (optional)

    Returns:
        np.ndarray shape (n, dim) dtype float32
    """
    if not texts:
        dim = get_embedding_dimension(model)
        return np.zeros((0, dim), dtype=np.float32)

    n = len(texts)
    dim = get_embedding_dimension(model)

    # Auto fallback: try 64, if OOM then 32
    effective_batch = batch_size
    # Pre-allocate list of batch embeddings; vstack at end avoids repeated copies
    batch_embeddings = []
    num_batches = (n + effective_batch - 1) // effective_batch

    for batch_idx, start in enumerate(range(0, n, effective_batch)):
        end = min(start + effective_batch, n)
        batch = texts[start:end]
        try:
            # Batch encode — model is loaded once via @st.cache_resource, not per batch
            emb = model.encode(
                batch,
                batch_size=effective_batch,
                show_progress_bar=False,  # we provide custom st.progress per batch
                convert_to_numpy=True
            )
        except RuntimeError as e:
            if effective_batch == 64 and ("memory" in str(e).lower() or "oom" in str(e).lower()):
                # Fallback to 32 and restart from current batch with smaller size
                # For simplicity, re-encode remaining with 32 and continue
                effective_batch = 32
                num_batches = (n + effective_batch - 1) // effective_batch
                # Recompute batch slicing for fallback: re-encode this batch with 32
                emb = model.encode(
                    batch,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
            else:
                raise
        if emb.dtype != np.float32:
            emb = emb.astype(np.float32)
        batch_embeddings.append(emb)

        # Real progress update per batch
        if progress_bar is not None:
            try:
                frac = end / n
                try:
                    progress_bar.progress(frac, text=f"Generating embeddings... {end}/{n} (batch {batch_idx+1}/{num_batches})")
                except TypeError:
                    progress_bar.progress(frac)
                if status_text is not None:
                    status_text.info(f"Generating embeddings... {end}/{n} chunks (batch {batch_idx+1}/{num_batches})")
            except Exception:
                pass

    # Concatenate all batches efficiently
    if len(batch_embeddings) == 1:
        embeddings = batch_embeddings[0]
    else:
        embeddings = np.vstack(batch_embeddings)

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    assert embeddings.shape[0] == n, f"Expected {n} embeddings, got {embeddings.shape[0]}"
    assert embeddings.shape[1] == dim, f"Expected dim {dim}, got {embeddings.shape[1]}"
    return embeddings


def embed_query(query: str, model) -> np.ndarray:
    """Embed single query to (1, dim) float32."""
    emb = model.encode([query], normalize_embeddings=False, convert_to_numpy=True)
    return emb.astype(np.float32)
