"""
FAISS vector store utilities.

- Converts embeddings to float32
- Uses IndexFlatL2 (Euclidean) — appropriate for raw embeddings
- Also supports cosine via normalized IP if needed (we keep L2 as spec says IndexFlatL2)
- Stores chunks + index in Streamlit session state (not rebuilt per question)
"""

import faiss
import numpy as np
from typing import List, Dict, Tuple


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Create FAISS IndexFlatL2 and add embeddings.

    Args:
        embeddings: np.ndarray (n, dim) float32

    Returns:
        faiss.Index
    """
    if embeddings.size == 0:
        raise ValueError("No embeddings to index (0 chunks).")

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    n, dim = embeddings.shape
    # IndexFlatL2: exact L2 search, good for small-medium datasets (<100k)
    index = faiss.IndexFlatL2(dim)
    # Optional: if you prefer cosine, normalize and use IndexFlatIP
    # faiss.normalize_L2(embeddings)  # then IndexFlatIP

    # Ensure contiguous
    embeddings = np.ascontiguousarray(embeddings)
    index.add(embeddings)
    return index


def search_faiss(
    query_embedding: np.ndarray,
    index: faiss.Index,
    chunks: List[Dict],
    top_k: int = 4
) -> List[Dict]:
    """
    Search FAISS for top-k relevant chunks.

    Handles edge: top_k > len(chunks) -> clamp.
    Returns list of chunks with added 'score' (L2 distance) and 'rank'.

    Note: For L2, smaller distance = more similar.
    We convert to a pseudo-score for display: score = 1 / (1 + distance)
    """
    if index is None or index.ntotal == 0:
        return []
    if not chunks:
        return []
    if query_embedding.dtype != np.float32:
        query_embedding = query_embedding.astype(np.float32)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    query_embedding = np.ascontiguousarray(query_embedding)

    # Clamp top_k
    k = min(top_k, len(chunks), index.ntotal)
    if k <= 0:
        return []

    try:
        distances, indices = index.search(query_embedding, k)
    except Exception as e:
        raise RuntimeError(f"FAISS search failed: {e}") from e

    results: List[Dict] = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        if idx == -1:
            continue
        if idx >= len(chunks):
            continue
        chunk = chunks[idx]
        # Convert L2 distance to intuitive score (higher is better for UI)
        # For display we keep both distance and similarity
        similarity = float(1 / (1 + dist))
        results.append({
            **chunk,
            "score": round(similarity, 4),
            "distance": round(float(dist), 4),
            "rank": rank + 1
        })
    return results


def get_index_stats(index: faiss.Index, chunks: List[Dict]) -> Dict:
    """Return stats for UI display."""
    if index is None:
        return {"vectors": 0, "dimension": 0, "indexed": False}
    try:
        dim = index.d
        ntotal = index.ntotal
        return {
            "vectors": int(ntotal),
            "dimension": int(dim),
            "indexed": ntotal > 0,
            "chunks": len(chunks)
        }
    except Exception:
        return {"vectors": 0, "dimension": 0, "indexed": False}
