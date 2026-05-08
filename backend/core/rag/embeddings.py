"""
VidIntel AI — Embeddings
Wraps sentence-transformers (BGE small) to produce dense vectors.
Singleton pattern to avoid reloading the model on every call.
"""

from __future__ import annotations

from typing import List, Optional
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL


_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[Embeddings] Loading model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.
    Returns list of float lists (one per input text).
    """
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        normalize_embeddings=True,      # L2-normalize for cosine similarity
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """Embed a single query string."""
    # BGE models benefit from a query prefix
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    model = get_embedding_model()
    vec = model.encode(prefixed, normalize_embeddings=True, convert_to_numpy=True)
    return vec.tolist()
