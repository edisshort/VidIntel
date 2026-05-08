"""
VidIntel AI — Hybrid Retriever
Combines BM25 (keyword) and ChromaDB (semantic) results using
Reciprocal Rank Fusion (RRF) — a robust, parameter-light fusion method.

alpha controls the blend:
  alpha=0.0 → pure BM25
  alpha=1.0 → pure vector
  alpha=0.5 → equal weight (default)

Why RRF?
  RRF is more stable than score normalisation because it only uses ranks,
  not raw scores (which differ in scale between BM25 and cosine similarity).
"""

from typing import List, Optional

from config import HYBRID_ALPHA, DEFAULT_TOP_K
from core.rag.vector_store import semantic_search
from core.rag.bm25_retriever import bm25_search


# ─── RRF ───────────────────────────────────────────────────────────────────────

def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score."""
    return 1.0 / (k + rank)


def reciprocal_rank_fusion(
    vector_results: List[dict],
    bm25_results: List[dict],
    alpha: float = 0.5,
    top_k: int = 5,
) -> List[dict]:
    """
    Merge two ranked lists using RRF.
    alpha weights vector vs. BM25 contribution.
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    # Key = text fingerprint (first 120 chars)
    def key(d: dict) -> str:
        return d["text"][:120]

    # Vector results
    for rank, result in enumerate(vector_results):
        k_ = key(result)
        scores[k_] = scores.get(k_, 0) + alpha * _rrf_score(rank)
        if k_ not in docs:
            docs[k_] = result

    # BM25 results
    for rank, result in enumerate(bm25_results):
        k_ = key(result)
        scores[k_] = scores.get(k_, 0) + (1 - alpha) * _rrf_score(rank)
        if k_ not in docs:
            docs[k_] = result

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    merged = []
    for k_, score in ranked[:top_k]:
        doc = dict(docs[k_])
        doc["score"] = round(score, 6)
        doc["retrieval_type"] = "hybrid"
        merged.append(doc)

    return merged


# ─── Public API ────────────────────────────────────────────────────────────────

def hybrid_search(
    collection_name: str,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    alpha: float = HYBRID_ALPHA,
    source_filter: Optional[str] = None,  # "transcript" | "ocr" | None
) -> List[dict]:
    """
    Perform hybrid retrieval over a VidIntel collection.

    Args:
        collection_name: ChromaDB/BM25 collection to search.
        query:           User's natural language question.
        top_k:           Number of results to return.
        alpha:           Vector weight (0 = BM25 only, 1 = vector only).
        source_filter:   Optional filter to restrict to transcript or OCR docs.

    Returns:
        Ranked list of chunks with metadata and scores.
    """
    fetch_k = top_k * 3   # Over-fetch before fusion

    where_filter = None
    if source_filter:
        where_filter = {"source": source_filter}

    vec_results = semantic_search(
        collection_name, query, top_k=fetch_k, where=where_filter
    )
    bm25_results = bm25_search(collection_name, query, top_k=fetch_k)

    # If source_filter, also filter BM25 results (BM25 has no where clause)
    if source_filter:
        bm25_results = [
            r for r in bm25_results
            if r.get("metadata", {}).get("source") == source_filter
        ]

    return reciprocal_rank_fusion(vec_results, bm25_results, alpha=alpha, top_k=top_k)


def semantic_only_search(
    collection_name: str,
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> List[dict]:
    return semantic_search(collection_name, query, top_k=top_k)


def keyword_only_search(
    collection_name: str,
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> List[dict]:
    return bm25_search(collection_name, query, top_k=top_k)
