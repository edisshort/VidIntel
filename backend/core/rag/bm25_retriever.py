"""
VidIntel AI — BM25 Keyword Retriever
Uses rank_bm25 to index and search corpus by exact/partial term matching.
Particularly effective for:
  - error codes (CUDA_ERROR_OUT_OF_MEMORY)
  - function names (useEffect, docker-compose)
  - exact terms that semantic search might paraphrase away
BM25 indexes are stored per collection in memory (rebuilt on startup).
For persistence across restarts, they are serialised to disk via pickle.
"""

import pickle
from pathlib import Path
from typing import List, Optional

from rank_bm25 import BM25Okapi

from config import CHROMA_DIR


# ─── Index registry ────────────────────────────────────────────────────────────
# Maps collection_name → {"index": BM25Okapi, "docs": List[dict]}
_indexes: dict = {}


def _index_path(collection_name: str) -> Path:
    return CHROMA_DIR / f"{collection_name}_bm25.pkl"


def _save_index(collection_name: str):
    data = _indexes[collection_name]
    with open(_index_path(collection_name), "wb") as f:
        pickle.dump(data, f)


def _load_index(collection_name: str) -> bool:
    path = _index_path(collection_name)
    if path.exists():
        with open(path, "rb") as f:
            _indexes[collection_name] = pickle.load(f)
        return True
    return False


# ─── Tokeniser ─────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer (keeps underscores, dots)."""
    import re
    tokens = re.findall(r"[\w\.]+", text.lower())
    return tokens


# ─── Build / update index ──────────────────────────────────────────────────────

def build_bm25_index(collection_name: str, documents: List[dict]):
    """
    Build (or rebuild) the BM25 index for a collection.

    Args:
        collection_name: Unique name for the collection.
        documents: List of {"text": str, "metadata": dict} dicts.
    """
    corpus = [_tokenize(d["text"]) for d in documents]
    index = BM25Okapi(corpus)
    _indexes[collection_name] = {"index": index, "docs": documents}
    _save_index(collection_name)
    print(f"[BM25] Built index for '{collection_name}' with {len(documents)} docs.")


def add_to_bm25_index(collection_name: str, documents: List[dict]):
    """Append documents to an existing index (rebuilds from scratch)."""
    existing = _indexes.get(collection_name, {}).get("docs", [])
    all_docs = existing + documents
    build_bm25_index(collection_name, all_docs)


# ─── Search ────────────────────────────────────────────────────────────────────

def bm25_search(
    collection_name: str,
    query: str,
    top_k: int = 5,
) -> List[dict]:
    """
    Run BM25 keyword search over the collection.
    Returns top_k results with normalised scores.
    """
    # Load from disk if not in memory
    if collection_name not in _indexes:
        if not _load_index(collection_name):
            print(f"[BM25] No index found for '{collection_name}'")
            return []

    data = _indexes[collection_name]
    index: BM25Okapi = data["index"]
    docs: List[dict] = data["docs"]

    tokens = _tokenize(query)
    raw_scores = index.get_scores(tokens)

    # Normalise scores 0–1
    max_score = max(raw_scores) if raw_scores.max() > 0 else 1.0
    scored = [
        (float(s / max_score), docs[i])
        for i, s in enumerate(raw_scores)
        if s > 0
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, doc in scored[:top_k]:
        results.append(
            {
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": round(score, 4),
                "retrieval_type": "bm25",
            }
        )

    return results
