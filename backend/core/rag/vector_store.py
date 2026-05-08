"""
VidIntel AI — ChromaDB Vector Store
Manages collections, upserts documents, and performs semantic search.
"""

from __future__ import annotations

import json
import uuid
from typing import List, Optional
import chromadb
from chromadb.config import Settings

from config import CHROMA_DIR, CHROMA_MODE, CHROMA_HOST, CHROMA_PORT
from core.rag.embeddings import embed_texts, embed_query


# ─── Client singleton ──────────────────────────────────────────────────────────

_client = None  # chromadb persistent/http client


def get_chroma_client() -> chromadb.Client:
    global _client
    if _client is None:
        if CHROMA_MODE == "remote":
            _client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        else:
            _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def get_or_create_collection(name: str) -> chromadb.Collection:
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


# ─── Upsert ────────────────────────────────────────────────────────────────────

def upsert_documents(
    collection_name: str,
    documents: List[dict],
    batch_size: int = 100,
) -> int:
    """
    Embed and upsert a list of document dicts:
      {"text": str, "metadata": dict}
    Returns number of documents upserted.
    """
    collection = get_or_create_collection(collection_name)
    total = 0

    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        texts = [d["text"] for d in batch]
        metas = [d["metadata"] for d in batch]
        ids = [str(uuid.uuid4()) for _ in batch]

        # Sanitise metadata (Chroma only accepts str/int/float/bool)
        metas = [_clean_meta(m) for m in metas]

        embeddings = embed_texts(texts)

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metas,
        )
        total += len(batch)

    return total


def _clean_meta(meta: dict) -> dict:
    clean = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif v is None:
            clean[k] = ""
        else:
            clean[k] = json.dumps(v)
    return clean


# ─── Semantic search ───────────────────────────────────────────────────────────

def semantic_search(
    collection_name: str,
    query: str,
    top_k: int = 5,
    where: Optional[dict] = None,
) -> List[dict]:
    """
    Run vector similarity search.
    Returns list of results sorted by relevance.
    """
    collection = get_or_create_collection(collection_name)
    q_vec = embed_query(query)

    results = collection.query(
        query_embeddings=[q_vec],
        n_results=min(top_k, collection.count() or 1),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append(
            {
                "text": doc,
                "metadata": meta,
                "score": round(1 - dist, 4),   # cosine sim from distance
                "retrieval_type": "vector",
            }
        )

    return hits


# ─── Collection management ─────────────────────────────────────────────────────

def list_collections() -> List[str]:
    client = get_chroma_client()
    return [c.name for c in client.list_collections()]


def collection_count(collection_name: str) -> int:
    col = get_or_create_collection(collection_name)
    return col.count()


def delete_collection(collection_name: str):
    client = get_chroma_client()
    client.delete_collection(collection_name)
