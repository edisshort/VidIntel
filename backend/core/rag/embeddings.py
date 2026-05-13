"""
VidIntel AI — Embeddings
Two modes:
  - HF_API_KEY set   → HuggingFace Inference API (no RAM, works on Render free tier)
  - HF_API_KEY unset → local sentence-transformers (default for local dev)
Same model, same vectors, same quality — just where it runs differs.
"""

from __future__ import annotations

import os
import time
import requests
from typing import List, Optional
from config import EMBEDDING_MODEL

HF_API_KEY = os.getenv("HF_API_KEY", "").strip()
_HF_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL}"

# Local model singleton (only used when HF_API_KEY is not set)
_local_model = None


def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[Embeddings] Loading local model: {EMBEDDING_MODEL}")
        _local_model = SentenceTransformer(EMBEDDING_MODEL)
    return _local_model


def get_embedding_model():
    """Returns HF API URL string or local model depending on config."""
    if HF_API_KEY:
        return "hf_api"
    return _get_local_model()


# ─── HuggingFace API ───────────────────────────────────────────────────────────

def _embed_via_hf_api(texts: List[str]) -> List[List[float]]:
    """Call HuggingFace Inference API for embeddings. Batches automatically."""
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    all_embeddings = []

    # HF free tier: batch max ~100 texts at a time
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(3):
            resp = requests.post(
                _HF_API_URL,
                headers=headers,
                json={"inputs": batch, "options": {"wait_for_model": True}},
                timeout=60,
            )
            if resp.status_code == 503:
                # Model loading on HF side — wait and retry
                wait = resp.json().get("estimated_time", 20)
                print(f"[Embeddings] HF model loading, waiting {wait:.0f}s...")
                time.sleep(min(wait, 30))
                continue
            resp.raise_for_status()
            all_embeddings.extend(resp.json())
            break

    return all_embeddings


def _embed_via_hf_api_single(text: str) -> List[float]:
    result = _embed_via_hf_api([text])
    return result[0]


# ─── Public API ────────────────────────────────────────────────────────────────

def embed_texts(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    """Generate embeddings for a list of texts."""
    if HF_API_KEY:
        print(f"[Embeddings] Using HuggingFace API for {len(texts)} texts")
        return _embed_via_hf_api(texts)

    model = _get_local_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """Embed a single query string."""
    prefixed = f"Represent this sentence for searching relevant passages: {query}"

    if HF_API_KEY:
        return _embed_via_hf_api_single(prefixed)

    model = _get_local_model()
    vec = model.encode(prefixed, normalize_embeddings=True, convert_to_numpy=True)
    return vec.tolist()
