"""
VidIntel AI — Visual Agent
Responsibility: Search what appears VISUALLY on screen in video frames.
Uses OCR-indexed chunks from the RAG pipeline.
Can find: code on screen, diagrams, slides, terminal output, folder structures,
          error messages — even when the speaker never said the words aloud.

Best for Test Case 3 — visual/OCR search.
"""

from __future__ import annotations

import json
import re
from typing import List
from core.rag.hybrid_retriever import hybrid_search
from core.rag.bm25_retriever import bm25_search
from core.llm.groq_client import simple_prompt
from core.models.schemas import VisualResult, VisualSearchResponse


# ─── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are VidIntel's Visual Agent — specialised in understanding
what appears visually on screen in educational and technical videos.

You receive OCR-extracted text from video frames along with timestamps.
Your job is to identify which frames are most relevant to the user's query
and explain what was shown at each timestamp.

Rules:
- Focus on visual content: code, diagrams, terminal output, slides, file structures.
- Reference exact timestamps for each relevant frame.
- Describe what the screen shows, not what the speaker says.
- If multiple frames show the same content, group them.
- Be specific about file names, function names, or visual elements seen."""

RERANK_SYSTEM = """You are a relevance scorer for visual OCR search results.
Given a user query and a list of video frames with their OCR text, score each frame
from 0 to 10 based on how relevant the visual content is to the query.

Score meaning:
- 0-2: Not relevant at all (UI noise, unrelated content)
- 3-5: Tangentially related
- 6-8: Relevant content visible on screen
- 9-10: Exact match — the query topic is clearly shown on screen

Return ONLY a JSON array of numbers, one score per frame, in the same order.
Example for 3 frames: [2, 8, 5]"""


def _llm_rerank(query: str, chunks: List[dict]) -> List[dict]:
    """
    Use the LLM to score each chunk for relevance to the query.
    Returns chunks sorted by LLM relevance score, filtered to score >= 4.
    Falls back to original order if LLM call fails.
    """
    if not chunks:
        return chunks

    # Build a compact frame list for the LLM
    frame_lines = []
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        ts = meta.get("timestamp_label", "?:??")
        text = c["text"][:200]  # Keep it short for speed
        frame_lines.append(f"Frame {i} [{ts}]: {text}")

    frames_text = "\n".join(frame_lines)
    user_msg = (
        f"Query: {query}\n\n"
        f"Frames to score:\n{frames_text}\n\n"
        f"Return a JSON array of {len(chunks)} relevance scores (0-10)."
    )

    try:
        response = simple_prompt(RERANK_SYSTEM, user_msg)
        # Extract JSON array from response
        match = re.search(r'\[[\d\s,\.]+\]', response)
        if not match:
            print("[VisualAgent] LLM rerank: no JSON array found, using original order")
            return chunks

        scores = json.loads(match.group())
        if len(scores) != len(chunks):
            print(f"[VisualAgent] LLM rerank: score count mismatch ({len(scores)} vs {len(chunks)})")
            return chunks

        # Attach LLM score to each chunk
        scored = []
        for chunk, llm_score in zip(chunks, scores):
            chunk = dict(chunk)  # shallow copy
            chunk["llm_score"] = float(llm_score)
            scored.append(chunk)

        # Filter low relevance, then sort by LLM score descending
        relevant = [c for c in scored if c["llm_score"] >= 4.0]
        relevant.sort(key=lambda x: x["llm_score"], reverse=True)

        print(f"[VisualAgent] LLM rerank: {len(chunks)} → {len(relevant)} relevant frames")
        return relevant if relevant else chunks  # fallback if all filtered out

    except Exception as e:
        print(f"[VisualAgent] LLM rerank failed ({e}), using original order")
        return chunks


def _build_context(chunks: List[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        ts = meta.get("timestamp_label", "?:??")
        title = meta.get("video_title", "Unknown")
        lines.append(f"[{i}] [{ts}] ({title})\nOCR TEXT: {c['text']}")
    return "\n\n".join(lines)


def _make_visual_results(chunks: List[dict]) -> List[VisualResult]:
    results = []
    for c in chunks:
        meta = c.get("metadata", {})
        results.append(
            VisualResult(
                video_id=meta.get("video_id", ""),
                video_title=meta.get("video_title", ""),
                timestamp_seconds=meta.get("timestamp_seconds", 0),
                timestamp_label=meta.get("timestamp_label", "0:00"),
                frame_path=meta.get("frame_path", ""),
                ocr_text=c["text"][:300],
                score=c.get("llm_score", c.get("score", 0)),
            )
        )
    return results


# ─── Agent entrypoint ──────────────────────────────────────────────────────────

def run_visual_agent(
    query: str,
    collection_name: str,
    top_k: int = 5,
) -> VisualSearchResponse:
    """
    Run the Visual Agent to find what appears on screen matching the query.

    Prioritises BM25 keyword search (better for exact visual text like
    'docker-compose.yml', 'useEffect', 'CUDA_ERROR') then re-ranks with
    semantic search, and finally uses LLM relevance scoring to ensure
    different queries return genuinely different, relevant frames.
    """
    print(f"[VisualAgent] Query: {query[:80]}...")

    # Prioritise BM25 for exact visual text matches — cast a wider net
    bm25_chunks = bm25_search(collection_name, query, top_k=top_k * 3)
    bm25_ocr = [
        c for c in bm25_chunks
        if c.get("metadata", {}).get("source") == "ocr"
    ]

    # Supplement with semantic OCR search
    semantic_ocr = hybrid_search(
        collection_name, query, top_k=top_k * 2, source_filter="ocr"
    )

    # Merge and deduplicate
    seen = set()
    all_chunks = []
    for chunk in bm25_ocr + semantic_ocr:
        fp = chunk.get("metadata", {}).get("frame_path", chunk["text"][:50])
        if fp not in seen:
            seen.add(fp)
            all_chunks.append(chunk)

    all_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)

    # Take top candidates for LLM re-ranking (cap at 15 to avoid token overrun)
    candidates = all_chunks[:min(15, len(all_chunks))]

    if not candidates:
        return VisualSearchResponse(
            query=query,
            results=[],
        )

    # LLM relevance re-ranking — this is what makes different queries return
    # different frames instead of the same high-scoring generic frames
    reranked = _llm_rerank(query, candidates)
    final_chunks = reranked[:top_k]

    return VisualSearchResponse(
        query=query,
        results=_make_visual_results(final_chunks),
    )
