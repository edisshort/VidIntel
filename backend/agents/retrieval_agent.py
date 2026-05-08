"""
VidIntel AI — Retrieval Agent
Responsibility: Given a user question, retrieve the most relevant
transcript/OCR chunks from the vector + BM25 indexes,
then generate a grounded, timestamped answer.

Handles: exact timestamp queries, concept lookups, tutorial navigation.
Best for Test Case 2 — exact timestamp retrieval from coding tutorials.
"""

from typing import List
from core.rag.hybrid_retriever import hybrid_search, keyword_only_search, semantic_only_search
from core.llm.groq_client import simple_prompt
from core.models.schemas import TimestampResult, QueryResponse


# ─── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are VidIntel's Retrieval Agent — an expert at finding exact moments
in video transcripts and returning precise, grounded answers.

Rules:
- Only use information from the provided context chunks.
- Always cite timestamps (e.g. "[12:34]") when referencing specific content.
- If the answer spans multiple timestamps, list all of them.
- If the information isn't in the context, say so honestly.
- Be concise and precise. Don't pad your answer.
- Format: Answer → Relevant Timestamps → Brief Explanation of each timestamp."""


def _build_context(chunks: List[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        ts = meta.get("timestamp_label", "?:??")
        title = meta.get("video_title", "Unknown")
        src = meta.get("source", "transcript")
        lines.append(
            f"[{i}] [{ts}] ({title}) [{src.upper()}]\n{c['text']}"
        )
    return "\n\n".join(lines)


def _chunks_to_timestamps(chunks: List[dict]) -> List[TimestampResult]:
    results = []
    for c in chunks:
        meta = c.get("metadata", {})
        results.append(
            TimestampResult(
                video_id=meta.get("video_id", ""),
                video_title=meta.get("video_title", ""),
                timestamp_seconds=meta.get("timestamp_seconds", 0),
                timestamp_label=meta.get("timestamp_label", "0:00"),
                text_snippet=c["text"][:200],
                score=c.get("score", 0),
                source=meta.get("source", "transcript"),
            )
        )
    return results


# ─── Agent entrypoint ──────────────────────────────────────────────────────────

def run_retrieval_agent(
    question: str,
    collection_name: str,
    mode: str = "hybrid",   # "hybrid" | "semantic" | "keyword"
    top_k: int = 5,
    include_visual: bool = True,
) -> QueryResponse:
    """
    Run the retrieval agent to answer a timestamped question.

    Args:
        question:        User's natural language question.
        collection_name: ChromaDB collection to search.
        mode:            Retrieval mode.
        top_k:           Number of results to retrieve.
        include_visual:  Whether to also search OCR/visual chunks.

    Returns:
        QueryResponse with answer, timestamps, and metadata.
    """
    print(f"[RetrievalAgent] Q: {question[:80]}...")

    # Retrieve transcript chunks
    if mode == "semantic":
        transcript_chunks = semantic_only_search(collection_name, question, top_k)
    elif mode == "keyword":
        transcript_chunks = keyword_only_search(collection_name, question, top_k)
    else:
        transcript_chunks = hybrid_search(
            collection_name, question, top_k=top_k, source_filter="transcript"
        )

    # Optionally retrieve OCR chunks (visual)
    ocr_chunks = []
    if include_visual:
        ocr_chunks = hybrid_search(
            collection_name, question, top_k=top_k // 2 or 2, source_filter="ocr"
        )

    all_chunks = transcript_chunks + ocr_chunks
    all_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
    all_chunks = all_chunks[:top_k]

    if not all_chunks:
        return QueryResponse(
            answer="No relevant content found. Try ingesting more videos first.",
            timestamps=[],
            agent_used="retrieval_agent",
            retrieval_mode=mode,
            sources_used=0,
        )

    context = _build_context(all_chunks)
    user_msg = (
        f"Question: {question}\n\n"
        f"Context from video transcripts and visual content:\n\n{context}"
    )

    answer = simple_prompt(SYSTEM_PROMPT, user_msg)
    timestamps = _chunks_to_timestamps(all_chunks)

    return QueryResponse(
        answer=answer,
        timestamps=timestamps,
        agent_used="retrieval_agent",
        retrieval_mode=mode,
        sources_used=len(all_chunks),
    )
