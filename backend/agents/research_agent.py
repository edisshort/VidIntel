"""
VidIntel AI — Research Agent (Orchestrator)
Responsibility: Acts as the master orchestrator. Analyses the user's query,
decides which sub-agents to invoke, collects their results, and synthesises
a final, comprehensive research-quality answer.

This is the default agent for /api/query requests with agent_mode="full".
"""

from typing import List
from core.rag.hybrid_retriever import hybrid_search
from core.llm.groq_client import simple_prompt, parse_json_response, chat_complete
from core.models.schemas import QueryResponse, TimestampResult
from agents.retrieval_agent import run_retrieval_agent
from agents.consensus_agent import run_consensus_agent
from agents.visual_agent import run_visual_agent


# ─── Query classifier ──────────────────────────────────────────────────────────

CLASSIFIER_SYSTEM = """You are a query router for a video intelligence system.
Classify the user's question into one of these categories:
- "timestamp": User wants to find a specific moment/topic in a video (e.g., "where does he explain X")
- "consensus": User wants to compare opinions across multiple creators (e.g., "what do reviewers think about X")
- "visual": User wants to find something shown on screen (e.g., "find where the folder structure is shown")
- "research": General research/synthesis question requiring full context

Return JSON: {"category": "timestamp" | "consensus" | "visual" | "research", "confidence": 0.9}"""


SYNTHESIS_SYSTEM = """You are VidIntel's Research Agent — an AI intelligence engine for
extracting structured knowledge from video content.

You have retrieved relevant transcript and visual context. Your job is to produce
a comprehensive, well-structured answer that:
- Synthesises information across all relevant timestamps and sources
- Clearly attributes information to specific videos/creators
- Highlights key insights, patterns, and relationships
- Provides actionable, specific information

Always reference timestamps in [MM:SS] format when citing specific moments.
Structure complex answers with clear sections."""


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _classify_query(question: str) -> str:
    """Classify query to route to the right agent."""
    try:
        raw = simple_prompt(CLASSIFIER_SYSTEM, f"Question: {question}", json_mode=True)
        result = parse_json_response(raw)
        return result.get("category", "research")
    except Exception:
        return "research"


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

def run_research_agent(
    question: str,
    collection_name: str,
    top_k: int = 5,
    include_visual: bool = True,
) -> QueryResponse:
    """
    Master orchestrator. Routes to specialist agents or handles directly.

    Routing logic:
      - "timestamp" → RetrievalAgent
      - "consensus" → ConsensusAgent (returns wrapped as QueryResponse)
      - "visual"    → VisualAgent (returns wrapped as QueryResponse)
      - "research"  → Full retrieval + LLM synthesis
    """
    print(f"[ResearchAgent] Classifying: {question[:80]}...")
    category = _classify_query(question)
    print(f"[ResearchAgent] Category: {category}")

    # ── Delegate to specialist agents ──────────────────────────────────────────
    if category == "timestamp":
        return run_retrieval_agent(
            question, collection_name, top_k=top_k, include_visual=include_visual
        )

    if category == "consensus":
        consensus = run_consensus_agent(question, collection_name, top_k=top_k)
        # Wrap ConsensusResponse into QueryResponse for unified API
        ts_results = [
            TimestampResult(
                video_id=op.video_id,
                video_title=op.video_title,
                timestamp_seconds=op.timestamp_seconds or 0,
                timestamp_label=op.timestamp_label or "0:00",
                text_snippet=op.opinion,
                score=consensus.confidence_score,
                source="transcript",
            )
            for op in consensus.creator_opinions
        ]
        return QueryResponse(
            answer=(
                f"**Consensus Summary**\n{consensus.consensus_summary}\n\n"
                f"**Agreements**\n" + "\n".join(f"• {a}" for a in consensus.agreements) +
                "\n\n**Disagreements**\n" + "\n".join(f"• {d}" for d in consensus.disagreements)
            ),
            timestamps=ts_results,
            agent_used="consensus_agent",
            retrieval_mode="hybrid",
            sources_used=len(consensus.creator_opinions),
        )

    if category == "visual":
        visual = run_visual_agent(question, collection_name, top_k=top_k)
        ts_results = [
            TimestampResult(
                video_id=r.video_id,
                video_title=r.video_title,
                timestamp_seconds=r.timestamp_seconds,
                timestamp_label=r.timestamp_label,
                text_snippet=r.ocr_text,
                score=r.score,
                source="ocr",
            )
            for r in visual.results
        ]
        desc = (
            f"Found {len(visual.results)} visual matches for '{question}'.\n"
            + "\n".join(
                f"• [{r.timestamp_label}] {r.ocr_text[:100]}"
                for r in visual.results[:5]
            )
        )
        return QueryResponse(
            answer=desc,
            timestamps=ts_results,
            agent_used="visual_agent",
            retrieval_mode="hybrid",
            sources_used=len(visual.results),
        )

    # ── Full research synthesis ────────────────────────────────────────────────
    chunks = hybrid_search(collection_name, question, top_k=top_k)
    if include_visual:
        visual_chunks = hybrid_search(
            collection_name, question, top_k=max(2, top_k // 3), source_filter="ocr"
        )
        chunks = (chunks + visual_chunks)[:top_k]
        chunks.sort(key=lambda x: x.get("score", 0), reverse=True)

    if not chunks:
        return QueryResponse(
            answer="No relevant content found. Ingest some videos first.",
            timestamps=[],
            agent_used="research_agent",
            retrieval_mode="hybrid",
            sources_used=0,
        )

    context_lines = []
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        context_lines.append(
            f"[{i}] [{meta.get('timestamp_label', '?')}] "
            f"({meta.get('video_title', 'Unknown')}) "
            f"[{meta.get('source', 'transcript').upper()}]\n{c['text']}"
        )
    context = "\n\n".join(context_lines)

    messages = [
        {"role": "system", "content": SYNTHESIS_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Research question: {question}\n\n"
                f"Retrieved content:\n\n{context}"
            ),
        },
    ]
    answer = chat_complete(messages)

    return QueryResponse(
        answer=answer,
        timestamps=_chunks_to_timestamps(chunks),
        agent_used="research_agent",
        retrieval_mode="hybrid",
        sources_used=len(chunks),
    )
