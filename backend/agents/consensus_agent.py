"""
VidIntel AI — Consensus Agent
Responsibility: Compare opinions/viewpoints across multiple creators/videos.
Detects agreements, disagreements, and produces confidence-weighted summaries.

Best for Test Case 1 — tech product comparison across review videos.
"""

import json
from pathlib import Path
from typing import List
from core.rag.hybrid_retriever import hybrid_search
from core.llm.groq_client import simple_prompt, parse_json_response
from core.models.schemas import (
    ConsensusResponse,
    CreatorOpinion,
    TimestampResult,
)

_REGISTRY_PATH = Path(__file__).parent.parent / "data" / "video_registry.json"

def _get_video_url(video_id: str) -> str:
    """Look up the original YouTube URL from the video registry."""
    try:
        if _REGISTRY_PATH.exists():
            registry = json.loads(_REGISTRY_PATH.read_text())
            return registry.get(video_id, {}).get("url", "")
    except Exception:
        pass
    return ""


# ─── Prompts ───────────────────────────────────────────────────────────────────

OPINION_EXTRACTION_SYSTEM = """You are an expert at extracting and summarising creator opinions from video transcripts.

Given transcript chunks from a video, extract the creator's opinion on the user's question.

Return JSON with this exact structure:
{
  "creator": "channel or creator name",
  "opinion": "concise 1-2 sentence opinion",
  "sentiment": "positive" | "negative" | "neutral" | "mixed",
  "key_points": ["point 1", "point 2", "point 3"]
}"""


CONSENSUS_SYNTHESIS_SYSTEM = """You are VidIntel's Consensus Agent — an expert at synthesising
multiple creator opinions into structured intelligence.

Given opinions from multiple creators about the same topic, produce:
1. A consensus summary (what most agree on)
2. Clear agreements (points where creators align)
3. Clear disagreements (points where creators conflict)
4. Overall confidence in the consensus

Return JSON with this exact structure:
{
  "consensus_summary": "2-3 sentence summary of overall consensus",
  "agreements": ["agreement 1", "agreement 2", "agreement 3"],
  "disagreements": ["disagreement 1", "disagreement 2"],
  "confidence_score": 0.85
}

Be specific. Reference actual product aspects, specs, or claims."""


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _group_by_video(chunks: List[dict]) -> dict:
    """Group retrieved chunks by video_id."""
    groups: dict[str, list] = {}
    for chunk in chunks:
        vid = chunk.get("metadata", {}).get("video_id", "unknown")
        groups.setdefault(vid, []).append(chunk)
    return groups


def _extract_opinion_for_video(
    question: str,
    chunks: List[dict],
) -> dict | None:
    """Use LLM to extract a creator's opinion from their chunks."""
    meta = chunks[0].get("metadata", {})
    context = "\n\n".join(
        f"[{c['metadata'].get('timestamp_label','?')}] {c['text']}"
        for c in chunks[:5]
    )
    user_msg = (
        f"Question: {question}\n\n"
        f"Creator: {meta.get('channel', 'Unknown')}\n"
        f"Video: {meta.get('video_title', 'Unknown')}\n\n"
        f"Transcript excerpts:\n{context}"
    )
    try:
        raw = simple_prompt(OPINION_EXTRACTION_SYSTEM, user_msg, json_mode=True)
        opinion = parse_json_response(raw)
        opinion["video_id"] = meta.get("video_id", "")
        opinion["video_title"] = meta.get("video_title", "")
        opinion["timestamp_seconds"] = chunks[0]["metadata"].get("timestamp_seconds", 0)
        opinion["timestamp_label"] = chunks[0]["metadata"].get("timestamp_label", "0:00")
        return opinion
    except Exception as e:
        print(f"[ConsensusAgent] Opinion extraction failed: {e}")
        return None


# ─── Agent entrypoint ──────────────────────────────────────────────────────────

def run_consensus_agent(
    question: str,
    collection_name: str,
    top_k: int = 8,
) -> ConsensusResponse:
    """
    Run the Consensus Agent to compare creator opinions.

    Steps:
      1. Retrieve relevant chunks across all videos in the collection.
      2. Group chunks by video/creator.
      3. Extract per-creator opinion using LLM.
      4. Synthesise consensus across all opinions.

    Returns:
        ConsensusResponse with agreements, disagreements, and creator opinions.
    """
    print(f"[ConsensusAgent] Q: {question[:80]}...")

    # Retrieve broadly (no source filter — want transcript opinions)
    chunks = hybrid_search(collection_name, question, top_k=top_k)

    if not chunks:
        return ConsensusResponse(
            question=question,
            consensus_summary="No relevant content found across videos.",
            agreements=[],
            disagreements=[],
            creator_opinions=[],
            confidence_score=0.0,
        )

    # Group by video
    by_video = _group_by_video(chunks)
    print(f"[ConsensusAgent] Found content across {len(by_video)} videos.")

    # Extract per-video opinion
    raw_opinions = []
    for vid_id, vid_chunks in by_video.items():
        opinion = _extract_opinion_for_video(question, vid_chunks)
        if opinion:
            raw_opinions.append(opinion)

    if not raw_opinions:
        return ConsensusResponse(
            question=question,
            consensus_summary="Could not extract opinions from retrieved content.",
            agreements=[],
            disagreements=[],
            creator_opinions=[],
            confidence_score=0.0,
        )

    # Synthesise consensus
    opinions_text = json.dumps(raw_opinions, indent=2)
    user_msg = f"Question: {question}\n\nCreator opinions:\n{opinions_text}"

    try:
        raw = simple_prompt(CONSENSUS_SYNTHESIS_SYSTEM, user_msg, json_mode=True)
        synthesis = parse_json_response(raw)
    except Exception as e:
        print(f"[ConsensusAgent] Synthesis failed: {e}")
        synthesis = {
            "consensus_summary": "Unable to synthesise consensus.",
            "agreements": [],
            "disagreements": [],
            "confidence_score": 0.0,
        }

    # Build CreatorOpinion objects
    creator_opinions = [
        CreatorOpinion(
            video_id=op.get("video_id", ""),
            video_title=op.get("video_title", ""),
            creator=op.get("creator", "Unknown"),
            opinion=op.get("opinion", ""),
            timestamp_seconds=op.get("timestamp_seconds"),
            timestamp_label=op.get("timestamp_label"),
            sentiment=op.get("sentiment", "neutral"),
            url=_get_video_url(op.get("video_id", "")),
        )
        for op in raw_opinions
    ]

    return ConsensusResponse(
        question=question,
        consensus_summary=synthesis.get("consensus_summary", ""),
        agreements=synthesis.get("agreements", []),
        disagreements=synthesis.get("disagreements", []),
        creator_opinions=creator_opinions,
        confidence_score=float(synthesis.get("confidence_score", 0.5)),
    )
