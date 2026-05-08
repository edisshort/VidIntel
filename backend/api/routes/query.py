"""
VidIntel AI — Query Routes
POST /api/query     — Ask a question (auto-routes to best agent)
POST /api/consensus — Cross-video opinion analysis
"""

from fastapi import APIRouter, HTTPException
from core.models.schemas import (
    QueryRequest,
    QueryResponse,
    ConsensusRequest,
    ConsensusResponse,
    QueryMode,
    AgentMode,
)
from agents.research_agent import run_research_agent
from agents.retrieval_agent import run_retrieval_agent
from agents.consensus_agent import run_consensus_agent

router = APIRouter(prefix="/api", tags=["Query"])


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    Main query endpoint. Routes to the appropriate agent based on agent_mode:
      - full:      Research Agent (auto-classifies and routes)
      - retrieval: Retrieval Agent (timestamp/concept lookup)
      - consensus: Consensus Agent (cross-video opinion comparison)
      - research:  Research Agent (forced full synthesis)
    """
    try:
        if req.agent_mode == AgentMode.RETRIEVAL:
            return run_retrieval_agent(
                question=req.question,
                collection_name=req.collection_name,
                mode=req.mode.value,
                top_k=req.top_k,
                include_visual=req.include_visual,
            )

        if req.agent_mode == AgentMode.CONSENSUS:
            consensus = run_consensus_agent(
                question=req.question,
                collection_name=req.collection_name,
                top_k=req.top_k,
            )
            # Wrap for unified response
            from core.models.schemas import TimestampResult
            return QueryResponse(
                answer=consensus.consensus_summary,
                timestamps=[
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
                ],
                agent_used="consensus_agent",
                retrieval_mode="hybrid",
                sources_used=len(consensus.creator_opinions),
            )

        # Default: full research agent (auto-routes)
        return run_research_agent(
            question=req.question,
            collection_name=req.collection_name,
            top_k=req.top_k,
            include_visual=req.include_visual,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/consensus", response_model=ConsensusResponse)
async def consensus_analysis(req: ConsensusRequest):
    """
    Dedicated consensus analysis endpoint.
    Compares creator opinions across all videos in a collection.
    Returns structured agreements, disagreements, and per-creator opinions.
    """
    try:
        return run_consensus_agent(
            question=req.question,
            collection_name=req.collection_name,
            top_k=req.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
