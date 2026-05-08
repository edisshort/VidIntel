"""
VidIntel AI — Pydantic schemas for request/response models.
"""

from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict, Any
from enum import Enum


# ─── Enums ────────────────────────────────────────────────────────────────────

class QueryMode(str, Enum):
    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    VISUAL = "visual"


class AgentMode(str, Enum):
    RETRIEVAL = "retrieval"
    CONSENSUS = "consensus"
    RESEARCH = "research"
    FULL = "full"


# ─── Ingestion ─────────────────────────────────────────────────────────────────

class VideoIngestRequest(BaseModel):
    url: str
    extract_frames: bool = True
    frame_interval: int = 30          # seconds between extracted frames
    run_ocr: bool = True
    collection_name: Optional[str] = None   # custom ChromaDB collection


class VideoIngestResponse(BaseModel):
    video_id: str
    title: str
    duration_seconds: int
    transcript_chunks: int
    frames_extracted: int
    ocr_text_chunks: int
    collection_name: str
    status: str                        # "success" | "partial" | "failed"
    message: str


class BatchIngestRequest(BaseModel):
    urls: List[str]
    collection_name: str               # group all under one collection
    extract_frames: bool = True
    frame_interval: int = 30
    run_ocr: bool = True


# ─── Query / Chat ──────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    collection_name: str
    mode: QueryMode = QueryMode.HYBRID
    agent_mode: AgentMode = AgentMode.FULL
    top_k: int = 5
    include_visual: bool = True


class TimestampResult(BaseModel):
    video_id: str
    video_title: str
    timestamp_seconds: float
    timestamp_label: str               # e.g. "12:34"
    text_snippet: str
    score: float
    source: str                        # "transcript" | "ocr"


class QueryResponse(BaseModel):
    answer: str
    timestamps: List[TimestampResult]
    agent_used: str
    retrieval_mode: str
    sources_used: int


# ─── Consensus Analysis ────────────────────────────────────────────────────────

class ConsensusRequest(BaseModel):
    question: str
    collection_name: str
    top_k: int = 8


class CreatorOpinion(BaseModel):
    video_id: str
    video_title: str
    creator: str
    opinion: str
    timestamp_seconds: Optional[float]
    timestamp_label: Optional[str]
    sentiment: str                     # "positive" | "negative" | "neutral"
    url: Optional[str] = ""           # original YouTube URL for timestamp links


class ConsensusResponse(BaseModel):
    question: str
    consensus_summary: str
    agreements: List[str]
    disagreements: List[str]
    creator_opinions: List[CreatorOpinion]
    confidence_score: float            # 0.0 – 1.0


# ─── Visual Search ─────────────────────────────────────────────────────────────

class VisualSearchRequest(BaseModel):
    query: str
    collection_name: str
    top_k: int = 5


class VisualResult(BaseModel):
    video_id: str
    video_title: str
    timestamp_seconds: float
    timestamp_label: str
    frame_path: str
    ocr_text: str
    score: float


class VisualSearchResponse(BaseModel):
    query: str
    results: List[VisualResult]


# ─── Library / Collections ─────────────────────────────────────────────────────

class VideoMeta(BaseModel):
    video_id: str
    title: str
    channel: str
    duration_seconds: int
    url: str
    thumbnail_url: Optional[str]
    collection_name: str
    ingested_at: str


class CollectionInfo(BaseModel):
    name: str
    video_count: int
    total_chunks: int
    videos: List[VideoMeta]


# ─── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    components: Dict[str, str]
