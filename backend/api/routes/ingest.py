"""
VidIntel AI — Ingestion Routes
POST /api/ingest       — Single video
POST /api/ingest/batch — Multiple videos
GET  /api/library      — List all collections and videos
"""

import json
from pathlib import Path
from typing import List
from fastapi import APIRouter, HTTPException, BackgroundTasks

from core.models.schemas import (
    VideoIngestRequest,
    VideoIngestResponse,
    BatchIngestRequest,
    CollectionInfo,
    VideoMeta,
)
from core.ingestion import (
    get_video_metadata,
    sanitize_id,
    extract_transcript,
    extract_frames,
    process_frames_ocr,
    ocr_chunks_to_documents,
)
from core.rag.chunker import chunk_transcript
from core.rag.vector_store import upsert_documents, list_collections, collection_count
from core.rag.bm25_retriever import add_to_bm25_index
from config import DATA_DIR

router = APIRouter(prefix="/api", tags=["Ingestion"])

# Simple in-memory video registry (persisted to JSON)
REGISTRY_PATH = DATA_DIR / "video_registry.json"


def _load_registry() -> dict:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {}


def _save_registry(registry: dict):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def _register_video(meta: dict, collection_name: str):
    registry = _load_registry()
    registry[meta["video_id"]] = {**meta, "collection_name": collection_name}
    _save_registry(registry)


# ─── Core ingestion logic ──────────────────────────────────────────────────────

async def _ingest_video(
    url: str,
    collection_name: str,
    extract_frames_flag: bool,
    frame_interval: int,
    run_ocr: bool,
) -> VideoIngestResponse:
    try:
        # 1. Metadata
        meta = get_video_metadata(url)
        video_id = meta["video_id"]
        col_name = collection_name or f"vidIntel_{video_id}"

        # 2. Transcript
        segments = extract_transcript(url)
        transcript_docs = chunk_transcript(segments, meta)

        # 3. Index transcript
        upsert_documents(col_name, transcript_docs)
        add_to_bm25_index(col_name, transcript_docs)

        frames_count = 0
        ocr_count = 0

        # 4. Frames + OCR (graceful — transcript ingestion succeeds even if this fails)
        if extract_frames_flag:
            try:
                frames = extract_frames(url, interval_sec=frame_interval)
                frames_count = len(frames)

                if run_ocr and frames:
                    ocr_chunks = process_frames_ocr(video_id, frames)
                    ocr_docs = ocr_chunks_to_documents(ocr_chunks)

                    for doc in ocr_docs:
                        doc["metadata"].update({
                            "video_title": meta["title"],
                            "channel": meta["channel"],
                            "url": url,
                        })

                    upsert_documents(col_name, ocr_docs)
                    add_to_bm25_index(col_name, ocr_docs)
                    ocr_count = len(ocr_docs)
            except Exception as frame_err:
                print(f"[Ingest] Frame/OCR extraction failed (non-fatal): {frame_err}")

        # 5. Register
        _register_video(meta, col_name)

        return VideoIngestResponse(
            video_id=video_id,
            title=meta["title"],
            duration_seconds=meta["duration_seconds"],
            transcript_chunks=len(transcript_docs),
            frames_extracted=frames_count,
            ocr_text_chunks=ocr_count,
            collection_name=col_name,
            status="success",
            message=f"Successfully ingested '{meta['title']}'",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Routes ────────────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=VideoIngestResponse)
async def ingest_video(req: VideoIngestRequest):
    """
    Ingest a single YouTube video: extract transcript, frames, and run OCR.
    """
    return await _ingest_video(
        url=req.url,
        collection_name=req.collection_name or f"vid_{sanitize_id(req.url)}",
        extract_frames_flag=req.extract_frames,
        frame_interval=req.frame_interval,
        run_ocr=req.run_ocr,
    )


@router.post("/ingest/batch", response_model=List[VideoIngestResponse])
async def ingest_batch(req: BatchIngestRequest, background_tasks: BackgroundTasks):
    """
    Ingest multiple videos into the same collection.
    Videos are processed sequentially (background processing planned).
    """
    results = []
    for url in req.urls:
        result = await _ingest_video(
            url=url,
            collection_name=req.collection_name,
            extract_frames_flag=req.extract_frames,
            frame_interval=req.frame_interval,
            run_ocr=req.run_ocr,
        )
        results.append(result)
    return results


@router.get("/library", response_model=List[CollectionInfo])
async def get_library():
    """List all ingested video collections."""
    registry = _load_registry()
    collections_map: dict[str, list] = {}

    for vid_data in registry.values():
        col = vid_data.get("collection_name", "unknown")
        collections_map.setdefault(col, []).append(vid_data)

    result = []
    for col_name, videos in collections_map.items():
        video_metas = [
            VideoMeta(
                video_id=v["video_id"],
                title=v.get("title", ""),
                channel=v.get("channel", ""),
                duration_seconds=v.get("duration_seconds", 0),
                url=v.get("url", ""),
                thumbnail_url=v.get("thumbnail_url"),
                collection_name=col_name,
                ingested_at=v.get("upload_date", ""),
            )
            for v in videos
        ]
        result.append(
            CollectionInfo(
                name=col_name,
                video_count=len(videos),
                total_chunks=collection_count(col_name),
                videos=video_metas,
            )
        )

    return result
