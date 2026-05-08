"""
VidIntel AI — Visual Search Routes
POST /api/visual/search  — Find content shown on screen (OCR-based)
GET  /api/visual/frame   — Serve a specific frame image
"""

import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from core.models.schemas import VisualSearchRequest, VisualSearchResponse
from agents.visual_agent import run_visual_agent

router = APIRouter(prefix="/api/visual", tags=["Visual Search"])


@router.post("/search", response_model=VisualSearchResponse)
async def visual_search(req: VisualSearchRequest):
    """
    Search for content that appears visually on screen in video frames.
    Uses OCR-indexed text combined with BM25 + semantic search.

    Great for finding:
    - Code shown on screen
    - Folder structures
    - Terminal output
    - Slides / diagrams
    - Error messages
    - File contents
    """
    try:
        return run_visual_agent(
            query=req.query,
            collection_name=req.collection_name,
            top_k=req.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/frame")
async def get_frame(path: str):
    """
    Serve a frame image by absolute path.
    Frontend uses this to display matched frames.
    """
    frame_path = Path(path)
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found")
    if not frame_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=400, detail="Invalid file type")
    return FileResponse(str(frame_path), media_type="image/jpeg")
