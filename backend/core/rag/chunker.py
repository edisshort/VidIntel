"""
VidIntel AI — Text Chunker
Converts transcript segments into overlapping chunks suitable for RAG.
Preserves timestamp metadata on each chunk for exact retrieval.
"""

from typing import List
from config import TRANSCRIPTS_DIR


# ─── Config ────────────────────────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE = 400       # characters
DEFAULT_CHUNK_OVERLAP = 80     # characters


# ─── Transcript → Chunks ───────────────────────────────────────────────────────

def chunk_transcript(
    segments: List[dict],
    video_meta: dict,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[dict]:
    """
    Group transcript segments into overlapping text chunks.
    Each chunk keeps the start/end timestamps of the segments it covers.

    Returns list of:
      {
        "text": str,
        "metadata": {
          "video_id": str,
          "video_title": str,
          "channel": str,
          "url": str,
          "timestamp_seconds": float,       # start of the chunk
          "timestamp_end_seconds": float,   # end of the chunk
          "timestamp_label": str,
          "source": "transcript"
        }
      }
    """
    video_id = video_meta.get("video_id", "unknown")
    chunks = []
    buffer_text = ""
    buffer_start = 0.0
    buffer_end = 0.0
    buffer_label = "0:00"

    for seg in segments:
        seg_text = seg.get("text", "").strip()
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", seg_start + 2.0)
        seg_label = seg.get("timestamp_label", _fmt(seg_start))

        if not buffer_text:
            buffer_start = seg_start
            buffer_label = seg_label

        buffer_text += " " + seg_text
        buffer_end = seg_end

        if len(buffer_text) >= chunk_size:
            chunks.append(_make_chunk(
                buffer_text.strip(),
                video_meta,
                buffer_start,
                buffer_end,
                buffer_label,
            ))
            # Overlap: keep last `overlap` chars as context seed
            buffer_text = buffer_text[-overlap:]
            buffer_start = seg_start
            buffer_label = seg_label

    # Flush remaining
    if buffer_text.strip():
        chunks.append(_make_chunk(
            buffer_text.strip(),
            video_meta,
            buffer_start,
            buffer_end,
            buffer_label,
        ))

    return chunks


def _make_chunk(text, meta, t_start, t_end, t_label) -> dict:
    return {
        "text": text,
        "metadata": {
            "video_id": meta.get("video_id", ""),
            "video_title": meta.get("title", ""),
            "channel": meta.get("channel", ""),
            "url": meta.get("url", ""),
            "thumbnail_url": meta.get("thumbnail_url", ""),
            "timestamp_seconds": t_start,
            "timestamp_end_seconds": t_end,
            "timestamp_label": t_label,
            "source": "transcript",
        },
    }


def _fmt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"
