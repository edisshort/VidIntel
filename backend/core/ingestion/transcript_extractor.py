"""
VidIntel AI — Transcript Extractor
Strategy:
  1. Try YouTube auto-captions via yt-dlp (fastest, free, timestamped).
  2. Fall back to Groq's Whisper API (same GROQ_API_KEY, no local model needed).

Groq's whisper-large-v3 endpoint is free-tier, very fast, and requires
zero local dependencies — no FFmpeg headers, no CUDA, no model downloads.
Each segment: {"start": float, "end": float, "text": str}
"""

import json
from pathlib import Path
from typing import List, Optional

from config import TRANSCRIPTS_DIR, GROQ_API_KEY
from core.ingestion.video_downloader import (
    get_auto_captions,
    download_audio,
    sanitize_id,
)


# ─── Types ─────────────────────────────────────────────────────────────────────

TranscriptSegment = dict   # {"start": float, "end": float, "text": str}


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _save_transcript(video_id: str, segments: List[TranscriptSegment]) -> Path:
    path = TRANSCRIPTS_DIR / f"{video_id}_transcript.json"
    with open(path, "w") as f:
        json.dump(segments, f, indent=2)
    return path


def _load_cached(video_id: str) -> Optional[List[TranscriptSegment]]:
    path = TRANSCRIPTS_DIR / f"{video_id}_transcript.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ─── Caption-based (fast path) ─────────────────────────────────────────────────

def _captions_to_segments(
    captions: list, gap_threshold: float = 2.0
) -> List[TranscriptSegment]:
    """
    Merge raw caption events into proper segments with start/end times.
    """
    segments = []
    for i, cap in enumerate(captions):
        start = cap["start"]
        end = captions[i + 1]["start"] if i + 1 < len(captions) else start + 3.0
        segments.append(
            {
                "start": start,
                "end": end,
                "text": cap["text"].strip(),
                "timestamp_label": _format_timestamp(start),
                "source": "captions",
            }
        )
    return segments


# ─── Whisper fallback via Groq API ────────────────────────────────────────────

def _transcribe_with_groq(audio_path: Path) -> List[TranscriptSegment]:
    """
    Transcribe audio using Groq's Whisper API.
    Uses the same GROQ_API_KEY — no local model, no FFmpeg, no compilation.
    Returns timestamped segments using verbose_json response format.
    """
    from groq import Groq

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set — cannot use Groq Whisper fallback.")

    client = Groq(api_key=GROQ_API_KEY)
    print(f"[Whisper/Groq] Transcribing {audio_path.name} via Groq API...")

    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            file=(audio_path.name, f),
            model="whisper-large-v3",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    segments = []
    # verbose_json returns .segments list with start/end/text
    raw_segments = getattr(response, "segments", None) or []
    for seg in raw_segments:
        start = seg.get("start", 0.0) if isinstance(seg, dict) else getattr(seg, "start", 0.0)
        end   = seg.get("end",   start + 2.0) if isinstance(seg, dict) else getattr(seg, "end", start + 2.0)
        text  = seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")
        segments.append(
            {
                "start": float(start),
                "end":   float(end),
                "text":  text.strip(),
                "timestamp_label": _format_timestamp(float(start)),
                "source": "whisper-groq",
            }
        )

    # Fallback: if no segments returned, make one chunk from full text
    if not segments and hasattr(response, "text") and response.text:
        segments.append(
            {
                "start": 0.0,
                "end":   0.0,
                "text":  response.text.strip(),
                "timestamp_label": "0:00",
                "source": "whisper-groq",
            }
        )

    return segments


# ─── Public API ────────────────────────────────────────────────────────────────

def extract_transcript(url: str, force_whisper: bool = False) -> List[TranscriptSegment]:
    """
    Main entry point. Returns list of transcript segments with timestamps.
    Caches result to disk for re-use.
    """
    video_id = sanitize_id(url)

    # Check cache
    cached = _load_cached(video_id)
    if cached:
        print(f"[Transcript] Using cached transcript for {video_id}")
        return cached

    segments: List[TranscriptSegment] = []

    # Try auto-captions first (unless whisper forced)
    if not force_whisper:
        captions = get_auto_captions(url)
        if captions:
            print(f"[Transcript] Using auto-captions for {video_id}")
            segments = _captions_to_segments(captions)

    # Groq Whisper fallback (for videos without auto-captions)
    if not segments:
        print(f"[Transcript] Falling back to Groq Whisper API for {video_id}")
        audio_path = download_audio(url)
        segments = _transcribe_with_groq(audio_path)
        # Clean up downloaded audio
        audio_path.unlink(missing_ok=True)

    _save_transcript(video_id, segments)
    return segments


def get_full_text(segments: List[TranscriptSegment]) -> str:
    """Concatenate all segment texts into one string."""
    return " ".join(s["text"] for s in segments)
