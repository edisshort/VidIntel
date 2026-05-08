"""
VidIntel AI — Transcript Extractor
Strategy (in order):
  1. youtube-transcript-api  — hits YouTube's timedtext API, works from servers
  2. yt-dlp auto-captions    — fallback if transcript API fails
  3. Groq Whisper API        — last resort for videos with no captions at all
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


TranscriptSegment = dict  # {"start": float, "end": float, "text": str}


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


# ─── Method 1: youtube-transcript-api ─────────────────────────────────────────

def _fetch_via_transcript_api(video_id: str) -> Optional[List[TranscriptSegment]]:
    """
    Use youtube-transcript-api to fetch captions.
    Works from server IPs — uses YouTube's timedtext endpoint, not the main site.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Try manual English first, then auto-generated
        transcript = None
        try:
            transcript = transcript_list.find_manually_created_transcript(["en"])
        except Exception:
            pass
        if not transcript:
            try:
                transcript = transcript_list.find_generated_transcript(["en"])
            except Exception:
                pass
        if not transcript:
            # Take whatever is available and translate
            transcript = next(iter(transcript_list))

        data = transcript.fetch()
        segments = []
        for i, entry in enumerate(data):
            start = float(entry.get("start", 0))
            duration = float(entry.get("duration", 2.0))
            text = entry.get("text", "").strip()
            if not text:
                continue
            segments.append({
                "start": start,
                "end": start + duration,
                "text": text,
                "timestamp_label": _format_timestamp(start),
                "source": "youtube-transcript-api",
            })

        print(f"[Transcript] youtube-transcript-api: {len(segments)} segments")
        return segments if segments else None

    except Exception as e:
        print(f"[Transcript] youtube-transcript-api failed: {e}")
        return None


# ─── Method 2: yt-dlp captions ────────────────────────────────────────────────

def _captions_to_segments(captions: list) -> List[TranscriptSegment]:
    segments = []
    for i, cap in enumerate(captions):
        start = cap["start"]
        end = captions[i + 1]["start"] if i + 1 < len(captions) else start + 3.0
        segments.append({
            "start": start,
            "end": end,
            "text": cap["text"].strip(),
            "timestamp_label": _format_timestamp(start),
            "source": "captions",
        })
    return segments


# ─── Method 3: Groq Whisper ───────────────────────────────────────────────────

def _transcribe_with_groq(audio_path: Path) -> List[TranscriptSegment]:
    from groq import Groq

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set.")

    client = Groq(api_key=GROQ_API_KEY)
    print(f"[Whisper/Groq] Transcribing {audio_path.name}...")

    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            file=(audio_path.name, f),
            model="whisper-large-v3",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    segments = []
    raw_segments = getattr(response, "segments", None) or []
    for seg in raw_segments:
        start = seg.get("start", 0.0) if isinstance(seg, dict) else getattr(seg, "start", 0.0)
        end   = seg.get("end", start + 2.0) if isinstance(seg, dict) else getattr(seg, "end", start + 2.0)
        text  = seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")
        segments.append({
            "start": float(start),
            "end": float(end),
            "text": text.strip(),
            "timestamp_label": _format_timestamp(float(start)),
            "source": "whisper-groq",
        })

    if not segments and hasattr(response, "text") and response.text:
        segments.append({
            "start": 0.0, "end": 0.0,
            "text": response.text.strip(),
            "timestamp_label": "0:00",
            "source": "whisper-groq",
        })

    return segments


# ─── Public API ────────────────────────────────────────────────────────────────

def extract_transcript(url: str, force_whisper: bool = False) -> List[TranscriptSegment]:
    """
    Main entry point. Tries three methods in order, caches result to disk.
    """
    video_id = sanitize_id(url)

    cached = _load_cached(video_id)
    if cached:
        print(f"[Transcript] Using cached transcript for {video_id}")
        return cached

    segments: List[TranscriptSegment] = []

    if not force_whisper:
        # Method 1 — youtube-transcript-api (server-safe)
        segments = _fetch_via_transcript_api(video_id) or []

        # Method 2 — yt-dlp captions fallback
        if not segments:
            print(f"[Transcript] Trying yt-dlp captions for {video_id}")
            captions = get_auto_captions(url)
            if captions:
                segments = _captions_to_segments(captions)

    # Method 3 — Groq Whisper (needs audio download + ffmpeg)
    if not segments:
        print(f"[Transcript] Falling back to Groq Whisper for {video_id}")
        audio_path = download_audio(url)
        segments = _transcribe_with_groq(audio_path)
        audio_path.unlink(missing_ok=True)

    _save_transcript(video_id, segments)
    return segments


def get_full_text(segments: List[TranscriptSegment]) -> str:
    return " ".join(s["text"] for s in segments)
