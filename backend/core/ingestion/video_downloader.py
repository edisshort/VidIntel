"""
VidIntel AI — Video Downloader
Uses yt-dlp to extract metadata and audio from YouTube (and other platforms).
"""

import json
import subprocess
import re
from pathlib import Path
from typing import Optional
import yt_dlp

from config import TRANSCRIPTS_DIR


def sanitize_id(url: str) -> str:
    """Extract a safe video ID from a YouTube URL."""
    patterns = [
        r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:embed/)([A-Za-z0-9_-]{11})",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    # Fallback: hash the URL
    import hashlib
    return hashlib.md5(url.encode()).hexdigest()[:11]


def get_video_metadata(url: str) -> dict:
    """Fetch video metadata without downloading the video."""
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "writeinfojson": False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return {
            "video_id": info.get("id", sanitize_id(url)),
            "title": info.get("title", "Unknown Title"),
            "channel": info.get("uploader", "Unknown Channel"),
            "duration_seconds": info.get("duration", 0),
            "url": url,
            "thumbnail_url": info.get("thumbnail"),
            "description": info.get("description", ""),
            "upload_date": info.get("upload_date", ""),
            "view_count": info.get("view_count", 0),
        }


def download_audio(url: str, output_dir: Optional[Path] = None) -> Path:
    """
    Download audio-only stream for transcription.
    Returns path to the downloaded .mp3 file.
    """
    out_dir = output_dir or TRANSCRIPTS_DIR
    video_id = sanitize_id(url)
    out_path = out_dir / f"{video_id}.%(ext)s"

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_path),
        "quiet": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return out_dir / f"{video_id}.mp3"


def get_auto_captions(url: str) -> Optional[list]:
    """
    Try to pull auto-generated captions (with timestamps) directly from YouTube.
    Returns list of {"start": float, "text": str} dicts or None.
    """
    video_id = sanitize_id(url)
    ydl_opts = {
        "quiet": True,
        "writeautomaticsub": True,
        "subtitlesformat": "json3",
        "subtitleslangs": ["en"],
        "skip_download": True,
        "outtmpl": str(TRANSCRIPTS_DIR / f"{video_id}"),
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        sub_file = TRANSCRIPTS_DIR / f"{video_id}.en.json3"
        if sub_file.exists():
            with open(sub_file) as f:
                data = json.load(f)
            events = data.get("events", [])
            segments = []
            for event in events:
                if "segs" not in event:
                    continue
                start = event.get("tStartMs", 0) / 1000.0
                text = "".join(s.get("utf8", "") for s in event["segs"]).strip()
                if text:
                    segments.append({"start": start, "text": text})
            return segments
    except Exception:
        return None
