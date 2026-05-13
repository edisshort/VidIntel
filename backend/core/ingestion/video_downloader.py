"""
VidIntel AI — Video Downloader
Uses yt-dlp to extract metadata and audio from YouTube (and other platforms).
"""

import json
import os
import re
from pathlib import Path
from typing import Optional
import yt_dlp

from config import TRANSCRIPTS_DIR, YOUTUBE_COOKIES_FILE

_SERVER_MODE = os.getenv("SERVER_MODE", "false").lower() == "true"


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
    import hashlib
    return hashlib.md5(url.encode()).hexdigest()[:11]


def _base_opts() -> dict:
    """Base yt-dlp options shared across all calls (includes cookies if set)."""
    opts = {"quiet": True, "no_warnings": True}
    if YOUTUBE_COOKIES_FILE:
        opts["cookiefile"] = YOUTUBE_COOKIES_FILE
    return opts


def _metadata_via_oembed(url: str) -> Optional[dict]:
    """Fetch basic metadata via YouTube oEmbed API — no auth, works from any IP."""
    try:
        import urllib.request
        import urllib.parse
        oembed_url = f"https://www.youtube.com/oembed?url={urllib.parse.quote(url)}&format=json"
        with urllib.request.urlopen(oembed_url, timeout=10) as resp:
            data = json.loads(resp.read())
        video_id = sanitize_id(url)
        return {
            "video_id": video_id,
            "title": data.get("title", "Unknown Title"),
            "channel": data.get("author_name", "Unknown Channel"),
            "duration_seconds": 0,  # oEmbed doesn't provide duration
            "url": url,
            "thumbnail_url": data.get("thumbnail_url"),
            "description": "",
            "upload_date": "",
            "view_count": 0,
        }
    except Exception as e:
        print(f"[Downloader] oEmbed fallback failed: {e}")
        return None


def get_video_metadata(url: str) -> dict:
    """Fetch video metadata. Uses oEmbed first (works from any IP), yt-dlp as fallback."""
    # oEmbed is primary — no auth, no bot detection, works on all servers
    meta = _metadata_via_oembed(url)
    if meta:
        return meta

    # yt-dlp fallback — skip entirely in server mode (blocked by YouTube)
    if _SERVER_MODE:
        video_id = sanitize_id(url)
        print(f"[Downloader] SERVER_MODE: oEmbed failed, using minimal metadata for {video_id}")
        return {
            "video_id": video_id,
            "title": f"Video {video_id}",
            "channel": "Unknown",
            "duration_seconds": 0,
            "url": url,
            "thumbnail_url": None,
            "description": "",
            "upload_date": "",
            "view_count": 0,
        }

    try:
        ydl_opts = {**_base_opts(), "skip_download": True, "writeinfojson": False}
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
    except Exception as e:
        print(f"[Downloader] yt-dlp metadata also failed ({e}), using minimal metadata")
        return {
            "video_id": sanitize_id(url),
            "title": "Unknown Title",
            "channel": "Unknown Channel",
            "duration_seconds": 0,
            "url": url,
            "thumbnail_url": None,
            "description": "",
            "upload_date": "",
            "view_count": 0,
        }


def download_audio(url: str, output_dir: Optional[Path] = None) -> Path:
    """Download audio-only stream for transcription."""
    out_dir = output_dir or TRANSCRIPTS_DIR
    video_id = sanitize_id(url)
    out_path = out_dir / f"{video_id}.%(ext)s"

    ydl_opts = {
        **_base_opts(),
        "format": "bestaudio/best",
        "outtmpl": str(out_path),
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
        **_base_opts(),
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
