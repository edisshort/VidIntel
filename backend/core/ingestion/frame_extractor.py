"""
VidIntel AI — Frame Extractor
Downloads a video (video-only stream) and extracts frames at a fixed interval
using OpenCV. Saves frames as JPEG images named by timestamp.
"""

from __future__ import annotations

import os
import cv2
import yt_dlp
from pathlib import Path
from typing import List, Optional

from config import FRAMES_DIR
from core.ingestion.video_downloader import sanitize_id


# ─── Download video stream ─────────────────────────────────────────────────────

def _download_video_stream(url: str, out_dir: Path) -> Path:
    """
    Download the lowest-res video stream for frame extraction.
    Uses a robust fallback chain that works for all YouTube video types
    (including those that only have combined audio+video streams).
    """
    video_id = sanitize_id(url)
    out_path = out_dir / f"{video_id}_video.%(ext)s"

    ydl_opts = {
        "format": "worstvideo[ext=mp4]/worstvideo/bestvideo[height<=360][ext=mp4]/bestvideo[height<=360]/bestvideo/worst/best",
        "outtmpl": str(out_path),
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Find the downloaded file (extension may vary)
    for f in out_dir.iterdir():
        if f.stem == f"{video_id}_video":
            return f
    raise FileNotFoundError(f"Video stream not found for {video_id}")


# ─── Frame extraction ──────────────────────────────────────────────────────────

def extract_frames(
    url: str,
    interval_sec: int = 30,
    video_path: Optional[Path] = None,
) -> List[dict]:
    """
    Extract frames from a video at a fixed time interval.

    Args:
        url:          YouTube URL (used for video_id and downloading if needed).
        interval_sec: How often to grab a frame (in seconds).
        video_path:   If already downloaded, pass the path directly.

    Returns:
        List of dicts: {"frame_path": str, "timestamp_seconds": float, "timestamp_label": str}
    """
    video_id = sanitize_id(url)
    frame_dir = FRAMES_DIR / video_id
    frame_dir.mkdir(parents=True, exist_ok=True)

    # Check if frames already exist
    existing = sorted(frame_dir.glob("frame_*.jpg"))
    if existing:
        print(f"[Frames] Using {len(existing)} cached frames for {video_id}")
        return _load_frame_manifest(frame_dir, existing)

    # Download video if not provided
    tmp_dir = FRAMES_DIR / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    if video_path is None:
        print(f"[Frames] Downloading video stream for {video_id}...")
        video_path = _download_video_stream(url, tmp_dir)

    # OpenCV extraction
    print(f"[Frames] Extracting frames every {interval_sec}s...")
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_step = int(fps * interval_sec)

    frames = []
    frame_idx = 0

    while True:
        ret = cap.grab()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            ret, img = cap.retrieve()
            if ret:
                timestamp_sec = frame_idx / fps
                label = _fmt(timestamp_sec)
                fname = f"frame_{int(timestamp_sec):06d}.jpg"
                out_path = frame_dir / fname
                cv2.imwrite(str(out_path), img)
                frames.append(
                    {
                        "frame_path": str(out_path),
                        "timestamp_seconds": timestamp_sec,
                        "timestamp_label": label,
                    }
                )

        frame_idx += 1

    cap.release()

    # Clean up downloaded video
    if video_path and video_path.parent == tmp_dir:
        video_path.unlink(missing_ok=True)

    print(f"[Frames] Extracted {len(frames)} frames for {video_id}")
    return frames


def _fmt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"


def _load_frame_manifest(frame_dir: Path, files: list) -> List[dict]:
    frames = []
    for f in files:
        # Parse timestamp from filename: frame_XXXXXX.jpg
        stem = f.stem  # "frame_000030"
        secs = float(stem.split("_")[1])
        frames.append(
            {
                "frame_path": str(f),
                "timestamp_seconds": secs,
                "timestamp_label": _fmt(secs),
            }
        )
    return frames
