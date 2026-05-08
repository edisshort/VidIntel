"""
VidIntel AI — OCR Processor
Runs EasyOCR (or Tesseract as fallback) on extracted frames.
Produces text chunks with timestamps for Visual RAG.
"""

import json
import re
from pathlib import Path
from typing import List, Optional

from config import OCR_ENGINE, OCR_MIN_CONFIDENCE, FRAMES_DIR
from core.ingestion.video_downloader import sanitize_id

# ─── YouTube UI noise filter ───────────────────────────────────────────────────
# These words/phrases appear in almost every YouTube frame and add no signal
_NOISE_WORDS = {
    "subscribe", "like", "share", "comment", "notification", "bell",
    "views", "subscribers", "youtube", "watch", "follow", "paused",
    "cc", "settings", "fullscreen", "autoplay", "skip", "ad",
    "thumbs", "dislike", "save", "clip", "thanks", "join",
    "months ago", "years ago", "days ago", "hours ago", "ago",
}

def _is_noise(text: str) -> bool:
    """Return True if OCR text is just YouTube UI noise."""
    cleaned = text.lower().strip()
    # Too short to be meaningful
    if len(cleaned) < 4:
        return True
    # Only numbers (timestamps, view counts)
    if re.match(r'^[\d\s:,\.KMB%]+$', cleaned):
        return True
    # Pure noise word
    if cleaned in _NOISE_WORDS:
        return True
    return False


# ─── Lazy-load OCR engine ──────────────────────────────────────────────────────

_easyocr_reader = None


def _get_easyocr():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        print("[OCR] Loading EasyOCR model...")
        _easyocr_reader = easyocr.Reader(["en"], gpu=False)
    return _easyocr_reader


def _ocr_easyocr(image_path: str) -> List[dict]:
    """Run EasyOCR on one image. Returns list of {text, confidence}."""
    reader = _get_easyocr()
    results = reader.readtext(image_path, detail=1)
    texts = []
    for (_, text, conf) in results:
        if conf >= OCR_MIN_CONFIDENCE and text.strip():
            texts.append({"text": text.strip(), "confidence": round(conf, 3)})
    return texts


def _ocr_tesseract(image_path: str) -> List[dict]:
    """Run Tesseract on one image."""
    import pytesseract
    from PIL import Image

    img = Image.open(image_path)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    texts = []
    for i, word in enumerate(data["text"]):
        conf = int(data["conf"][i])
        if conf >= int(OCR_MIN_CONFIDENCE * 100) and word.strip():
            texts.append({"text": word.strip(), "confidence": conf / 100.0})
    return texts


def _run_ocr(image_path: str) -> List[dict]:
    if OCR_ENGINE == "tesseract":
        return _ocr_tesseract(image_path)
    return _ocr_easyocr(image_path)


# ─── Public API ────────────────────────────────────────────────────────────────

def process_frames_ocr(
    video_id: str,
    frames: List[dict],
) -> List[dict]:
    """
    Run OCR on a list of frame dicts (from frame_extractor).

    Returns list of OCR chunks:
      {
        "video_id": str,
        "frame_path": str,
        "timestamp_seconds": float,
        "timestamp_label": str,
        "ocr_text": str,           # all text merged from this frame
        "raw_detections": list,    # individual word detections
        "source": "ocr"
      }
    """
    cache_path = FRAMES_DIR / video_id / "ocr_results.json"

    # Return cached results if available
    if cache_path.exists():
        print(f"[OCR] Using cached OCR results for {video_id}")
        with open(cache_path) as f:
            return json.load(f)

    ocr_chunks = []
    total = len(frames)
    print(f"[OCR] Processing {total} frames for {video_id}...")

    for i, frame in enumerate(frames):
        if i % 10 == 0:
            print(f"[OCR] {i}/{total}...")

        detections = _run_ocr(frame["frame_path"])
        if not detections:
            continue

        # Filter out YouTube UI noise before indexing
        detections = [d for d in detections if not _is_noise(d["text"])]
        if not detections:
            continue

        merged_text = " ".join(d["text"] for d in detections)

        ocr_chunks.append(
            {
                "video_id": video_id,
                "frame_path": frame["frame_path"],
                "timestamp_seconds": frame["timestamp_seconds"],
                "timestamp_label": frame["timestamp_label"],
                "ocr_text": merged_text,
                "raw_detections": detections,
                "source": "ocr",
            }
        )

    # Cache to disk
    with open(cache_path, "w") as f:
        json.dump(ocr_chunks, f, indent=2)

    print(f"[OCR] Completed. {len(ocr_chunks)} frames had text.")
    return ocr_chunks


def ocr_chunks_to_documents(ocr_chunks: List[dict]) -> List[dict]:
    """
    Convert OCR results into document format compatible with the RAG pipeline.
    Each chunk becomes a document with metadata.
    """
    docs = []
    for chunk in ocr_chunks:
        if not chunk["ocr_text"].strip():
            continue
        docs.append(
            {
                "text": chunk["ocr_text"],
                "metadata": {
                    "video_id": chunk["video_id"],
                    "timestamp_seconds": chunk["timestamp_seconds"],
                    "timestamp_label": chunk["timestamp_label"],
                    "frame_path": chunk["frame_path"],
                    "source": "ocr",
                },
            }
        )
    return docs
