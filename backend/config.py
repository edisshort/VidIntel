"""
VidIntel AI — Centralised configuration loaded from environment variables.
Copy .env.example to .env and fill in your keys before running.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Look for .env in backend/ first, then fall back to parent (project root)
_backend_env = Path(__file__).parent / ".env"
_root_env = Path(__file__).parent.parent / ".env"

if _backend_env.exists():
    load_dotenv(_backend_env)
elif _root_env.exists():
    load_dotenv(_root_env)
else:
    load_dotenv()  # fallback: search standard locations

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FRAMES_DIR = DATA_DIR / "frames"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
CHROMA_DIR = DATA_DIR / "chroma"

# Create dirs on import
for d in [DATA_DIR, FRAMES_DIR, TRANSCRIPTS_DIR, CHROMA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Groq / LLM ────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MAX_TOKENS: int = int(os.getenv("GROQ_MAX_TOKENS", "4096"))
GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0.1"))

# ─── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
)

# ─── Retrieval ─────────────────────────────────────────────────────────────────
DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))
HYBRID_ALPHA: float = float(os.getenv("HYBRID_ALPHA", "0.5"))
# alpha=0 → pure BM25 ; alpha=1 → pure vector ; 0.5 → equal blend

# ─── OCR / Frame Extraction ────────────────────────────────────────────────────
FRAME_INTERVAL_SEC: int = int(os.getenv("FRAME_INTERVAL_SEC", "30"))
OCR_ENGINE: str = os.getenv("OCR_ENGINE", "easyocr")   # "easyocr" | "tesseract"
OCR_MIN_CONFIDENCE: float = float(os.getenv("OCR_MIN_CONFIDENCE", "0.5"))

# ─── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8001"))
CHROMA_MODE: str = os.getenv("CHROMA_MODE", "local")   # "local" | "remote"

# ─── YouTube Cookies (for server deployments blocked by YouTube bot detection) ─
# Paste full cookies.txt content into the YOUTUBE_COOKIES env var on Render.
# Export from your browser using "Get cookies.txt LOCALLY" Chrome extension.
import tempfile as _tempfile

YOUTUBE_COOKIES_FILE: Optional[str] = None
_cookies_content = os.getenv("YOUTUBE_COOKIES", "").strip()
if _cookies_content:
    _cookie_file = _tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    )
    _cookie_file.write(_cookies_content)
    _cookie_file.flush()
    YOUTUBE_COOKIES_FILE = _cookie_file.name

# ─── App ───────────────────────────────────────────────────────────────────────
APP_VERSION: str = "1.0.0"
APP_TITLE: str = "VidIntel AI"
CORS_ORIGINS: list = [
    o.strip() for o in
    os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]
