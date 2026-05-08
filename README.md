# VidIntel AI

Multimodal video knowledge intelligence — ask questions across YouTube videos, find exact timestamps, search what's shown on screen, and compare creator opinions.

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Frontend-Next.js%2014-black?style=flat-square)](https://nextjs.org)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.3%2070B-F55036?style=flat-square)](https://groq.com)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-8B5CF6?style=flat-square)](https://trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

---

## Features

| Mode | What it does |
|------|-------------|
| **AI Research** | Orchestrator routes your query to the right agent automatically |
| **Timestamp Search** | Hybrid RAG returns exact moments in the video with clickable timestamps |
| **Consensus Analysis** | Cross-video comparison — what do multiple creators agree or disagree on? |
| **Visual / OCR Search** | Finds text *shown on screen* in frames, not just spoken words |

---

## Architecture

```
YouTube URL
    ├── yt-dlp ──► captions / audio ──► Groq Whisper (fallback)
    │                                        │
    │                                   Chunker (500 tok, 50 overlap)
    │                                        ├── ChromaDB (BGE embeddings)
    │                                        └── BM25 Index
    │
    └── yt-dlp ──► video ──► OpenCV frames ──► EasyOCR + noise filter
                                                     └── indexed above

User Query
    └── Research Agent (orchestrator)
            ├── Retrieval Agent  →  Hybrid RRF (BM25 + vector) → LLM answer
            ├── Visual Agent     →  BM25 + semantic + LLM reranking → frames
            └── Consensus Agent  →  multi-video extraction → agreements/disagreements
```

**Hybrid retrieval** fuses BM25 and vector search rankings via Reciprocal Rank Fusion:
`score = α × 1/(k + rank_vec) + (1-α) × 1/(k + rank_bm25)`

**Visual reranking** retrieves 15 candidate frames then asks the LLM to score each 0–10 for relevance, filtering out irrelevant frames so different queries return genuinely different results.

---

## Stack

**Backend** — FastAPI · Groq API (Llama 3.3 70B + Whisper) · ChromaDB · sentence-transformers (BGE-small) · rank-bm25 · yt-dlp · OpenCV · EasyOCR · Pydantic v2

**Frontend** — Next.js 14 · Tailwind CSS · TypeScript

---

## Quick Start

**Prerequisites:** Python 3.11+, Node.js 18+, [Groq API key](https://console.groq.com) (free)

```bash
# 1. Configure
cp .env.example .env
# Add your GROQ_API_KEY to .env

# 2. Backend
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --port 8000
# → http://localhost:8000/docs

# 3. Frontend
cd frontend
npm install && npm run dev
# → http://localhost:3000

# Or run both with Docker
docker-compose up --build
```

---

## Configuration

Key settings in `.env`:

```env
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
HYBRID_ALPHA=0.5          # 0 = pure BM25 | 1 = pure vector
FRAME_INTERVAL_SEC=30     # OCR frame sampling rate
OCR_ENGINE=easyocr        # easyocr | tesseract
```

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ingest` | Ingest a single YouTube video |
| POST | `/ingest/batch` | Ingest multiple videos into one collection |
| POST | `/query` | AI Research or Timestamp Search |
| POST | `/consensus` | Cross-video consensus analysis |
| POST | `/visual-search` | OCR / visual frame search |
| GET | `/library` | List all collections and videos |
| GET | `/health` | Service health check |

Full interactive docs at `http://localhost:8000/docs`.

---

## License

MIT
