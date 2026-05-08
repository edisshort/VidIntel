"""
VidIntel AI — FastAPI Application Entry Point

Start with:
  uvicorn main:app --reload --port 8000

Swagger docs: http://localhost:8000/docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import APP_TITLE, APP_VERSION, CORS_ORIGINS, FRAMES_DIR
from api.routes import ingest_router, query_router, visual_router
from core.models.schemas import HealthResponse

# ─── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=(
        "Multimodal Video Knowledge Intelligence System — "
        "LLMs + AI Agents + Hybrid RAG + Visual RAG"
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Static files (serve extracted frames) ─────────────────────────────────────
app.mount("/frames", StaticFiles(directory=str(FRAMES_DIR)), name="frames")

# ─── Routers ───────────────────────────────────────────────────────────────────
app.include_router(ingest_router)
app.include_router(query_router)
app.include_router(visual_router)

# ─── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    from core.rag.vector_store import get_chroma_client
    from core.llm.groq_client import get_groq_client
    components = {}

    try:
        get_chroma_client()
        components["chromadb"] = "ok"
    except Exception as e:
        components["chromadb"] = f"error: {e}"

    try:
        get_groq_client()
        components["groq"] = "ok"
    except Exception as e:
        components["groq"] = f"error: {e}"

    try:
        from core.rag.embeddings import get_embedding_model
        get_embedding_model()
        components["embeddings"] = "ok"
    except Exception as e:
        components["embeddings"] = f"error: {e}"

    all_ok = all(v == "ok" for v in components.values())
    return HealthResponse(
        status="healthy" if all_ok else "degraded",
        version=APP_VERSION,
        components=components,
    )


@app.get("/", tags=["Health"])
async def root():
    return {
        "name": APP_TITLE,
        "version": APP_VERSION,
        "docs": "/docs",
        "status": "running",
    }
