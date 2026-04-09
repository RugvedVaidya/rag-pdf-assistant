from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger
from app.core.database import init_db
from app.api.routes import ingest, query, sessions
from app.models.schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger = get_logger("startup")
    settings = get_settings()
    init_db()
    logger.info("app_starting", env=settings.app_env,
                llm=settings.groq_model, index=settings.pinecone_index_name)
    yield
    logger.info("app_shutdown")


app = FastAPI(
    title="RAG PDF Assistant",
    description="Retrieval-Augmented Generation over PDF documents — Groq + Pinecone + FastEmbed",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# API routes
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(sessions.router)

# Serve frontend at root
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    from app.services.embedder import get_embedder
    from app.services.vector_store import VectorStore
    from app.core.config import get_settings
    cfg = get_settings()
    embedder = get_embedder()
    vector_store = VectorStore()
    return HealthResponse(
        status="ok",
        llm=f"groq/{cfg.groq_model}",
        pinecone="ok" if vector_store.check_health() else "unreachable",
    )