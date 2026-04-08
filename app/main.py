from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

    # Initialize SQLite schema (idempotent — safe on every restart)
    init_db()

    logger.info("app_starting", env=settings.app_env, llm=settings.ollama_llm_model,
                embed=settings.ollama_embed_model, index=settings.pinecone_index_name)
    yield
    logger.info("app_shutdown")


settings = get_settings()
app = FastAPI(
    title="RAG PDF Assistant",
    description="Retrieval-Augmented Generation over PDF documents using Ollama + Pinecone",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(sessions.router)


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    from app.services.embedder import OllamaEmbedder
    from app.services.vector_store import VectorStore
    from app.services.document_registry import get_document_registry
    embedder = OllamaEmbedder()
    vector_store = VectorStore()
    registry = get_document_registry()
    return HealthResponse(
        status="ok",
        ollama="ok" if embedder.check_health() else "unreachable",
        pinecone="ok" if vector_store.check_health() else "unreachable",
    )


@app.get("/", tags=["root"])
async def root():
    return {"message": "RAG PDF Assistant is running. Visit /docs for the API reference."}