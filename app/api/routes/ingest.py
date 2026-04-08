from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from functools import lru_cache
from datetime import datetime

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.pdf_processor import PDFProcessor
from app.services.embedder import OllamaEmbedder
from app.services.vector_store import VectorStore
from app.services.document_registry import get_document_registry
from app.models.schemas import (
    IngestResponse, DeleteResponse, ListDocumentsResponse, DocumentInfo,
)

router = APIRouter(prefix="/api/ingest", tags=["ingest"])
logger = get_logger(__name__)


@lru_cache
def get_processor(): return PDFProcessor()

@lru_cache
def get_embedder(): return OllamaEmbedder()

@lru_cache
def get_vector_store(): return VectorStore()


@router.post("/upload", response_model=IngestResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    namespace: str = Form(default="default"),
):
    settings = get_settings()

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()
    if len(content) > settings.max_file_size_bytes:
        raise HTTPException(status_code=413,
            detail=f"File exceeds max size of {settings.max_file_size_mb}MB.")

    registry = get_document_registry()
    vector_store = get_vector_store()

    # ── Duplicate detection via SQLite registry (reliable) ──
    duplicates = registry.find_by_filename(file.filename, namespace)
    replaced = False
    if duplicates:
        logger.info("duplicate_detected", filename=file.filename, count=len(duplicates))
        for d in duplicates:
            vector_store.delete_document(d["doc_id"], namespace=namespace)
            registry.delete(d["doc_id"])
        replaced = True

    logger.info("upload_received", filename=file.filename, size=len(content), namespace=namespace)

    processor = get_processor()
    embedder = get_embedder()

    doc = processor.process(content, file.filename)
    if not doc.chunks:
        raise HTTPException(status_code=422, detail="Could not extract text from PDF.")

    texts = [chunk.text for chunk in doc.chunks]
    embeddings = embedder.embed_batch(texts)
    upserted = vector_store.upsert_chunks(doc.chunks, embeddings, namespace=namespace)

    # ── Register in SQLite ──
    registry.register(
        doc_id=doc.doc_id,
        filename=doc.filename,
        namespace=namespace,
        chunks=upserted,
        pages=doc.pages,
        table_count=doc.table_count,
        file_size=len(content),
    )

    msg = (f"Re-indexed {upserted} chunks from {doc.pages} pages (replaced old version)."
           if replaced else
           f"Successfully indexed {upserted} chunks from {doc.pages} pages.")

    return IngestResponse(
        doc_id=doc.doc_id,
        filename=doc.filename,
        namespace=namespace,
        chunks_indexed=upserted,
        table_count=doc.table_count,
        pages=doc.pages,
        message=msg,
    )


@router.delete("/delete/{doc_id}", response_model=DeleteResponse)
async def delete_document(doc_id: str, namespace: str = "default"):
    vector_store = get_vector_store()
    registry = get_document_registry()
    vector_store.delete_document(doc_id, namespace=namespace)
    registry.delete(doc_id)
    return DeleteResponse(doc_id=doc_id,
                          message=f"Document {doc_id} deleted from namespace '{namespace}'.")


@router.get("/list", response_model=ListDocumentsResponse)
async def list_documents(namespace: str = "default"):
    """List documents from SQLite registry — fast and reliable."""
    registry = get_document_registry()
    docs = registry.list_by_namespace(namespace)
    documents = [
        DocumentInfo(
            doc_id=d["doc_id"],
            filename=d["filename"],
            namespace=d["namespace"],
            chunks=d["chunks"],
            pages=d["pages"],
            indexed_at=datetime.fromisoformat(d["indexed_at"]),
        )
        for d in docs
    ]
    return ListDocumentsResponse(namespace=namespace, documents=documents, total=len(documents))