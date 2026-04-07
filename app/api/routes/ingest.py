from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from functools import lru_cache

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.pdf_processor import PDFProcessor
from app.services.embedder import OllamaEmbedder
from app.services.vector_store import VectorStore
from app.models.schemas import (
    IngestResponse,
    DeleteResponse,
    ListDocumentsResponse,
    DocumentInfo,
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

    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()

    if len(content) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds max size of {settings.max_file_size_mb}MB.",
        )

    logger.info("upload_received", filename=file.filename, size=len(content), namespace=namespace)

    processor = get_processor()
    embedder = get_embedder()
    vector_store = get_vector_store()

    # Extract and chunk
    doc = processor.process(content, file.filename)

    if not doc.chunks:
        raise HTTPException(status_code=422, detail="Could not extract text from PDF.")

    # Embed all chunks
    texts = [chunk.text for chunk in doc.chunks]
    embeddings = embedder.embed_batch(texts)

    # Store in Pinecone
    upserted = vector_store.upsert_chunks(doc.chunks, embeddings, namespace=namespace)

    return IngestResponse(
        doc_id=doc.doc_id,
        filename=doc.filename,
        namespace=namespace,
        chunks_indexed=upserted,
        pages=doc.pages,
        message=f"Successfully indexed {upserted} chunks from {doc.pages} pages.",
    )


@router.delete("/delete/{doc_id}", response_model=DeleteResponse)
async def delete_document(doc_id: str, namespace: str = "default"):
    vector_store = get_vector_store()
    vector_store.delete_document(doc_id, namespace=namespace)
    return DeleteResponse(doc_id=doc_id, message=f"Document {doc_id} deleted from namespace '{namespace}'.")


@router.get("/list", response_model=ListDocumentsResponse)
async def list_documents(namespace: str = "default"):
    vector_store = get_vector_store()
    from datetime import datetime, timezone

    raw_docs = vector_store.list_documents(namespace=namespace)
    documents = [
        DocumentInfo(
            doc_id=d["doc_id"],
            filename=d["filename"],
            namespace=namespace,
            chunks=0,  # Pinecone doesn't cheaply count per-doc vectors
            pages=0,
            indexed_at=datetime.fromisoformat(d["indexed_at"]) if d.get("indexed_at") else datetime.now(timezone.utc),
        )
        for d in raw_docs
    ]
    return ListDocumentsResponse(namespace=namespace, documents=documents, total=len(documents))