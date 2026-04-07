from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# --- Ingest ---

class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    namespace: str
    chunks_indexed: int
    pages: int
    message: str


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    namespace: str
    chunks: int
    pages: int
    indexed_at: datetime


class ListDocumentsResponse(BaseModel):
    namespace: str
    documents: list[DocumentInfo]
    total: int


class DeleteResponse(BaseModel):
    doc_id: str
    message: str


# --- Query ---

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    namespace: str = Field(default="default")
    top_k: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    session_id: Optional[str] = None   # ← this line must be present


class SourceChunk(BaseModel):
    text: str
    filename: str
    page: int
    score: float
    doc_id: str


class QueryResponse(BaseModel):
    answer: str
    question: str
    sources: list[SourceChunk]
    model: str
    namespace: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3)
    namespace: str = Field(default="default")
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResponse(BaseModel):
    query: str
    results: list[SourceChunk]
    namespace: str


# --- Health ---

class HealthResponse(BaseModel):
    status: str
    ollama: str
    pinecone: str
    version: str = "1.0.0"
    
# --- Sessions ---

class SessionCreate(BaseModel):
    namespace: str = "default"
    title: Optional[str] = None


class TurnInfo(BaseModel):
    role: str
    content: str
    sources: list = []
    timestamp: datetime


class SessionInfo(BaseModel):
    id: str
    title: Optional[str] = None
    namespace: str
    created_at: datetime
    updated_at: datetime


class SessionDetail(BaseModel):
    id: str
    title: Optional[str] = None
    namespace: str
    turns: list[TurnInfo]
    created_at: datetime
    updated_at: datetime


class DeleteSessionResponse(BaseModel):
    session_id: str
    message: str