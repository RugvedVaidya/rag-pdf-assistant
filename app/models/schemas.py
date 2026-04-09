from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    namespace: str
    chunks_indexed: int
    pages: int
    table_count: int = 0
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

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    namespace: str = Field(default="default")
    top_k: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    session_id: Optional[str] = None

class SourceChunk(BaseModel):
    text: str
    filename: str
    page: int
    score: float
    doc_id: str
    chunk_type: str = "text"

class QueryResponse(BaseModel):
    answer: str
    question: str
    sources: list[SourceChunk]
    model: str
    namespace: str
    session_id: Optional[str] = None
    rewritten_query: Optional[str] = None   # set when query was rewritten
    was_rewritten: bool = False

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3)
    namespace: str = Field(default="default")
    top_k: int = Field(default=5, ge=1, le=20)

class SearchResponse(BaseModel):
    query: str
    results: list[SourceChunk]
    namespace: str

class SessionCreate(BaseModel):
    namespace: str = "default"
    title: str = "New chat"

class SessionInfo(BaseModel):
    id: str
    title: str
    namespace: str
    turn_count: int
    created_at: str
    updated_at: str

class TurnInfo(BaseModel):
    role: str
    content: str
    sources: list = []
    timestamp: str

class SessionDetail(BaseModel):
    id: str
    title: str
    namespace: str
    turns: list[TurnInfo]
    created_at: str
    updated_at: str

class DeleteSessionResponse(BaseModel):
    session_id: str
    message: str

class HealthResponse(BaseModel):
    status: str
    llm: str
    pinecone: str
    version: str = "1.0.0"