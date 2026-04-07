from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from functools import lru_cache
import json

from app.core.logging import get_logger
from app.services.rag_chain import RAGChain
from app.services.memory import get_memory_store
from app.models.schemas import QueryRequest, QueryResponse, SearchRequest, SearchResponse

router = APIRouter(prefix="/api/query", tags=["query"])
logger = get_logger(__name__)


@lru_cache
def get_rag_chain() -> RAGChain:
    return RAGChain()


@router.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    store = get_memory_store()
    session = store.get_or_create(request.session_id, request.namespace)
    history_text = session.get_history_text()

    chain = get_rag_chain()
    result = chain.ask(
        question=request.question,
        namespace=request.namespace,
        top_k=request.top_k,
        temperature=request.temperature,
        history_text=history_text,
    )
    result.session_id = session.id
    session.add_turn("user", request.question)
    session.add_turn("assistant", result.answer, [s.model_dump() for s in result.sources])
    return result


@router.post("/ask/stream")
async def ask_question_stream(request: QueryRequest):
    store = get_memory_store()
    session = store.get_or_create(request.session_id, request.namespace)
    history_text = session.get_history_text()
    chain = get_rag_chain()

    async def event_stream():
        collected_answer = ""
        collected_sources = []
        try:
            # Retrieve
            query_embedding = chain._embedder.embed(request.question)
            matches = chain._vector_store.query(query_embedding, namespace=request.namespace, top_k=request.top_k)

            if not matches:
                yield f"event: error\ndata: {json.dumps({'message': 'No relevant documents found.'})}\n\n"
                return

            context_parts, sources = [], []
            for match in matches:
                meta = match.get("metadata", {})
                text = meta.get("text", "")
                filename = meta.get("filename", "unknown")
                page = meta.get("page", 0)
                chunk_type = meta.get("chunk_type", "text")
                context_parts.append(f"[Source: {filename}, Page {page}]\n{text}")
                src = {
                    "text": text[:400], "filename": filename, "page": page,
                    "score": round(match.get("score", 0.0), 4),
                    "doc_id": meta.get("doc_id", ""),
                    "chunk_type": chunk_type,
                }
                sources.append(src)

            collected_sources = sources

            # Send session_id and sources immediately
            yield f"event: session\ndata: {json.dumps({'session_id': session.id})}\n\n"
            yield f"event: sources\ndata: {json.dumps(sources)}\n\n"

            # Build prompt with history
            context = "\n\n---\n\n".join(context_parts)
            history_section = f"\nConversation history:\n{history_text}\n" if history_text.strip() else ""
            prompt = chain._chain.first.format(
                context=context, question=request.question, history_section=history_section
            )

            from langchain_ollama import OllamaLLM
            from app.core.config import get_settings
            settings = get_settings()
            llm = OllamaLLM(base_url=settings.ollama_base_url,
                            model=settings.ollama_llm_model, temperature=request.temperature)

            async for chunk in llm.astream(prompt):
                if chunk:
                    collected_answer += chunk
                    yield f"event: token\ndata: {json.dumps({'text': chunk})}\n\n"

            yield f"event: done\ndata: {json.dumps({'model': settings.ollama_llm_model, 'session_id': session.id})}\n\n"

        except Exception as e:
            logger.error("stream_error", error=str(e))
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
        finally:
            # Save to memory regardless of streaming outcome
            if collected_answer:
                session.add_turn("user", request.question)
                session.add_turn("assistant", collected_answer, collected_sources)

    return StreamingResponse(event_stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})


@router.post("/search", response_model=SearchResponse)
async def similarity_search(request: SearchRequest):
    chain = get_rag_chain()
    return chain.search(query=request.query, namespace=request.namespace, top_k=request.top_k)