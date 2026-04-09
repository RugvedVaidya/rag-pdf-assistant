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
    store.save_turn(session.id, "user", request.question, [])
    store.save_turn(session.id, "assistant", result.answer,
                    [s.model_dump() for s in result.sources])
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
            # 1. Rewrite query
            search_query, was_rewritten = chain._rewriter.rewrite(
                request.question, history_text
            )
            if was_rewritten:
                yield f"event: rewrite\ndata: {json.dumps({'original': request.question, 'rewritten': search_query})}\n\n"

            # 2. Retrieve + deduplicate
            query_embedding = chain._embedder.embed(search_query)
            raw_matches = chain._vector_store.query(
                query_embedding,
                namespace=request.namespace,
                top_k=request.top_k + 3,
            )
            matches = chain._deduplicate_matches(raw_matches)[:request.top_k]

            if not matches:
                yield f"event: error\ndata: {json.dumps({'message': 'No relevant documents found.'})}\n\n"
                return

            # 3. Build context and sources
            context_parts, sources = [], []
            for i, match in enumerate(matches, start=1):
                meta = match.get("metadata", {})
                text = meta.get("text", "")
                filename = meta.get("filename", "unknown")
                page = meta.get("page", 0)
                chunk_type = meta.get("chunk_type", "text")
                context_parts.append(f"[{i}] Source: {filename}, Page {page}\n{text}")
                sources.append({
                    "text": text[:400], "filename": filename, "page": page,
                    "score": round(match.get("score", 0.0), 4),
                    "doc_id": meta.get("doc_id", ""),
                    "chunk_type": chunk_type,
                })
            collected_sources = sources

            # 4. Emit session + sources immediately
            yield f"event: session\ndata: {json.dumps({'session_id': session.id})}\n\n"
            yield f"event: sources\ndata: {json.dumps(sources)}\n\n"

            # 5. Build context string and history
            context = "\n\n---\n\n".join(context_parts)
            history_section = (
                f"Conversation history (for follow-up context):\n{history_text}\n"
                if history_text.strip() else ""
            )

            # 6. Stream via the RAG chain directly
            # chain._chain is: ChatPromptTemplate | ChatGroq | StrOutputParser
            # We pass the same variables and stream the output
            from app.core.config import get_settings
            settings = get_settings()

            async for chunk in chain._chain.astream({
                "context": context,
                "question": request.question,
                "history_section": history_section,
            }):
                if chunk:
                    collected_answer += chunk
                    yield f"event: token\ndata: {json.dumps({'text': chunk})}\n\n"

            yield f"event: done\ndata: {json.dumps({'model': settings.groq_model, 'session_id': session.id, 'was_rewritten': was_rewritten})}\n\n"

        except Exception as e:
            logger.error("stream_error", error=str(e))
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
        finally:
            if collected_answer:
                store.save_turn(session.id, "user", request.question, [])
                store.save_turn(session.id, "assistant", collected_answer, collected_sources)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )


@router.post("/search", response_model=SearchResponse)
async def similarity_search(request: SearchRequest):
    chain = get_rag_chain()
    return chain.search(query=request.query, namespace=request.namespace, top_k=request.top_k)