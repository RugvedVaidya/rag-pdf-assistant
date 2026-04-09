"""
RAG chain using Groq for LLM inference and FastEmbed for embeddings.
Everything else — retrieval, dedup, history, rewriting — unchanged.
"""
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.embedder import Embedder
from app.services.vector_store import VectorStore
from app.services.query_rewriter import get_query_rewriter
from app.models.schemas import QueryResponse, SourceChunk, SearchResponse

logger = get_logger(__name__)

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are a knowledgeable assistant helping users understand their PDF documents.
{history_section}
Use the context passages below to answer the question. Follow these guidelines:

1. Synthesize information across passages — do not repeat the same point multiple times.
2. If the context covers the question well, answer from it and cite sources (filename + page).
3. If the context is only partially relevant, use what is there and clearly note what is missing.
4. If the context contains no relevant information at all, say so briefly.
5. Be concise. Prefer one clear answer over multiple hedged restatements.

Context passages:
{context}"""),
    ("human", "{question}"),
])


class RAGChain:
    def __init__(self):
        settings = get_settings()
        self._embedder = Embedder()
        self._vector_store = VectorStore()
        self._rewriter = get_query_rewriter()
        self._llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=0.1,
            streaming=True,
        )
        self._chain = RAG_PROMPT | self._llm | StrOutputParser()

    def _deduplicate_matches(self, matches: list) -> list:
        if not matches:
            return matches

        def overlap_ratio(a: str, b: str) -> float:
            words_a = set(a.lower().split())
            words_b = set(b.lower().split())
            if not words_a or not words_b:
                return 0.0
            return len(words_a & words_b) / min(len(words_a), len(words_b))

        kept = []
        for match in matches:
            meta = match.get("metadata", {})
            text = meta.get("text", "")
            page = meta.get("page", 0)
            score = match.get("score", 0.0)
            is_dup = False
            for k in kept:
                km = k.get("metadata", {})
                if (overlap_ratio(text, km.get("text", "")) > 0.6 or
                        (km.get("page") == page and abs(score - k.get("score", 0)) < 0.01)):
                    is_dup = True
                    break
            if not is_dup:
                kept.append(match)

        removed = len(matches) - len(kept)
        if removed > 0:
            logger.info("dedup_removed_chunks", removed=removed, kept=len(kept))
        return kept

    def _build_context(self, matches: list) -> tuple[str, list[SourceChunk]]:
        context_parts, sources = [], []
        for i, match in enumerate(matches, start=1):
            meta = match.get("metadata", {})
            text = meta.get("text", "")
            filename = meta.get("filename", "unknown")
            page = meta.get("page", 0)
            score = match.get("score", 0.0)
            chunk_type = meta.get("chunk_type", "text")
            context_parts.append(f"[{i}] Source: {filename}, Page {page}\n{text}")
            sources.append(SourceChunk(
                text=text[:500], filename=filename, page=page,
                score=round(score, 4), doc_id=meta.get("doc_id", ""),
                chunk_type=chunk_type,
            ))
        return "\n\n---\n\n".join(context_parts), sources

    def ask(self, question: str, namespace: str = "default",
            top_k: int = 5, temperature: float = 0.1,
            history_text: str = "") -> QueryResponse:
        logger.info("rag_query", question=question[:80], namespace=namespace)

        search_query, was_rewritten = self._rewriter.rewrite(question, history_text)

        query_embedding = self._embedder.embed(search_query)
        raw_matches = self._vector_store.query(
            query_embedding, namespace=namespace, top_k=top_k + 3
        )

        if not raw_matches:
            return QueryResponse(
                answer="No relevant documents found in the specified namespace.",
                question=question, sources=[], model=self._llm.model_name,
                namespace=namespace,
                rewritten_query=search_query if was_rewritten else None,
                was_rewritten=was_rewritten,
            )

        matches = self._deduplicate_matches(raw_matches)[:top_k]
        context, sources = self._build_context(matches)
        history_section = (
            f"Conversation history (for follow-up context):\n{history_text}\n"
            if history_text.strip() else ""
        )

        answer = self._chain.invoke({
            "context": context,
            "question": question,
            "history_section": history_section,
        })

        logger.info("rag_answer_generated", chars=len(answer), sources=len(sources))

        return QueryResponse(
            answer=answer.strip(), question=question, sources=sources,
            model=self._llm.model_name, namespace=namespace,
            rewritten_query=search_query if was_rewritten else None,
            was_rewritten=was_rewritten,
        )

    def search(self, query: str, namespace: str = "default", top_k: int = 5) -> SearchResponse:
        search_query, _ = self._rewriter.rewrite(query)
        query_embedding = self._embedder.embed(search_query)
        matches = self._vector_store.query(query_embedding, namespace=namespace, top_k=top_k)
        deduped = self._deduplicate_matches(matches)
        results = [
            SourceChunk(
                text=m.get("metadata", {}).get("text", "")[:500],
                filename=m.get("metadata", {}).get("filename", ""),
                page=m.get("metadata", {}).get("page", 0),
                score=round(m.get("score", 0.0), 4),
                doc_id=m.get("metadata", {}).get("doc_id", ""),
                chunk_type=m.get("metadata", {}).get("chunk_type", "text"),
            )
            for m in deduped
        ]
        return SearchResponse(query=query, results=results, namespace=namespace)