from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.embedder import OllamaEmbedder
from app.services.vector_store import VectorStore
from app.models.schemas import QueryResponse, SourceChunk, SearchResponse

logger = get_logger(__name__)

RAG_PROMPT = PromptTemplate.from_template("""You are a helpful assistant that answers questions based strictly on the provided context from PDF documents.

Rules:
- Answer only from the context below. Do not use prior knowledge.
- If the context doesn't contain enough information, say "I couldn't find relevant information in the provided documents."
- Cite the source document and page number when possible.
- Be concise and precise.

Context:
{context}

Question: {question}

Answer:""")


class RAGChain:
    def __init__(self):
        settings = get_settings()
        self._embedder = OllamaEmbedder()
        self._vector_store = VectorStore()
        self._llm = OllamaLLM(
            base_url=settings.ollama_base_url,
            model=settings.ollama_llm_model,
            temperature=0.1,
        )
        self._chain = RAG_PROMPT | self._llm | StrOutputParser()

    def ask(
        self,
        question: str,
        namespace: str = "default",
        top_k: int = 5,
        temperature: float = 0.1,
    ) -> QueryResponse:
        logger.info("rag_query", question=question[:80], namespace=namespace, top_k=top_k)

        # 1. Embed the question
        query_embedding = self._embedder.embed(question)

        # 2. Retrieve relevant chunks
        matches = self._vector_store.query(query_embedding, namespace=namespace, top_k=top_k)

        if not matches:
            return QueryResponse(
                answer="No relevant documents found in the specified namespace.",
                question=question,
                sources=[],
                model=self._llm.model,
                namespace=namespace,
            )

        # 3. Build context string and source list
        context_parts = []
        sources = []
        for match in matches:
            meta = match.get("metadata", {})
            text = meta.get("text", "")
            filename = meta.get("filename", "unknown")
            page = meta.get("page", 0)
            score = match.get("score", 0.0)

            context_parts.append(f"[Source: {filename}, Page {page}]\n{text}")
            sources.append(SourceChunk(
                text=text[:500],
                filename=filename,
                page=page,
                score=round(score, 4),
                doc_id=meta.get("doc_id", ""),
            ))

        context = "\n\n---\n\n".join(context_parts)

        # 4. Run LLM chain
        if temperature != 0.1:
            self._llm.temperature = temperature

        answer = self._chain.invoke({"context": context, "question": question})

        logger.info("rag_answer_generated", chars=len(answer), sources=len(sources))

        return QueryResponse(
            answer=answer.strip(),
            question=question,
            sources=sources,
            model=self._llm.model,
            namespace=namespace,
        )

    def search(
        self, query: str, namespace: str = "default", top_k: int = 5
    ) -> SearchResponse:
        """Raw similarity search without LLM generation."""
        query_embedding = self._embedder.embed(query)
        matches = self._vector_store.query(query_embedding, namespace=namespace, top_k=top_k)

        results = [
            SourceChunk(
                text=m.get("metadata", {}).get("text", "")[:500],
                filename=m.get("metadata", {}).get("filename", ""),
                page=m.get("metadata", {}).get("page", 0),
                score=round(m.get("score", 0.0), 4),
                doc_id=m.get("metadata", {}).get("doc_id", ""),
            )
            for m in matches
        ]

        return SearchResponse(query=query, results=results, namespace=namespace)