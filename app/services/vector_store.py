from datetime import datetime, timezone
from pinecone import Pinecone, ServerlessSpec

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.pdf_processor import TextChunk

logger = get_logger(__name__)

# nomic-embed-text produces 768-dimensional vectors
EMBEDDING_DIM = 768


class VectorStore:
    """Manages Pinecone index operations: upsert, query, delete."""

    def __init__(self):
        settings = get_settings()
        self._pc = Pinecone(api_key=settings.pinecone_api_key)
        self._index_name = settings.pinecone_index_name
        self._index = self._get_or_create_index(settings.pinecone_environment)

    def _get_or_create_index(self, environment: str):
        existing = [i.name for i in self._pc.list_indexes()]
        if self._index_name not in existing:
            logger.info("creating_pinecone_index", index=self._index_name)
            self._pc.create_index(
                name=self._index_name,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=environment),
            )
        return self._pc.Index(self._index_name)

    def upsert_chunks(
        self,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
        namespace: str = "default",
    ) -> int:
        """Upsert chunk embeddings with metadata into Pinecone."""
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vector_id = f"{chunk.doc_id}_{chunk.chunk_index}"
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "text": chunk.text,
                    "page": chunk.page,
                    "filename": chunk.filename,
                    "doc_id": chunk.doc_id,
                    "chunk_index": chunk.chunk_index,
                    "indexed_at": datetime.now(timezone.utc).isoformat(),
                },
            })

        # Pinecone recommends batches of 100
        batch_size = 100
        upserted = 0
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self._index.upsert(vectors=batch, namespace=namespace)
            upserted += len(batch)
            logger.info("upserted_batch", count=upserted, total=len(vectors))

        return upserted

    def query(
        self,
        embedding: list[float],
        namespace: str = "default",
        top_k: int = 5,
    ) -> list[dict]:
        """Query the index and return top-k results with metadata."""
        result = self._index.query(
            vector=embedding,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
        )
        return result.get("matches", [])

    def delete_document(self, doc_id: str, namespace: str = "default") -> None:
        """Delete all vectors belonging to a document."""
        # Fetch all vector IDs for this doc_id via metadata filter
        results = self._index.query(
            vector=[0.0] * EMBEDDING_DIM,
            top_k=10000,
            namespace=namespace,
            filter={"doc_id": {"$eq": doc_id}},
            include_metadata=False,
        )
        ids = [m["id"] for m in results.get("matches", [])]
        if ids:
            self._index.delete(ids=ids, namespace=namespace)
            logger.info("deleted_vectors", doc_id=doc_id, count=len(ids))

    def list_documents(self, namespace: str = "default") -> list[dict]:
        """Return unique documents indexed in a namespace."""
        stats = self._index.describe_index_stats()
        ns_stats = stats.get("namespaces", {}).get(namespace, {})
        total_vectors = ns_stats.get("vector_count", 0)

        if total_vectors == 0:
            return []

        # Sample to find unique doc_ids (Pinecone doesn't natively list metadata)
        result = self._index.query(
            vector=[0.0] * EMBEDDING_DIM,
            top_k=min(total_vectors, 10000),
            namespace=namespace,
            include_metadata=True,
        )

        seen = {}
        for match in result.get("matches", []):
            meta = match.get("metadata", {})
            doc_id = meta.get("doc_id")
            if doc_id and doc_id not in seen:
                seen[doc_id] = {
                    "doc_id": doc_id,
                    "filename": meta.get("filename", "unknown"),
                    "namespace": namespace,
                    "indexed_at": meta.get("indexed_at"),
                }

        return list(seen.values())

    def check_health(self) -> bool:
        try:
            self._index.describe_index_stats()
            return True
        except Exception:
            return False