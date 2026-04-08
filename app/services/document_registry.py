"""
SQLite-backed document registry.
Tracks every indexed document with its metadata — no more zero-vector
queries against Pinecone just to list what's been uploaded.
"""
from datetime import datetime, timezone
from typing import Optional

from app.core.database import get_connection
from app.core.logging import get_logger

logger = get_logger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DocumentRegistry:
    """Reliable document metadata store backed by SQLite."""

    def register(
        self,
        doc_id: str,
        filename: str,
        namespace: str,
        chunks: int,
        pages: int,
        table_count: int = 0,
        file_size: int = 0,
    ) -> None:
        """Insert or replace a document record after successful Pinecone upsert."""
        conn = get_connection()
        conn.execute("""
            INSERT OR REPLACE INTO documents
                (doc_id, filename, namespace, chunks, pages, table_count, file_size, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (doc_id, filename, namespace, chunks, pages, table_count, file_size, _now()))
        conn.commit()
        logger.info("document_registered", doc_id=doc_id, filename=filename,
                    namespace=namespace, chunks=chunks)

    def get(self, doc_id: str) -> Optional[dict]:
        conn = get_connection()
        row = conn.execute(
            "SELECT * FROM documents WHERE doc_id=?", (doc_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_by_namespace(self, namespace: str) -> list[dict]:
        conn = get_connection()
        rows = conn.execute(
            "SELECT * FROM documents WHERE namespace=? ORDER BY indexed_at DESC",
            (namespace,)
        ).fetchall()
        return [dict(r) for r in rows]

    def find_by_filename(self, filename: str, namespace: str) -> list[dict]:
        """Return all records matching this filename in this namespace — used for duplicate detection."""
        conn = get_connection()
        rows = conn.execute(
            "SELECT * FROM documents WHERE filename=? AND namespace=?",
            (filename, namespace)
        ).fetchall()
        return [dict(r) for r in rows]

    def delete(self, doc_id: str) -> bool:
        conn = get_connection()
        affected = conn.execute(
            "DELETE FROM documents WHERE doc_id=?", (doc_id,)
        ).rowcount
        conn.commit()
        return affected > 0

    def delete_by_filename(self, filename: str, namespace: str) -> list[str]:
        """Delete all records for a filename in a namespace. Returns deleted doc_ids."""
        conn = get_connection()
        rows = conn.execute(
            "SELECT doc_id FROM documents WHERE filename=? AND namespace=?",
            (filename, namespace)
        ).fetchall()
        doc_ids = [r["doc_id"] for r in rows]
        if doc_ids:
            conn.execute(
                "DELETE FROM documents WHERE filename=? AND namespace=?",
                (filename, namespace)
            )
            conn.commit()
        return doc_ids

    def count(self, namespace: Optional[str] = None) -> int:
        conn = get_connection()
        if namespace:
            return conn.execute(
                "SELECT COUNT(*) FROM documents WHERE namespace=?", (namespace,)
            ).fetchone()[0]
        return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]


# Singleton
_registry = DocumentRegistry()

def get_document_registry() -> DocumentRegistry:
    return _registry