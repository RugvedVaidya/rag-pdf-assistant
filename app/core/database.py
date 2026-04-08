"""
SQLite database layer.
Handles connection management, schema creation, and migrations.
Database file is stored at data/docwise.db relative to project root.
"""
import sqlite3
import threading
from pathlib import Path
from app.core.logging import get_logger

logger = get_logger(__name__)

# Thread-local storage for connections (sqlite3 connections are not thread-safe)
_local = threading.local()

DB_PATH = Path("data/docwise.db")


def get_connection() -> sqlite3.Connection:
    """Return a thread-local SQLite connection, creating it if needed."""
    if not hasattr(_local, "conn") or _local.conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row          # rows behave like dicts
        conn.execute("PRAGMA journal_mode=WAL")  # better concurrent read performance
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
    return _local.conn


def init_db() -> None:
    """Create all tables if they don't exist. Safe to call on every startup."""
    conn = get_connection()
    conn.executescript("""
        -- ── Sessions ──────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            title       TEXT NOT NULL DEFAULT 'New chat',
            namespace   TEXT NOT NULL DEFAULT 'default',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );

        -- ── Turns ─────────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS turns (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            role        TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content     TEXT NOT NULL,
            sources     TEXT NOT NULL DEFAULT '[]',   -- JSON array
            timestamp   TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);

        -- ── Documents ─────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS documents (
            doc_id      TEXT PRIMARY KEY,
            filename    TEXT NOT NULL,
            namespace   TEXT NOT NULL DEFAULT 'default',
            chunks      INTEGER NOT NULL DEFAULT 0,
            pages       INTEGER NOT NULL DEFAULT 0,
            table_count INTEGER NOT NULL DEFAULT 0,
            file_size   INTEGER NOT NULL DEFAULT 0,
            indexed_at  TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_docs_namespace ON documents(namespace);
        CREATE INDEX IF NOT EXISTS idx_docs_filename  ON documents(filename, namespace);
    """)
    conn.commit()
    logger.info("db_initialized", path=str(DB_PATH))