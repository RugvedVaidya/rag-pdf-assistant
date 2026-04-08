"""
SQLite-backed conversation store.
Replaces the previous in-process MemoryStore — sessions and turns now
survive server restarts.
"""
import uuid
import json
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional

from app.core.database import get_connection
from app.core.logging import get_logger

logger = get_logger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Turn:
    role: str
    content: str
    sources: list = field(default_factory=list)
    timestamp: str = field(default_factory=_now)


@dataclass
class Session:
    id: str
    title: str
    namespace: str
    turns: list[Turn] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def add_turn(self, role: str, content: str, sources: list = None):
        turn = Turn(role=role, content=content, sources=sources or [])
        self.turns.append(turn)
        self.updated_at = _now()
        if role == "user" and self.title == "New chat":
            self.title = content[:50] + ("..." if len(content) > 50 else "")
        return turn

    def get_history_text(self, max_turns: int = 6) -> str:
        recent = self.turns[-(max_turns * 2):]
        lines = []
        for t in recent:
            prefix = "User" if t.role == "user" else "Assistant"
            lines.append(f"{prefix}: {t.content}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "namespace": self.namespace,
            "turn_count": len(self.turns),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class MemoryStore:
    """SQLite-backed session store. All data persists across server restarts."""

    def create_session(self, namespace: str = "default", title: str = "New chat") -> Session:
        session_id = str(uuid.uuid4())
        now = _now()
        conn = get_connection()
        conn.execute(
            "INSERT INTO sessions (id, title, namespace, created_at, updated_at) VALUES (?,?,?,?,?)",
            (session_id, title, namespace, now, now),
        )
        conn.commit()
        logger.info("session_created", session_id=session_id, namespace=namespace)
        return Session(id=session_id, title=title, namespace=namespace,
                       created_at=now, updated_at=now)

    def get_session(self, session_id: str) -> Optional[Session]:
        conn = get_connection()
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not row:
            return None
        session = Session(
            id=row["id"], title=row["title"], namespace=row["namespace"],
            created_at=row["created_at"], updated_at=row["updated_at"],
        )
        session.turns = self._load_turns(session_id)
        return session

    def get_or_create(self, session_id: Optional[str], namespace: str) -> Session:
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        return self.create_session(namespace=namespace)

    def list_sessions(self) -> list[dict]:
        conn = get_connection()
        rows = conn.execute("""
            SELECT s.*, COUNT(t.id) as turn_count
            FROM sessions s
            LEFT JOIN turns t ON t.session_id = s.id
            GROUP BY s.id
            ORDER BY s.updated_at DESC
        """).fetchall()
        return [dict(r) for r in rows]

    def save_turn(self, session_id: str, role: str, content: str, sources: list) -> None:
        """Persist a single turn and update the session title + updated_at."""
        conn = get_connection()
        now = _now()
        conn.execute(
            "INSERT INTO turns (session_id, role, content, sources, timestamp) VALUES (?,?,?,?,?)",
            (session_id, role, content, json.dumps(sources), now),
        )
        conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now, session_id))
        if role == "user":
            title_row = conn.execute(
                "SELECT title FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
            if title_row and title_row["title"] == "New chat":
                new_title = content[:50] + ("..." if len(content) > 50 else "")
                conn.execute("UPDATE sessions SET title=? WHERE id=?", (new_title, session_id))
        conn.commit()

    def delete_session(self, session_id: str) -> bool:
        conn = get_connection()
        affected = conn.execute("DELETE FROM sessions WHERE id=?", (session_id,)).rowcount
        conn.commit()
        return affected > 0

    def clear_session(self, session_id: str) -> bool:
        conn = get_connection()
        conn.execute("DELETE FROM turns WHERE session_id=?", (session_id,))
        conn.execute(
            "UPDATE sessions SET title='New chat', updated_at=? WHERE id=?",
            (_now(), session_id),
        )
        conn.commit()
        return True

    def _load_turns(self, session_id: str) -> list[Turn]:
        conn = get_connection()
        rows = conn.execute(
            "SELECT * FROM turns WHERE session_id=? ORDER BY id ASC", (session_id,)
        ).fetchall()
        turns = []
        for row in rows:
            try:
                sources = json.loads(row["sources"])
            except Exception:
                sources = []
            turns.append(Turn(
                role=row["role"], content=row["content"],
                sources=sources, timestamp=row["timestamp"],
            ))
        return turns


_store = MemoryStore()

def get_memory_store() -> MemoryStore:
    return _store