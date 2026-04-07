"""
In-memory conversation store with optional summarization.
Each session has an ID, a list of turns, and a namespace it belongs to.
"""
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Turn:
    role: str           # "user" | "assistant"
    content: str
    sources: list = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class Session:
    id: str
    title: str
    namespace: str
    turns: list[Turn] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add_turn(self, role: str, content: str, sources: list = None):
        self.turns.append(Turn(role=role, content=content, sources=sources or []))
        self.updated_at = datetime.now(timezone.utc).isoformat()
        # Auto-title from first user message
        if role == "user" and self.title == "New chat":
            self.title = content[:50] + ("…" if len(content) > 50 else "")

    def get_history_text(self, max_turns: int = 6) -> str:
        """Return last N turns formatted for injection into the RAG prompt."""
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
    """Simple in-process session store. Sessions live for the server's lifetime."""

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def create_session(self, namespace: str = "default", title: str = "New chat") -> Session:
        session_id = str(uuid.uuid4())
        session = Session(id=session_id, title=title, namespace=namespace)
        self._sessions[session_id] = session
        logger.info("session_created", session_id=session_id, namespace=namespace)
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: Optional[str], namespace: str) -> Session:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        return self.create_session(namespace=namespace)

    def list_sessions(self) -> list[dict]:
        return sorted(
            [s.to_dict() for s in self._sessions.values()],
            key=lambda s: s["updated_at"],
            reverse=True,
        )

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def clear_session(self, session_id: str) -> bool:
        session = self._sessions.get(session_id)
        if session:
            session.turns = []
            session.title = "New chat"
            return True
        return False


# Singleton
_store = MemoryStore()

def get_memory_store() -> MemoryStore:
    return _store