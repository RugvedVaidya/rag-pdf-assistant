from fastapi import APIRouter, HTTPException
from app.services.memory import get_memory_store
from app.models.schemas import (
    SessionCreate, SessionInfo, SessionDetail, TurnInfo, DeleteSessionResponse
)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("", response_model=SessionInfo)
def create_session(body: SessionCreate):
    store = get_memory_store()
    session = store.create_session(namespace=body.namespace, title=body.title)
    return SessionInfo(**session.to_dict())


@router.get("", response_model=list[SessionInfo])
def list_sessions():
    store = get_memory_store()
    return [SessionInfo(**s) for s in store.list_sessions()]


@router.get("/{session_id}", response_model=SessionDetail)
def get_session(session_id: str):
    store = get_memory_store()
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionDetail(
        id=session.id,
        title=session.title,
        namespace=session.namespace,
        turns=[TurnInfo(role=t.role, content=t.content,
                        sources=t.sources, timestamp=t.timestamp)
               for t in session.turns],
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@router.delete("/{session_id}", response_model=DeleteSessionResponse)
def delete_session(session_id: str):
    store = get_memory_store()
    if not store.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return DeleteSessionResponse(session_id=session_id, message="Session deleted.")


@router.post("/{session_id}/clear", response_model=SessionInfo)
def clear_session(session_id: str):
    store = get_memory_store()
    if not store.clear_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    session = store.get_session(session_id)
    return SessionInfo(**session.to_dict())