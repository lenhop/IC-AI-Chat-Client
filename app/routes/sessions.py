"""
Session REST for M3 (only mounted when REDIS_ENABLED=true).

Creates server-side session ids and returns stored messages for refresh recovery.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.config import get_config
from app.deps import require_session_store
from app.memory.session_store import (
    SessionAccessDeniedError,
    SessionNotFoundError,
    SessionStore,
    messages_for_openai_payload,
)

router = APIRouter(tags=["internal_ui"])


class AppendSessionMessageBody(BaseModel):
    """Typed single-message append for agents / debugging (canonical Redis schema)."""

    type: str = Field(..., min_length=1, description="Stored message type, e.g. plan, clarification.")
    content: str = Field(..., min_length=1)
    turn_id: str = Field(..., min_length=1, description="Shared turn UUID for this line.")


@router.post("/api/sessions")
async def create_session(store: SessionStore = Depends(require_session_store)) -> dict:
    """
    Allocate a new Redis-backed session for the configured USER_ID.

    Returns:
        JSON object with session_id (UUID string).
    """
    cfg = get_config()
    session_id = store.create_session(cfg.user_id, cfg.llm_backend)
    return {"session_id": session_id}


@router.get("/api/sessions/{session_id}/messages")
async def list_session_messages(
    session_id: str,
    store: SessionStore = Depends(require_session_store),
) -> dict:
    """
    Return stored messages for hydrate (legacy JS / debugging).

    Raises:
        HTTPException: 404 when session missing, 403 when user_id mismatch.
    """
    cfg = get_config()
    try:
        raw = store.get_messages(session_id, cfg.user_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found") from None
    except SessionAccessDeniedError:
        raise HTTPException(status_code=403, detail="Forbidden") from None
    return {"messages": messages_for_openai_payload(raw)}


@router.post("/api/sessions/{session_id}/messages")
async def append_session_message(
    session_id: str,
    body: AppendSessionMessageBody,
    store: SessionStore = Depends(require_session_store),
) -> dict:
    """
    Append one canonical message row (same validation as :meth:`SessionStore.append_memory_message`).

    Raises:
        HTTPException: 404 / 403 for session errors; 400 for invalid body fields.
    """
    cfg = get_config()
    try:
        store.append_memory_message(
            session_id,
            cfg.user_id,
            message_type=body.type.strip(),
            content=body.content,
            turn_id=body.turn_id.strip(),
        )
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found") from None
    except SessionAccessDeniedError:
        raise HTTPException(status_code=403, detail="Forbidden") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}
