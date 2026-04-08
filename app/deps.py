"""
FastAPI dependencies for Redis session store (M3).

Routers that require Redis are only mounted when REDIS_ENABLED=true.
"""

from __future__ import annotations

from fastapi import HTTPException, Request

from app.memory.session_store import SessionStore


def require_session_store(request: Request) -> SessionStore:
    """
    Build SessionStore from app.state (lifespan must have set redis + settings).

    Raises:
        HTTPException: 503 when Redis client is missing or disabled.
    """
    rs = getattr(request.app.state, "redis_settings", None)
    client = getattr(request.app.state, "redis", None)
    if rs is None or not rs.enabled or client is None:
        raise HTTPException(
            status_code=503,
            detail="Redis session store is disabled or not initialized.",
        )
    return SessionStore(client, rs)
