"""
Starlette session helpers for Gradio multi-message turns (M3 v3.2).

Stores the active ``turn_id`` UUID in the same signed cookie session as
``icai_gradio_session_id`` so clarifications and follow-up user lines share one
Redis ``turn_id`` until the assistant answer is persisted successfully.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

# Starlette session key; must match lifecycle in gradio_chat (clear after answer / new session / clear chat).
ACTIVE_TURN_ID_SESSION_KEY = "icai_active_turn_id"


class GradioSessionTurn:
    """
    Read/write active turn id on the Starlette session dict behind Gradio ``Request``.

    Uses ``@classmethod`` only for a single entry style (project AI dev rules).
    """

    @classmethod
    def get_active_turn_id(cls, session: Any) -> str:
        """
        Return the current active ``turn_id`` string, or empty if missing/invalid.

        Args:
            session: ``request.session`` mapping or ``None``.

        Returns:
            Non-empty UUID string or ``""``.
        """
        if session is None:
            return ""
        raw = session.get(ACTIVE_TURN_ID_SESSION_KEY)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        return ""

    @classmethod
    def ensure_active_turn_id(cls, session: Any) -> str:
        """
        Return existing active ``turn_id`` or allocate a new UUID and store it.

        Args:
            session: Starlette session mapping (mutated when a new id is created).

        Returns:
            ``turn_id`` string.

        Example:
            >>> s = {}
            >>> tid = GradioSessionTurn.ensure_active_turn_id(s)
            >>> assert s[ACTIVE_TURN_ID_SESSION_KEY] == tid
        """
        if session is None:
            return str(uuid.uuid4())
        existing = cls.get_active_turn_id(session)
        if existing:
            return existing
        tid = str(uuid.uuid4())
        session[ACTIVE_TURN_ID_SESSION_KEY] = tid
        return tid

    @classmethod
    def clear_active_turn_id(cls, session: Any) -> None:
        """Remove active turn id so the next user line starts a new turn."""
        if session is None:
            return
        session.pop(ACTIVE_TURN_ID_SESSION_KEY, None)
