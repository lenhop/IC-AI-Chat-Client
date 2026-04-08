"""
Session meta + typed message list in Redis (M3).

New records: user_id, session_id, type, content, timestamp (ISO8601 UTC), optional turn_id.
Legacy records: role, content, ts (unix) — normalized on read.

Redis key helpers (``{prefix}session:{id}:…``) live in this module; naming aligns with
tasks/project_goal.md §2.3.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import redis

from app.config import RedisSettings

logger = logging.getLogger(__name__)


def session_meta_key(prefix: str, session_id: str) -> str:
    """Hash: user_id, created_at, last_active, provider."""
    return f"{prefix}session:{session_id}:meta"


def session_messages_key(prefix: str, session_id: str) -> str:
    """List of JSON message objects (role, content, ts)."""
    return f"{prefix}session:{session_id}:messages"


def session_events_key(prefix: str, session_id: str) -> str:
    """Placeholder list for M4 step events (optional touch on create)."""
    return f"{prefix}session:{session_id}:events"

# Allowed message types for new writes; unknown types still accepted on read.
MEMORY_MESSAGE_TYPES = frozenset(
    {
        "query",
        "answer",
        "clarification",
        "rewriting",
        "classification",
        "reason",
        "plan",
    }
)


class SessionNotFoundError(Exception):
    """No meta key for the given session id."""


class SessionAccessDeniedError(Exception):
    """Meta user_id does not match the requesting user."""


def _iso_utc_now() -> str:
    """UTC timestamp string for new messages."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _iso_from_unix_ts(ts: Any) -> str:
    """Convert legacy unix ``ts`` to ISO-like UTC string."""
    try:
        sec = int(ts)
    except (TypeError, ValueError):
        return ""
    if sec <= 0:
        return ""
    return datetime.fromtimestamp(sec, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def normalize_stored_message(
    obj: Any,
    session_id: str,
    owner_user_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Normalize one Redis JSON object to canonical typed shape.

    Legacy: ``role`` + ``content`` + ``ts`` -> ``query`` / ``answer``.

    Args:
        obj: Parsed JSON dict.
        session_id: Session id for this list.
        owner_user_id: Meta user_id for filling missing user_id.

    Returns:
        Canonical dict or None if unusable.
    """
    if not isinstance(obj, dict):
        return None
    if (obj.get("type") or "").strip():
        return {
            "user_id": str(obj.get("user_id") or owner_user_id).strip(),
            "session_id": str(obj.get("session_id") or session_id).strip(),
            "type": str(obj.get("type") or "").strip(),
            "content": str(obj.get("content") or ""),
            "timestamp": str(obj.get("timestamp") or "").strip() or _iso_from_unix_ts(obj.get("ts")),
            "turn_id": str(obj.get("turn_id") or "").strip(),
        }
    role = (obj.get("role") or "").strip()
    content = str(obj.get("content") or "")
    if role == "user":
        mtype = "query"
    elif role == "assistant":
        mtype = "answer"
    else:
        return None
    return {
        "user_id": str(owner_user_id).strip(),
        "session_id": str(session_id).strip(),
        "type": mtype,
        "content": content,
        "timestamp": _iso_from_unix_ts(obj.get("ts")) or _iso_utc_now(),
        "turn_id": "",
    }


class SessionStore:
    """CRUD for icai:session:{id}:meta and :messages."""

    def __init__(self, client: redis.Redis, settings: RedisSettings) -> None:
        self._r = client
        self._settings = settings
        self._prefix = settings.key_prefix
        self._ttl = settings.session_ttl_seconds

    def create_session(self, user_id: str, provider: str) -> str:
        """
        Allocate a new session id and write meta (+ empty messages/events keys).

        Args:
            user_id: Owner id from AppConfig / gateway.
            provider: LLM backend label (e.g. deepseek, ollama).

        Returns:
            New session UUID string.
        """
        session_id = str(uuid.uuid4())
        now = int(time.time())
        meta_k = session_meta_key(self._prefix, session_id)
        msg_k = session_messages_key(self._prefix, session_id)
        ev_k = session_events_key(self._prefix, session_id)
        pipe = self._r.pipeline(transaction=True)
        pipe.hset(
            meta_k,
            mapping={
                "user_id": user_id,
                "created_at": str(now),
                "last_active": str(now),
                "provider": provider,
            },
        )
        pipe.delete(msg_k)
        pipe.delete(ev_k)
        pipe.execute()
        self._touch_ttl(session_id)
        return session_id

    def get_messages(self, session_id: str, user_id: str) -> List[Dict[str, Any]]:
        """
        Load normalized message list after verifying session ownership.

        Args:
            session_id: Client-held session id.
            user_id: Caller user id (must match meta).

        Returns:
            List of canonical dicts (type, content, timestamp, user_id, session_id, turn_id).

        Raises:
            SessionNotFoundError: Missing session.
            SessionAccessDeniedError: user_id mismatch.
        """
        self._assert_session_user(session_id, user_id)
        msg_k = session_messages_key(self._prefix, session_id)
        raw_list = self._r.lrange(msg_k, 0, -1)
        out: List[Dict[str, Any]] = []
        for raw in raw_list:
            try:
                obj = json.loads(raw)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("skip bad message json in %s: %s", msg_k, exc)
                continue
            norm = normalize_stored_message(obj, session_id, user_id)
            if norm is not None:
                out.append(norm)
        return out

    def clear_messages(self, session_id: str, user_id: str) -> None:
        """
        Remove all stored messages for a session (UI clear button).

        Args:
            session_id: Target session.
            user_id: Must match meta.
        """
        self._assert_session_user(session_id, user_id)
        msg_k = session_messages_key(self._prefix, session_id)
        try:
            self._r.delete(msg_k)
        except redis.RedisError as exc:
            logger.warning("redis delete failed for %s: %s", msg_k, exc)
            raise
        self._touch_ttl(session_id)

    def append_turn(
        self,
        session_id: str,
        user_id: str,
        user_content: str,
        assistant_content: str,
    ) -> None:
        """
        Append one query + one answer with full typed schema; refresh meta last_active.

        Args:
            session_id: Target session.
            user_id: Must match meta.
            user_content: Latest user turn text.
            assistant_content: Full assistant reply for this turn.
        """
        self._assert_session_user(session_id, user_id)
        ts = _iso_utc_now()
        turn_id = str(uuid.uuid4())
        msg_k = session_messages_key(self._prefix, session_id)
        meta_k = session_meta_key(self._prefix, session_id)
        base = {
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": ts,
            "turn_id": turn_id,
        }
        user_blob = json.dumps(
            {**base, "type": "query", "content": user_content},
            ensure_ascii=False,
        )
        asst_blob = json.dumps(
            {**base, "type": "answer", "content": assistant_content},
            ensure_ascii=False,
        )
        now_unix = int(time.time())
        pipe = self._r.pipeline(transaction=True)
        pipe.rpush(msg_k, user_blob, asst_blob)
        pipe.hset(meta_k, "last_active", str(now_unix))
        pipe.execute()
        self._touch_ttl(session_id)

    def _assert_session_user(self, session_id: str, user_id: str) -> None:
        meta_k = session_meta_key(self._prefix, session_id)
        owner = self._r.hget(meta_k, "user_id")
        if owner is None:
            raise SessionNotFoundError(session_id)
        if str(owner) != str(user_id):
            raise SessionAccessDeniedError(session_id)

    def _touch_ttl(self, session_id: str) -> None:
        """Renew TTL on meta, messages, and events keys."""
        ttl = int(self._ttl)
        for key_fn in (session_meta_key, session_messages_key, session_events_key):
            k = key_fn(self._prefix, session_id)
            try:
                self._r.expire(k, ttl)
            except redis.RedisError as exc:
                logger.warning("redis expire failed for %s: %s", k, exc)


def messages_for_openai_payload(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Map normalized or legacy-shaped records to OpenAI-style role/content.

    Args:
        messages: Normalized dicts (preferred) or legacy role-based.

    Returns:
        OpenAI-style message dicts (skips empty content).
    """
    out: List[Dict[str, str]] = []
    for m in messages:
        mtype = (m.get("type") or "").strip()
        content = (m.get("content") or "").strip()
        if mtype == "query":
            role = "user"
        elif mtype == "answer":
            role = "assistant"
        else:
            role = str(m.get("role") or "")
        if not content or role not in ("system", "user", "assistant"):
            continue
        out.append({"role": role, "content": content})
    return out


def gradio_history_from_stored(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build Gradio Chatbot(type='messages') rows from normalized stored dicts.

    query/answer map to user/assistant; other types become assistant lines with type tag.

    Args:
        messages: Normalized dicts from ``get_messages``.

    Returns:
        List of {role, content} for Gradio.
    """
    rows: List[Dict[str, Any]] = []
    for m in messages:
        mtype = (m.get("type") or "").strip()
        content = str(m.get("content") or "")
        if mtype == "query":
            rows.append({"role": "user", "content": content})
        elif mtype == "answer":
            rows.append({"role": "assistant", "content": content})
        elif mtype not in ("query", "answer") and mtype:
            label = f"[{mtype}] {content}".strip()
            rows.append({"role": "assistant", "content": label})
        elif (m.get("role") or "") in ("system", "user", "assistant"):
            rows.append({"role": str(m.get("role")), "content": content})
    return rows
