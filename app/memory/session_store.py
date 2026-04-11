"""
Session meta + typed message list in Redis (M3).

Each list element is JSON with required fields: ``user_id``, ``session_id``, ``type``,
``content``, ``timestamp``, ``turn_id``. Historical non-canonical ``role``/``ts`` rows are **not** read
(M3 v3.1 option B); migrate Redis data or clear old lists. Key helpers live here;
naming aligns with tasks/project_goal.md §2.3.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import redis

from app.config import MessageDisplayOptions, RedisSettings
from app.ui.message_model import GradioMessageFormatter

logger = logging.getLogger(__name__)


def session_meta_key(prefix: str, session_id: str) -> str:
    """Hash: user_id, created_at, last_active, provider."""
    return f"{prefix}session:{session_id}:meta"


def session_messages_key(prefix: str, session_id: str) -> str:
    """List of JSON message objects (canonical typed schema)."""
    return f"{prefix}session:{session_id}:messages"


def session_events_key(prefix: str, session_id: str) -> str:
    """Placeholder list for M4 step events (optional touch on create)."""
    return f"{prefix}session:{session_id}:events"


# Advisory set for writers; unknown types are still accepted and stored.
MEMORY_MESSAGE_TYPES = frozenset(
    {
        "query",
        "answer",
        "clarification",
        "rewriting",
        "classification",
        "reason",
        "plan",
        "context",
        "dispatcher",
    }
)


class SessionNotFoundError(Exception):
    """No meta key for the given session id."""


class SessionAccessDeniedError(Exception):
    """Meta user_id does not match the requesting user."""


def _iso_utc_now() -> str:
    """UTC timestamp string for new messages."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _warn_if_unknown_message_type(message_type: str) -> None:
    """Log when type is outside the advisory ``MEMORY_MESSAGE_TYPES`` set."""
    if message_type not in MEMORY_MESSAGE_TYPES:
        logger.warning(
            "message type %r is not in MEMORY_MESSAGE_TYPES; storing anyway",
            message_type,
        )


def _canonical_json_blob(
    *,
    user_id: str,
    session_id: str,
    turn_id: str,
    timestamp: str,
    message_type: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Serialize one canonical message for Redis RPUSH."""
    safe_metadata = metadata if isinstance(metadata, dict) else {}
    return json.dumps(
        {
            "user_id": user_id,
            "session_id": session_id,
            "turn_id": turn_id,
            "timestamp": timestamp,
            "type": message_type,
            "content": content,
            "metadata": safe_metadata,
        },
        ensure_ascii=False,
    )


def normalize_stored_message(
    obj: Any,
    session_id: str,
    owner_user_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Normalize one Redis JSON object to canonical typed shape (strict).

    Requires non-empty ``type`` and ``timestamp``. Does not support historical ``role``/``ts`` rows.

    Args:
        obj: Parsed JSON value (must be a dict).
        session_id: Session id for this list.
        owner_user_id: Meta user_id used when ``user_id`` is missing on the object.

    Returns:
        Canonical dict or ``None`` if invalid or historical non-canonical payload.

    Example:
        >>> normalize_stored_message(
        ...     {"type": "query", "content": "hi", "timestamp": "2020-01-01 00:00:00 UTC"},
        ...     "sid",
        ...     "u1",
        ... )
        {'user_id': 'u1', 'session_id': 'sid', 'type': 'query', 'content': 'hi', ...}
    """
    if not isinstance(obj, dict):
        return None
    mtype = str(obj.get("type") or "").strip()
    if not mtype:
        logger.debug("skip stored message without type (session_id=%s)", session_id)
        return None
    ts = str(obj.get("timestamp") or "").strip()
    if not ts:
        logger.warning(
            "skip stored message missing timestamp (type=%r session_id=%s)",
            mtype,
            session_id,
        )
        return None
    return {
        "user_id": str(obj.get("user_id") or owner_user_id).strip(),
        "session_id": str(obj.get("session_id") or session_id).strip(),
        "type": mtype,
        "content": str(obj.get("content") or ""),
        "timestamp": ts,
        "turn_id": str(obj.get("turn_id") or "").strip(),
        "metadata": obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {},
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

    def ensure_session_exists(self, session_id: str, user_id: str, provider: str) -> None:
        """
        Ensure a known session id has meta keys for ingress writes.

        Args:
            session_id: Session id carried by external envelope.
            user_id: Owner id used for access check.
            provider: Backend/provider label for meta.

        Raises:
            ValueError: If ``session_id`` is blank.
            SessionAccessDeniedError: Existing session belongs to another user.
        """
        sid = (session_id or "").strip()
        if not sid:
            raise ValueError("session_id must be non-empty")
        meta_k = session_meta_key(self._prefix, sid)
        owner = self._r.hget(meta_k, "user_id")
        if owner is not None:
            if str(owner) != str(user_id):
                raise SessionAccessDeniedError(sid)
            self._touch_ttl(sid)
            return

        now = int(time.time())
        msg_k = session_messages_key(self._prefix, sid)
        ev_k = session_events_key(self._prefix, sid)
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
        self._touch_ttl(sid)

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

    def _rpush_canonical_blobs(
        self,
        session_id: str,
        user_id: str,
        blobs: List[str],
    ) -> None:
        """RPUSH one or more JSON strings, refresh meta ``last_active``, renew TTL."""
        self._assert_session_user(session_id, user_id)
        msg_k = session_messages_key(self._prefix, session_id)
        meta_k = session_meta_key(self._prefix, session_id)
        now_unix = int(time.time())
        pipe = self._r.pipeline(transaction=True)
        for blob in blobs:
            pipe.rpush(msg_k, blob)
        pipe.hset(meta_k, "last_active", str(now_unix))
        pipe.execute()
        self._touch_ttl(session_id)

    def append_turn(
        self,
        session_id: str,
        user_id: str,
        user_content: str,
        assistant_content: str,
        *,
        turn_id: Optional[str] = None,
    ) -> None:
        """
        Append one query + one answer with full typed schema; refresh meta last_active.

        Args:
            session_id: Target session.
            user_id: Must match meta.
            user_content: Latest user turn text.
            assistant_content: Full assistant reply for this turn.
            turn_id: When set, both messages share this id; otherwise a new UUID is used.
        """
        ts = _iso_utc_now()
        tid_in = (turn_id or "").strip()
        turn_uuid = tid_in if tid_in else str(uuid.uuid4())
        blobs = [
            _canonical_json_blob(
                user_id=user_id,
                session_id=session_id,
                turn_id=turn_uuid,
                timestamp=ts,
                message_type="query",
                content=user_content,
                metadata={},
            ),
            _canonical_json_blob(
                user_id=user_id,
                session_id=session_id,
                turn_id=turn_uuid,
                timestamp=ts,
                message_type="answer",
                content=assistant_content,
                metadata={},
            ),
        ]
        self._rpush_canonical_blobs(session_id, user_id, blobs)

    def append_memory_message(
        self,
        session_id: str,
        user_id: str,
        *,
        message_type: str,
        content: str,
        turn_id: str = "",
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Append a single canonical message (any ``type``).

        Args:
            session_id: Target session.
            user_id: Must match meta.
            message_type: Stored ``type`` field (non-empty).
            content: Message body.
            turn_id: Optional shared turn UUID; empty string if none.
            timestamp: ISO UTC string; defaults to now when missing or blank.
            metadata: Optional structured metadata used by UI message templates.

        Raises:
            ValueError: If ``message_type`` is blank.
            SessionNotFoundError: Missing session meta.
            SessionAccessDeniedError: user_id mismatch.

        Example:
            >>> store.append_memory_message(sid, "u1", message_type="plan", content="step a")
        """
        mt = (message_type or "").strip()
        if not mt:
            raise ValueError("message_type must be non-empty")
        _warn_if_unknown_message_type(mt)
        ts = (timestamp or "").strip() or _iso_utc_now()
        tid = (turn_id or "").strip()
        blob = _canonical_json_blob(
            user_id=user_id,
            session_id=session_id,
            turn_id=tid,
            timestamp=ts,
            message_type=mt,
            content=content,
            metadata=metadata,
        )
        self._rpush_canonical_blobs(session_id, user_id, [blob])

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


def gradio_history_from_stored(
    messages: List[Dict[str, Any]],
    display: Optional[MessageDisplayOptions] = None,
) -> List[Dict[str, Any]]:
    """
    Build Gradio Chatbot(type='messages') rows from normalized stored dicts.

    Args:
        messages: Normalized dicts from ``get_messages``.
        display: Optional toggles for non-query/answer types; default shows all.

    Returns:
        List of ``{role, content}`` for Gradio.

    Example:
        >>> gradio_history_from_stored(
        ...     [{"type": "query", "content": "x", "timestamp": "t", ...}],
        ...     MessageDisplayOptions.all_enabled(),
        ... )
        [{'role': 'user', 'content': 'x'}]
    """
    opt = display if display is not None else MessageDisplayOptions.all_enabled()
    rows: List[Dict[str, Any]] = []
    for m in messages:
        mtype = (m.get("type") or "").strip()
        if not GradioMessageFormatter.should_display_type(mtype, opt):
            continue
        row = GradioMessageFormatter.to_chat_row(m)
        if row is not None:
            # Add a blank placeholder line to previous same-role row so adjacent
            # cards are visually separated while preserving next heading parsing.
            if rows and rows[-1].get("role") == row.get("role"):
                rows[-1] = {**rows[-1], "content": f"{rows[-1].get('content', '')}\n\n&nbsp;\n"}
            rows.append(row)
    return rows
