"""
Persistence services for Gradio chat runtime.

This module centralizes Redis/session read-write logic so UI layout and stream
handlers do not touch storage details directly.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from app.config import MessageDisplayOptions
from app.memory.redis_runtime import get_redis_for_gradio
from app.memory.session_store import (
    SessionAccessDeniedError,
    SessionNotFoundError,
    SessionStore,
    gradio_history_from_stored,
)
from app.services.prompt_render import (
    format_messages_markdown_for_prompt,
    render_chat_prompt,
    select_rounds_for_prompt,
    select_rounds_for_ui,
)
from app.services.llm_transport import validate_or_normalize_messages
from app.ui.gradio_session_turn import GradioSessionTurn

logger = logging.getLogger(__name__)

GRADIO_SESSION_KEY = "icai_gradio_session_id"
STAGE_MESSAGE_TYPES = frozenset(
    {
        "clarification",
        "rewriting",
        "classification",
        "reason",
        "plan",
        "context",
        "dispatcher",
    }
)


class GradioPersistenceService:
    """Session + Redis persistence service methods for Gradio callbacks."""

    @classmethod
    def session_from_gradio_request(cls, request: Optional[gr.Request]) -> Any:
        """Resolve the Starlette session mapping from a Gradio request wrapper."""
        inner = getattr(request, "request", None)
        if inner is None:
            return None
        return getattr(inner, "session", None)

    @classmethod
    def hydrate_or_create_session(
        cls,
        request: Optional[gr.Request],
        *,
        user_id: str,
        llm_backend: str,
        memory_rounds: int,
        display_options: MessageDisplayOptions,
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Create or resume Redis-backed session and return hydrated Gradio history.

        Returns:
            Tuple of ``(session_id, gradio_history)``. When Redis is disabled,
            returns ``(None, [])``.
        """
        client, redis_settings = get_redis_for_gradio()
        if redis_settings is None or not redis_settings.enabled or client is None:
            return None, []

        store = SessionStore(client, redis_settings)
        session_map = cls.session_from_gradio_request(request)
        sid: Optional[str] = None
        if session_map is not None:
            raw_sid = session_map.get(GRADIO_SESSION_KEY)
            if isinstance(raw_sid, str) and raw_sid.strip():
                sid = raw_sid.strip()

        if sid:
            try:
                stored = store.get_messages(sid, user_id)
                selected = (
                    select_rounds_for_ui(stored, memory_rounds)
                    if memory_rounds > 0
                    else stored
                )
                return sid, gradio_history_from_stored(selected, display_options)
            except (SessionNotFoundError, SessionAccessDeniedError):
                if session_map is not None:
                    session_map.pop(GRADIO_SESSION_KEY, None)

        new_sid = store.create_session(user_id, llm_backend)
        if session_map is not None:
            session_map[GRADIO_SESSION_KEY] = new_sid
            GradioSessionTurn.clear_active_turn_id(session_map)
        return new_sid, []

    @classmethod
    def persist_query(
        cls,
        session_id: Optional[str],
        request: Optional[gr.Request],
        *,
        user_id: str,
        content: str,
    ) -> None:
        """Persist one non-empty query row and keep/create active turn_id."""
        sid = (session_id or "").strip()
        text = (content or "").strip()
        if not sid or not text:
            return

        client, redis_settings = get_redis_for_gradio()
        if redis_settings is None or not redis_settings.enabled or client is None:
            return

        session_map = cls.session_from_gradio_request(request)
        try:
            turn_id = GradioSessionTurn.ensure_active_turn_id(session_map)
            SessionStore(client, redis_settings).append_memory_message(
                sid,
                user_id,
                message_type="query",
                content=text,
                turn_id=turn_id,
            )
        except SessionNotFoundError:
            logger.warning("gradio redis persist query: session not found %s", sid)
        except SessionAccessDeniedError:
            logger.warning("gradio redis persist query: access denied %s", sid)
        except ValueError as exc:
            logger.warning("gradio redis persist query: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("gradio redis persist query failed: %s", exc)

    @classmethod
    def persist_stage_message(
        cls,
        session_id: Optional[str],
        request: Optional[gr.Request],
        *,
        user_id: str,
        message_type: str,
        content: str,
    ) -> None:
        """Persist one typed stage message for the currently active turn."""
        sid = (session_id or "").strip()
        if not sid:
            return
        stage_type = (message_type or "").strip()
        if stage_type not in STAGE_MESSAGE_TYPES:
            return
        text = (content or "").strip()
        if not text:
            return

        client, redis_settings = get_redis_for_gradio()
        if redis_settings is None or not redis_settings.enabled or client is None:
            return

        session_map = cls.session_from_gradio_request(request)
        turn_id = GradioSessionTurn.get_active_turn_id(session_map)
        if not turn_id:
            return

        try:
            SessionStore(client, redis_settings).append_memory_message(
                sid,
                user_id,
                message_type=stage_type,
                content=text,
                turn_id=turn_id,
            )
        except SessionNotFoundError:
            logger.warning("gradio redis persist stage: session not found %s", sid)
        except SessionAccessDeniedError:
            logger.warning("gradio redis persist stage: access denied %s", sid)
        except ValueError as exc:
            logger.warning("gradio redis persist stage: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("gradio redis persist stage failed: %s", exc)

    @classmethod
    def persist_answer_and_finish_turn(
        cls,
        session_id: Optional[str],
        request: Optional[gr.Request],
        *,
        user_id: str,
        assistant_text: str,
    ) -> None:
        """Persist one answer and clear active turn_id only on success."""
        sid = (session_id or "").strip()
        text = (assistant_text or "").strip()
        if not sid or not text:
            return

        client, redis_settings = get_redis_for_gradio()
        if redis_settings is None or not redis_settings.enabled or client is None:
            return

        session_map = cls.session_from_gradio_request(request)
        turn_id = GradioSessionTurn.get_active_turn_id(session_map)
        if not turn_id:
            return

        try:
            SessionStore(client, redis_settings).append_memory_message(
                sid,
                user_id,
                message_type="answer",
                content=text,
                turn_id=turn_id,
            )
            GradioSessionTurn.clear_active_turn_id(session_map)
        except SessionNotFoundError:
            logger.warning("gradio redis persist answer: session not found %s", sid)
        except SessionAccessDeniedError:
            logger.warning("gradio redis persist answer: access denied %s", sid)
        except ValueError as exc:
            logger.warning("gradio redis persist answer: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("gradio redis persist answer failed: %s", exc)

    @classmethod
    def clear_session_messages(
        cls,
        session_id: Optional[str],
        request: Optional[gr.Request],
        *,
        user_id: str,
    ) -> None:
        """Clear Redis message list for a session and reset active turn_id."""
        session_map = cls.session_from_gradio_request(request)
        GradioSessionTurn.clear_active_turn_id(session_map)

        sid = (session_id or "").strip()
        if not sid:
            return

        client, redis_settings = get_redis_for_gradio()
        if redis_settings is None or not redis_settings.enabled or client is None:
            return
        try:
            SessionStore(client, redis_settings).clear_messages(sid, user_id)
        except (SessionNotFoundError, SessionAccessDeniedError) as exc:
            logger.warning("gradio clear session: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("gradio clear redis: %s", exc)

    @classmethod
    def build_prompt_template_messages(
        cls,
        session_id: Optional[str],
        *,
        user_id: str,
        memory_rounds: int,
        user_text: str,
    ) -> Optional[List[Dict[str, str]]]:
        """
        Build prompt-template single-message payload from stored history.

        Returns:
            Normalized single-user message payload, or ``None`` when template mode
            cannot be applied safely.
        """
        sid = (session_id or "").strip()
        query = (user_text or "").strip()
        if not sid or not query:
            return None

        client, redis_settings = get_redis_for_gradio()
        if redis_settings is None or not redis_settings.enabled or client is None:
            return None

        try:
            store = SessionStore(client, redis_settings)
            stored = store.get_messages(sid, user_id)
            selected = select_rounds_for_prompt(stored, memory_rounds)
            markdown = format_messages_markdown_for_prompt(selected)
            prompt = render_chat_prompt(
                current_query=query,
                historical_message=markdown,
            )
            return validate_or_normalize_messages([{"role": "user", "content": prompt}])
        except Exception as exc:  # noqa: BLE001
            logger.warning("gradio prompt_template fallback: %s", exc)
            return None
