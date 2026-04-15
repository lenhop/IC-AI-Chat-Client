"""Handler services for Gradio chat actions and streaming callbacks."""

from __future__ import annotations

import logging
from typing import Any, Dict, Generator, List, Optional, Tuple

import gradio as gr

from app.runtime_config import RuntimeConfig
from app.services.llm_transport import iter_chat_text_deltas, validate_or_normalize_messages
from app.ui.chat_history_normalize import normalize_chat_history
from app.ui.gradio_persistence import GradioPersistenceService

logger = logging.getLogger(__name__)


class GradioHandlerService:
    """User/assistant handler methods for Gradio event pipeline."""

    @classmethod
    def clone_message_history(cls, history: Any) -> List[Dict[str, str]]:
        """Normalize and deep-copy chat history rows as mutable dict list."""
        return list(normalize_chat_history(history))

    @classmethod
    def messages_for_api(cls, history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Build OpenAI-style API payload from visible chat history."""
        trimmed = history
        if (
            trimmed
            and trimmed[-1].get("role") == "assistant"
            and not str(trimmed[-1].get("content") or "").strip()
        ):
            trimmed = trimmed[:-1]

        payload: List[Dict[str, str]] = []
        for row in trimmed:
            role = row.get("role")
            content = str(row.get("content") or "").strip()
            if role in {"system", "user", "assistant"} and content:
                payload.append({"role": str(role), "content": content})
        return payload

    @classmethod
    def handle_user_turn(
        cls,
        message: str,
        history: Any,
        session_id: Any,
        request: Optional[gr.Request] = None,
        *,
        user_id: str,
    ) -> Tuple[str, List[Dict[str, Any]], Any]:
        """Append user message to UI history and persist query when Redis is enabled."""
        history_list = cls.clone_message_history(history)
        text = (message or "").strip()
        if not text:
            return "", history_list, session_id

        GradioPersistenceService.persist_query(
            str(session_id or ""),
            request,
            user_id=user_id,
            content=text,
        )
        history_list.append({"role": "user", "content": text})
        return "", history_list, session_id

    @classmethod
    def stream_assistant(
        cls,
        history: Any,
        session_id: Any,
        request: Optional[gr.Request] = None,
        *,
        user_id: str,
        chat_mode: str,
        memory_rounds: int,
        runtime: Optional[RuntimeConfig] = None,
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """Stream assistant text deltas into the chat history with robust error handling."""
        history_list = cls.clone_message_history(history)
        if not history_list or history_list[-1].get("role") != "user":
            yield cls.clone_message_history(history_list)
            return

        history_list.append({"role": "assistant", "content": ""})
        yield cls.clone_message_history(history_list)

        api_messages = cls.messages_for_api(history_list)
        try:
            normalized = validate_or_normalize_messages(api_messages)
        except ValueError as exc:
            history_list[-1]["content"] = f"[Error] {exc}"
            yield cls.clone_message_history(history_list)
            return

        use_prompt_template = runtime is None and chat_mode == "prompt_template" and session_id
        if use_prompt_template and len(history_list) >= 2:
            user_text = str(history_list[-2].get("content") or "").strip()
            prompt_messages = GradioPersistenceService.build_prompt_template_messages(
                str(session_id or ""),
                user_id=user_id,
                memory_rounds=memory_rounds,
                user_text=user_text,
            )
            if prompt_messages is not None:
                normalized = prompt_messages

        stream_kwargs: Dict[str, Any] = {"messages": normalized}
        if runtime is not None:
            stream_kwargs["runtime"] = runtime
        stream_kwargs["on_stage_message"] = (
            lambda message_type, content: GradioPersistenceService.persist_stage_message(
                str(session_id or ""),
                request,
                user_id=user_id,
                message_type=message_type,
                content=content,
            )
        )

        accumulated = ""
        try:
            for delta in iter_chat_text_deltas(**stream_kwargs):
                accumulated += delta
                history_list[-1]["content"] = accumulated
                yield cls.clone_message_history(history_list)

            GradioPersistenceService.persist_answer_and_finish_turn(
                str(session_id or ""),
                request,
                user_id=user_id,
                assistant_text=accumulated,
            )
        except (RuntimeError, ValueError) as exc:
            logger.warning("Gradio stream_chat failed: %s", exc)
            history_list[-1]["content"] = (
                accumulated + f"\n[Error] {exc}" if accumulated else f"[Error] {exc}"
            )
            yield cls.clone_message_history(history_list)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Gradio chat unexpected failure")
            history_list[-1]["content"] = (
                accumulated + f"\n[Error] {exc}" if accumulated else f"[Error] {exc}"
            )
            yield cls.clone_message_history(history_list)

    @classmethod
    def handle_clear_chat(
        cls,
        session_id: Any,
        request: Optional[gr.Request] = None,
        *,
        user_id: str,
    ) -> Tuple[List[Dict[str, Any]], str, Any]:
        """Clear chat UI and truncate session messages in Redis when available."""
        GradioPersistenceService.clear_session_messages(
            str(session_id or ""),
            request,
            user_id=user_id,
        )
        return [], "", session_id
