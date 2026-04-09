"""Layout assembly service for Gradio chat UI components and event wiring."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import gradio as gr

logger = logging.getLogger(__name__)


class GradioLayoutService:
    """Build Gradio blocks with injected callback functions."""

    # Shared callback types keep request-forwarding contracts explicit.
    OnLoadCallback = Callable[[Optional[gr.Request]], Tuple[Optional[str], List[Dict[str, Any]]]]
    OnUserTurnCallback = Callable[
        [str, Any, Any, Optional[gr.Request]],
        Tuple[str, List[Dict[str, Any]], Any],
    ]
    OnStreamAssistantCallback = Callable[[Any, Any, Optional[gr.Request]], Any]
    OnClearChatCallback = Callable[[Any, Optional[gr.Request]], Tuple[List[Dict[str, Any]], str, Any]]

    @classmethod
    def _normalize_chat_row(cls, item: Any) -> Optional[Dict[str, str]]:
        """
        Normalize one chatbot row to Gradio ``messages`` shape.

        This is a last-mile safety guard before values are sent to Gradio's
        Chatbot component, preventing runtime format errors from malformed rows.
        """
        role: Any = None
        content: Any = None
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
        else:
            role = getattr(item, "role", None)
            content = getattr(item, "content", None)
        role_text = str(role or "").strip()
        if role_text not in {"system", "user", "assistant"}:
            return None
        return {"role": role_text, "content": str(content or "")}

    @classmethod
    def normalize_chat_history(cls, history: Any) -> List[Dict[str, str]]:
        """Normalize any history-like value to safe Chatbot messages rows."""
        rows = history if isinstance(history, list) else (history or [])
        out: List[Dict[str, str]] = []
        for row in rows:
            item = cls._normalize_chat_row(row)
            if item is not None:
                out.append(item)
        return out

    @classmethod
    def _chatbot_kwargs(cls) -> Dict[str, Any]:
        """Build Chatbot constructor kwargs compatible with Gradio 4.x-6.x."""
        chatbot_kwargs: Dict[str, Any] = {
            "height": 560,
            "label": "对话",
            "layout": "bubble",
            "scale": 1,
        }
        try:
            if "type" in inspect.signature(gr.Chatbot.__init__).parameters:
                chatbot_kwargs["type"] = "messages"
        except (TypeError, ValueError) as exc:
            logger.debug("Chatbot signature introspection skipped: %s", exc)
        return chatbot_kwargs

    @classmethod
    def build_blocks(
        cls,
        *,
        page_title: str,
        header_html: str,
        theme_key: str,
        on_load: OnLoadCallback,
        on_user_turn: OnUserTurnCallback,
        on_stream_assistant: OnStreamAssistantCallback,
        on_clear_chat: OnClearChatCallback,
    ) -> gr.Blocks:
        """
        Build Gradio blocks and bind event handlers.

        Note:
            Theme and CSS are intentionally not passed here to avoid Gradio 6
            constructor warnings. They are applied during mount.
        """
        with gr.Blocks(
            title=page_title,
            analytics_enabled=False,
            fill_width=True,
        ) as demo:
            with gr.Column(scale=1, min_width=960, elem_classes=["icai-chat-root"]):
                if theme_key == "minimal":
                    gr.Markdown(header_html)
                else:
                    gr.HTML(header_html)

                session_state = gr.State(value=None)
                chatbot = gr.Chatbot(**cls._chatbot_kwargs())
                message_box = gr.Textbox(
                    show_label=False,
                    lines=2,
                    placeholder="输入消息后按发送或 Enter…",
                    scale=1,
                )
                with gr.Row():
                    submit_btn = gr.Button("发送", scale=1)
                    clear_btn = gr.Button("清空会话", scale=1)

                def _on_load_wrapped(
                    request: Optional[gr.Request] = None,
                ) -> Tuple[Optional[str], List[Dict[str, str]]]:
                    sid, history = on_load(request)
                    return sid, cls.normalize_chat_history(history)

                def _on_user_turn_wrapped(
                    message: str,
                    history: Any,
                    session_id: Any,
                    request: Optional[gr.Request] = None,
                ) -> Tuple[str, List[Dict[str, str]], Any]:
                    msg, rows, sid = on_user_turn(message, history, session_id, request)
                    return msg, cls.normalize_chat_history(rows), sid

                def _on_stream_assistant_wrapped(
                    history: Any,
                    session_id: Any,
                    request: Optional[gr.Request] = None,
                ) -> Generator[List[Dict[str, str]], None, None]:
                    for rows in on_stream_assistant(history, session_id, request):
                        yield cls.normalize_chat_history(rows)

                def _on_clear_chat_wrapped(
                    session_id: Any,
                    request: Optional[gr.Request] = None,
                ) -> Tuple[List[Dict[str, str]], str, Any]:
                    rows, msg, sid = on_clear_chat(session_id, request)
                    return cls.normalize_chat_history(rows), msg, sid

                demo.load(_on_load_wrapped, None, [session_state, chatbot], queue=False)

                submit_event = message_box.submit(
                    _on_user_turn_wrapped,
                    [message_box, chatbot, session_state],
                    [message_box, chatbot, session_state],
                    queue=False,
                )
                submit_event.then(
                    _on_stream_assistant_wrapped,
                    [chatbot, session_state],
                    [chatbot],
                )

                submit_btn.click(
                    _on_user_turn_wrapped,
                    [message_box, chatbot, session_state],
                    [message_box, chatbot, session_state],
                    queue=False,
                ).then(
                    _on_stream_assistant_wrapped,
                    [chatbot, session_state],
                    [chatbot],
                )

                clear_btn.click(
                    _on_clear_chat_wrapped,
                    [session_state],
                    [chatbot, message_box, session_state],
                    queue=False,
                )
        return demo
