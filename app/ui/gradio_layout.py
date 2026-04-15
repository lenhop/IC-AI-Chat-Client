"""Layout assembly service for Gradio chat UI components and event wiring."""

from __future__ import annotations

import html
import inspect
import logging
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import gradio as gr

from app.ui.chat_history_normalize import normalize_chat_history as normalize_chat_history_rows

logger = logging.getLogger(__name__)

# Inline styles so the title bar stays deep blue even if theme/mount CSS fails to load.
_TITLE_BAR_INLINE_STYLE = (
    "display:block;width:100%;margin:0;padding:14px 20px;box-sizing:border-box;"
    "background:#0d47a1;color:#ffffff;font-size:1.125rem;font-weight:600;"
    "letter-spacing:0.02em;border:none;font-family:system-ui,sans-serif;"
)


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
    def _format_session_id(cls, session_id: Any) -> str:
        """Return stable user-visible session id text."""
        sid = str(session_id or "").strip()
        return sid if sid else "Not available"

    @classmethod
    def _meta_markdown(cls, label: str, value: str) -> str:
        """Build sidebar metadata as Markdown (no Textbox chrome)."""
        text = str(value or "").strip()
        return f"**{label}**\n\n{text}"

    @classmethod
    def _session_markdown(cls, session_id: Any) -> str:
        """Session block: heading + monospace id for readability."""
        sid = cls._format_session_id(session_id)
        if sid == "Not available":
            return cls._meta_markdown("Session ID", "Not available")
        return f"**Session ID**\n\n`{sid}`"

    @classmethod
    def normalize_chat_history(cls, history: Any) -> List[Dict[str, str]]:
        """Normalize any history-like value to safe Chatbot messages rows."""
        return normalize_chat_history_rows(history)

    @classmethod
    def _chatbot_kwargs(cls) -> Dict[str, Any]:
        """Build Chatbot constructor kwargs compatible with Gradio 4.x-6.x."""
        chatbot_kwargs: Dict[str, Any] = {
            "height": 560,
            "layout": "bubble",
            "scale": 1,
            "elem_classes": ["icai-dialog-box"],
        }
        try:
            params = inspect.signature(gr.Chatbot.__init__).parameters
            if "show_label" in params:
                chatbot_kwargs["show_label"] = False
            if "type" in params:
                chatbot_kwargs["type"] = "messages"
        except (TypeError, ValueError) as exc:
            logger.debug("Chatbot signature introspection skipped: %s", exc)
        return chatbot_kwargs

    @classmethod
    def build_blocks(
        cls,
        *,
        page_title: str,
        theme_key: str,
        backend_label: str,
        model_label: str,
        user_label: str,
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
            # Keep the argument wired for compatibility with existing call sites.
            _ = theme_key
            with gr.Column(scale=1, min_width=960, elem_classes=["icai-chat-root"]):
                safe_title = html.escape(page_title)
                gr.HTML(
                    f'<header class="icai-client-title-bar" style="{_TITLE_BAR_INLINE_STYLE}">{safe_title}</header>',
                    elem_classes=["icai-title-container"],
                )
                session_state = gr.State(value=None)
                with gr.Row(elem_classes=["icai-main-row"]):
                    with gr.Column(scale=1, min_width=260, elem_classes=["icai-sidebar"]):
                        gr.Markdown(
                            cls._meta_markdown("Backend", backend_label),
                            elem_classes=["icai-meta-md"],
                        )
                        gr.Markdown(
                            cls._meta_markdown("Model", model_label),
                            elem_classes=["icai-meta-md"],
                        )
                        gr.Markdown(
                            cls._meta_markdown("User", user_label),
                            elem_classes=["icai-meta-md"],
                        )
                        session_md = gr.Markdown(
                            cls._session_markdown(None),
                            elem_classes=["icai-meta-md", "icai-session-md"],
                        )
                    with gr.Column(scale=3, min_width=520, elem_classes=["icai-chat-panel"]):
                        chatbot = gr.Chatbot(**cls._chatbot_kwargs())
                        message_box = gr.Textbox(
                            show_label=False,
                            lines=1,
                            max_lines=3,
                            placeholder="Type a message, then press Send or Enter...",
                            scale=1,
                            elem_classes=["icai-chat-input"],
                        )
                        with gr.Row(elem_classes=["icai-chat-actions"]):
                            submit_btn = gr.Button("Send", scale=1, variant="secondary")
                            clear_btn = gr.Button("Clear conversation", scale=1, variant="secondary")

                def _on_load_wrapped(
                    request: Optional[gr.Request] = None,
                ) -> Tuple[Optional[str], List[Dict[str, str]], str]:
                    sid, history = on_load(request)
                    return sid, cls.normalize_chat_history(history), cls._session_markdown(sid)

                def _on_user_turn_wrapped(
                    message: str,
                    history: Any,
                    session_id: Any,
                    request: Optional[gr.Request] = None,
                ) -> Tuple[str, List[Dict[str, str]], Any, str]:
                    msg, rows, sid = on_user_turn(message, history, session_id, request)
                    return msg, cls.normalize_chat_history(rows), sid, cls._session_markdown(sid)

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
                ) -> Tuple[List[Dict[str, str]], str, Any, str]:
                    rows, msg, sid = on_clear_chat(session_id, request)
                    return cls.normalize_chat_history(rows), msg, sid, cls._session_markdown(sid)

                demo.load(_on_load_wrapped, None, [session_state, chatbot, session_md], queue=False)

                submit_event = message_box.submit(
                    _on_user_turn_wrapped,
                    [message_box, chatbot, session_state],
                    [message_box, chatbot, session_state, session_md],
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
                    [message_box, chatbot, session_state, session_md],
                    queue=False,
                ).then(
                    _on_stream_assistant_wrapped,
                    [chatbot, session_state],
                    [chatbot],
                )

                clear_btn.click(
                    _on_clear_chat_wrapped,
                    [session_state],
                    [chatbot, message_box, session_state, session_md],
                    queue=False,
                )
        return demo
