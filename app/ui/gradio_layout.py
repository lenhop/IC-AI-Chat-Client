"""Layout assembly service for Gradio chat UI components and event wiring."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import gradio as gr

logger = logging.getLogger(__name__)


class GradioLayoutService:
    """Build Gradio blocks with injected callback functions."""

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
        on_load: Callable[..., Tuple[Optional[str], List[Dict[str, Any]]]],
        on_user_turn: Callable[..., Tuple[str, List[Dict[str, Any]], Any]],
        on_stream_assistant: Callable[..., Any],
        on_clear_chat: Callable[..., Tuple[List[Dict[str, Any]], str, Any]],
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

                demo.load(on_load, None, [session_state, chatbot], queue=False)

                submit_event = message_box.submit(
                    on_user_turn,
                    [message_box, chatbot, session_state],
                    [message_box, chatbot, session_state],
                    queue=False,
                )
                submit_event.then(
                    on_stream_assistant,
                    [chatbot, session_state],
                    [chatbot],
                )

                submit_btn.click(
                    on_user_turn,
                    [message_box, chatbot, session_state],
                    [message_box, chatbot, session_state],
                    queue=False,
                ).then(
                    on_stream_assistant,
                    [chatbot, session_state],
                    [chatbot],
                )

                clear_btn.click(
                    on_clear_chat,
                    [session_state],
                    [chatbot, message_box, session_state],
                    queue=False,
                )
        return demo
