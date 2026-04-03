"""
Gradio chat UI backed by ``call_llm.stream_chat`` (same contract as SSE route).

Mount with ``gr.mount_gradio_app(fastapi_app, blocks, path="/gradio")``.
Supports injectable ``AppConfig``, optional ``RuntimeConfig`` for LLM calls,
and ``GRADIO_UI_THEME`` / ``theme=`` for business | warm | minimal skins (m1_plan_v3).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Generator, List, Optional, Tuple, cast

import gradio as gr

from app.config import AppConfig, get_config, get_gradio_ui_theme, validate_app_config_for_ui
from app.runtime_config import RuntimeConfig, validate_runtime_config
from app.services.call_llm import normalize_messages, stream_chat
from app.ui.gradio_themes import (
    GradioUiTheme,
    build_gradio_theme,
    normalize_ui_theme,
    theme_extra_css,
    theme_header_html,
)

logger = logging.getLogger(__name__)


def _clone_message_history(history: Any) -> List[Dict[str, Any]]:
    """Deep-copy Gradio ``type='messages'`` history."""
    history = history or []
    return [dict(cast(Dict[str, Any], m)) for m in history]


def _messages_for_api(history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Build OpenAI-style messages from history, excluding trailing empty assistant."""
    # If last message is assistant with empty content, do not include it in the API payload.
    trimmed = history
    if (
        trimmed
        and trimmed[-1].get("role") == "assistant"
        and not (str(trimmed[-1].get("content") or "").strip())
    ):
        trimmed = trimmed[:-1]
    out: List[Dict[str, str]] = []
    for m in trimmed:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role in {"system", "user", "assistant"} and content:
            out.append({"role": str(role), "content": content})
    return out


def build_gradio_chat_blocks(
    app_config: Optional[AppConfig] = None,
    *,
    theme: Optional[str] = None,
    runtime: Optional[RuntimeConfig] = None,
) -> gr.Blocks:
    """
    Build Gradio Blocks for mounting under FastAPI.

    Args:
        app_config: If ``None``, uses ``get_config()`` from environment.
        theme: ``business`` | ``warm`` | ``minimal``; overrides ``GRADIO_UI_THEME`` when set.
        runtime: If set, ``stream_chat(..., runtime=...)`` and skip env-based LLM config for calls.

    Returns:
        Configured ``gr.Blocks`` with ``Chatbot(type='messages')`` and streaming callbacks.
    """
    resolved_cfg = app_config if app_config is not None else get_config()
    validate_app_config_for_ui(resolved_cfg)

    if runtime is not None:
        validate_runtime_config(runtime)

    theme_key: GradioUiTheme = normalize_ui_theme(theme if theme is not None else get_gradio_ui_theme())
    gradio_theme = build_gradio_theme(theme_key)
    extra_css = theme_extra_css(theme_key)

    def _user_turn(message: str, history: Any) -> Tuple[str, Any]:
        history_list = _clone_message_history(history)
        text = (message or "").strip()
        if not text:
            return "", history_list
        return "", history_list + [{"role": "user", "content": text}]

    def _stream_assistant(history: Any) -> Generator[Any, None, None]:
        history_list = _clone_message_history(history)
        if not history_list or history_list[-1].get("role") != "user":
            yield _clone_message_history(history_list)
            return

        history_list.append({"role": "assistant", "content": ""})
        yield _clone_message_history(history_list)

        api_messages = _messages_for_api(history_list)
        try:
            normalized = normalize_messages(api_messages)
        except ValueError as exc:
            history_list[-1]["content"] = f"[错误] {exc}"
            yield _clone_message_history(history_list)
            return

        accumulated = ""
        stream_kw: Dict[str, Any] = {"messages": normalized}
        if runtime is not None:
            stream_kw["runtime"] = runtime

        try:
            for delta in stream_chat(**stream_kw):
                accumulated += delta
                history_list[-1]["content"] = accumulated
                yield _clone_message_history(history_list)
        except (RuntimeError, ValueError) as exc:
            logger.warning("Gradio stream_chat failed: %s", exc)
            history_list[-1]["content"] = accumulated + (
                f"\n[错误] {exc}" if accumulated else f"[错误] {exc}"
            )
            yield _clone_message_history(history_list)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Gradio chat unexpected failure")
            history_list[-1]["content"] = accumulated + (
                f"\n[错误] {exc}" if accumulated else f"[错误] {exc}"
            )
            yield _clone_message_history(history_list)

    titles = {
        "business": "IC-AI Chat · 商务",
        "warm": "IC-AI Chat · 温馨",
        "minimal": "IC-AI Chat",
    }
    page_title = titles.get(theme_key, "IC-AI Chat")

    # PC layout: stretch blocks to viewport width (pairs with theme_extra_css).
    with gr.Blocks(
        title=page_title,
        theme=gradio_theme,
        analytics_enabled=False,
        css=extra_css,
        fill_width=True,
    ) as demo:
        # Root column: high min_width so empty chat does not start as a narrow strip;
        # CSS class icai-chat-root pairs with theme_extra_css for stable full-width flex.
        # min_width: avoid Gradio sizing the column to empty chat intrinsic width (~320px).
        with gr.Column(scale=1, min_width=960, elem_classes=["icai-chat-root"]):
            header_md = theme_header_html(resolved_cfg, theme_key)
            if theme_key == "minimal":
                gr.Markdown(header_md)
            else:
                gr.HTML(header_md)

            chatbot = gr.Chatbot(
                height=560,
                label="对话",
                type="messages",
                layout="bubble",
                scale=1,
            )
            msg = gr.Textbox(
                show_label=False,
                lines=2,
                placeholder="输入消息后按发送或 Enter…",
                scale=1,
            )
            with gr.Row():
                submit_btn = gr.Button("发送", scale=1)
                clear_btn = gr.Button("清空会话", scale=1)

            submit_event = msg.submit(_user_turn, [msg, chatbot], [msg, chatbot], queue=False)
            submit_event.then(_stream_assistant, chatbot, chatbot)

            submit_btn.click(_user_turn, [msg, chatbot], [msg, chatbot], queue=False).then(
                _stream_assistant,
                chatbot,
                chatbot,
            )

            clear_btn.click(lambda: ([], ""), None, [chatbot, msg], queue=False)

    return demo
