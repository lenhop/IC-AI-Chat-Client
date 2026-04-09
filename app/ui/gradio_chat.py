"""
Gradio chat UI backed by ``llm_transport.iter_chat_text_deltas`` (same contract as SSE route).

Mount with ``gr.mount_gradio_app(fastapi_app, blocks, path="/gradio")``.
Supports injectable ``AppConfig``, optional ``RuntimeConfig`` for LLM calls,
and ``GRADIO_UI_THEME`` / ``theme=`` for business | warm | minimal skins (m1_plan_v3).

M3: when Redis is enabled, Starlette ``SessionMiddleware`` stores ``icai_gradio_session_id``;
page load hydrates history (last ``MEMORY_ROUNDS`` rounds when ``> 0``); successful turns append
to Redis; clear truncates stored messages.

``CHAT_MODE=prompt_template`` + Redis: LLM receives ``chat_prompt.md`` with ``{historical_message}``
and ``{current_query}`` (not ``app.integrations``).
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, Generator, List, Optional, Tuple, cast

import gradio as gr

from app.config import AppConfig, MessageDisplayOptions, get_config, get_gradio_ui_theme, validate_app_config_for_ui
from app.memory.redis_runtime import get_redis_for_gradio
from app.memory.session_store import (
    SessionAccessDeniedError,
    SessionNotFoundError,
    SessionStore,
    gradio_history_from_stored,
)
from app.runtime_config import RuntimeConfig, validate_runtime_config
from app.services.llm_transport import iter_chat_text_deltas, validate_or_normalize_messages
from app.services.prompt_render import (
    format_messages_markdown_for_prompt,
    render_chat_prompt,
    select_rounds_for_prompt,
    select_rounds_for_ui,
)
from app.ui.gradio_session_turn import GradioSessionTurn
from app.ui.gradio_themes import (
    GradioUiTheme,
    build_gradio_theme,
    normalize_ui_theme,
    theme_extra_css,
    theme_header_html,
)

logger = logging.getLogger(__name__)

# Cookie/session key for Gradio browser refresh (paired with SessionMiddleware in main).
_GRADIO_SESSION_KEY = "icai_gradio_session_id"
_STAGE_MESSAGE_TYPES = frozenset(
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


def _starlette_session_from_gradio(request: gr.Request) -> Any:
    """Resolve Starlette session mapping from Gradio request wrapper (may be None)."""
    inner = getattr(request, "request", None)
    if inner is None:
        return None
    sess = getattr(inner, "session", None)
    return sess


def _clone_message_history(history: Any) -> List[Dict[str, Any]]:
    """Deep-copy Gradio ``type='messages'`` history."""
    history = history or []
    return [dict(cast(Dict[str, Any], m)) for m in history]


def _persist_stage_message(
    session_id: Optional[str],
    request: gr.Request,
    *,
    user_id: str,
    message_type: str,
    content: str,
) -> None:
    """
    Append one non-empty stage message to Redis for the active turn.

    This helper writes only typed business-stage rows (not token deltas).
    """
    sid = (session_id or "").strip()
    if not sid:
        return
    mtype = (message_type or "").strip()
    if mtype not in _STAGE_MESSAGE_TYPES:
        return
    text = (content or "").strip()
    if not text:
        return
    client, rs = get_redis_for_gradio()
    if rs is None or not rs.enabled or client is None:
        return
    sess = _starlette_session_from_gradio(request)
    turn_id = GradioSessionTurn.get_active_turn_id(sess)
    if not turn_id:
        return
    try:
        SessionStore(client, rs).append_memory_message(
            sid,
            user_id,
            message_type=mtype,
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
        runtime: If set, ``iter_chat_text_deltas(..., runtime=...)`` uses in-process LLM only.

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

    def _page_load(request: gr.Request) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """Create or resume Redis session; hydrate chat from stored messages."""
        client, rs = get_redis_for_gradio()
        if rs is None or not rs.enabled or client is None:
            return None, []
        store = SessionStore(client, rs)
        sess = _starlette_session_from_gradio(request)
        sid: Optional[str] = None
        if sess is not None:
            raw_sid = sess.get(_GRADIO_SESSION_KEY)
            if isinstance(raw_sid, str) and raw_sid.strip():
                sid = raw_sid.strip()
        if sid:
            try:
                raw_msgs = store.get_messages(sid, resolved_cfg.user_id)
                display = (
                    select_rounds_for_ui(raw_msgs, resolved_cfg.memory_rounds)
                    if resolved_cfg.memory_rounds > 0
                    else raw_msgs
                )
                return sid, gradio_history_from_stored(
                    display,
                    MessageDisplayOptions.from_app_config(resolved_cfg),
                )
            except (SessionNotFoundError, SessionAccessDeniedError):
                if sess is not None:
                    sess.pop(_GRADIO_SESSION_KEY, None)
        new_sid = store.create_session(resolved_cfg.user_id, resolved_cfg.llm_backend)
        if sess is not None:
            sess[_GRADIO_SESSION_KEY] = new_sid
            GradioSessionTurn.clear_active_turn_id(sess)
        return new_sid, []

    def _persist_answer_after_stream(
        session_id: Optional[str],
        request: gr.Request,
        assistant_text: str,
    ) -> None:
        """
        Append ``answer`` for the active Starlette ``turn_id`` and clear that turn.

        Query rows are written in ``_user_turn``; failures here are logged only.
        """
        text = (assistant_text or "").strip()
        if not session_id or not text:
            return
        client, rs = get_redis_for_gradio()
        if rs is None or not rs.enabled or client is None:
            return
        sess = _starlette_session_from_gradio(request)
        turn_id = GradioSessionTurn.get_active_turn_id(sess)
        if not turn_id:
            return
        try:
            store = SessionStore(client, rs)
            store.append_memory_message(
                str(session_id),
                resolved_cfg.user_id,
                message_type="answer",
                content=text,
                turn_id=turn_id,
            )
            GradioSessionTurn.clear_active_turn_id(sess)
        except SessionNotFoundError:
            logger.warning("gradio redis persist answer: session not found %s", session_id)
        except SessionAccessDeniedError:
            logger.warning("gradio redis persist answer: access denied %s", session_id)
        except ValueError as exc:
            logger.warning("gradio redis persist answer: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("gradio redis persist answer failed: %s", exc)

    def _user_turn(
        message: str,
        history: Any,
        session_id: Any,
        request: gr.Request,
    ) -> Tuple[str, Any, Any]:
        history_list = _clone_message_history(history)
        text = (message or "").strip()
        if not text:
            return "", history_list, session_id

        client, rs = get_redis_for_gradio()
        sid = session_id
        if sid and client and rs is not None and rs.enabled:
            sess = _starlette_session_from_gradio(request)
            try:
                turn_id = GradioSessionTurn.ensure_active_turn_id(sess)
                SessionStore(client, rs).append_memory_message(
                    str(sid),
                    resolved_cfg.user_id,
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

        return "", history_list + [{"role": "user", "content": text}], session_id

    def _stream_assistant(
        history: Any,
        session_id: Any,
        request: gr.Request,
    ) -> Generator[Any, None, None]:
        history_list = _clone_message_history(history)
        if not history_list or history_list[-1].get("role") != "user":
            yield _clone_message_history(history_list)
            return

        history_list.append({"role": "assistant", "content": ""})
        yield _clone_message_history(history_list)

        api_messages = _messages_for_api(history_list)
        try:
            normalized = validate_or_normalize_messages(api_messages)
        except ValueError as exc:
            history_list[-1]["content"] = f"[错误] {exc}"
            yield _clone_message_history(history_list)
            return

        use_tpl = (
            runtime is None
            and resolved_cfg.chat_mode == "prompt_template"
            and session_id
        )
        client, rs = get_redis_for_gradio()
        if use_tpl and client and rs is not None and rs.enabled and len(history_list) >= 2:
            user_text = (str(history_list[-2].get("content") or "")).strip()
            if user_text:
                try:
                    st = SessionStore(client, rs)
                    stored = st.get_messages(str(session_id), resolved_cfg.user_id)
                    hist_subset = select_rounds_for_prompt(stored, resolved_cfg.memory_rounds)
                    hist_md = format_messages_markdown_for_prompt(hist_subset)
                    full = render_chat_prompt(current_query=user_text, historical_message=hist_md)
                    normalized = validate_or_normalize_messages([{"role": "user", "content": full}])
                except Exception as exc:  # noqa: BLE001
                    logger.warning("gradio prompt_template fallback: %s", exc)

        accumulated = ""
        stream_kw: Dict[str, Any] = {"messages": normalized}
        if runtime is not None:
            stream_kw["runtime"] = runtime
        stream_kw["on_stage_message"] = (
            lambda message_type, content: _persist_stage_message(
                session_id,
                request,
                user_id=resolved_cfg.user_id,
                message_type=message_type,
                content=content,
            )
        )

        try:
            for delta in iter_chat_text_deltas(**stream_kw):
                accumulated += delta
                history_list[-1]["content"] = accumulated
                yield _clone_message_history(history_list)
            _persist_answer_after_stream(session_id, request, accumulated)
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

    def _clear_chat(session_id: Any, request: gr.Request) -> Tuple[List[Dict[str, Any]], str, Any]:
        """Clear UI; truncate Redis message list for current session when enabled."""
        sess = _starlette_session_from_gradio(request)
        GradioSessionTurn.clear_active_turn_id(sess)
        sid = session_id
        client, rs = get_redis_for_gradio()
        if sid and client and rs is not None and rs.enabled:
            try:
                SessionStore(client, rs).clear_messages(str(sid), resolved_cfg.user_id)
            except (SessionNotFoundError, SessionAccessDeniedError) as exc:
                logger.warning("gradio clear session: %s", exc)
            except Exception as exc:  # noqa: BLE001
                logger.warning("gradio clear redis: %s", exc)
        return [], "", sid

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

            session_state = gr.State(value=None)
            # Gradio 4.x defaults to tuples ``[[user, bot], ...]``; we use ``{role, content}``
            # rows everywhere. Gradio 6+ dropped ``type=`` (messages-only). Pass ``messages``
            # only when the constructor still accepts it so ``<7`` and 6+ both work.
            _chatbot_kw: Dict[str, Any] = {
                "height": 560,
                "label": "对话",
                "layout": "bubble",
                "scale": 1,
            }
            try:
                if "type" in inspect.signature(gr.Chatbot.__init__).parameters:
                    _chatbot_kw["type"] = "messages"
            except (TypeError, ValueError) as exc:
                logger.debug("Chatbot signature introspection skipped: %s", exc)
            chatbot = gr.Chatbot(**_chatbot_kw)
            msg = gr.Textbox(
                show_label=False,
                lines=2,
                placeholder="输入消息后按发送或 Enter…",
                scale=1,
            )
            with gr.Row():
                submit_btn = gr.Button("发送", scale=1)
                clear_btn = gr.Button("清空会话", scale=1)

            demo.load(_page_load, None, [session_state, chatbot], queue=False)

            submit_event = msg.submit(
                _user_turn,
                [msg, chatbot, session_state, gr.Request()],
                [msg, chatbot, session_state],
                queue=False,
            )
            submit_event.then(
                _stream_assistant,
                [chatbot, session_state, gr.Request()],
                [chatbot],
            )

            submit_btn.click(
                _user_turn,
                [msg, chatbot, session_state, gr.Request()],
                [msg, chatbot, session_state],
                queue=False,
            ).then(
                _stream_assistant,
                [chatbot, session_state, gr.Request()],
                [chatbot],
            )

            clear_btn.click(
                _clear_chat,
                [session_state, gr.Request()],
                [chatbot, msg, session_state],
                queue=False,
            )

    return demo
