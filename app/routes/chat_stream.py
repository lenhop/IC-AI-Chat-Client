"""
SSE streaming endpoint for chat messages.

Key points:
1) Validate request body via Pydantic models.
2) Delegate streaming to ``llm_transport.iter_chat_text_deltas`` (local or HTTP worker).
3) Stream JSON frames with SSE format for frontend incremental rendering.
4) M3: optional session_id persists the completed user/assistant turn to Redis (best-effort).
5) CHAT_MODE=prompt_template + Redis: single user message from chat_prompt.md + history.
"""

from __future__ import annotations

import json
import logging
from typing import Generator, List, Literal, Optional, Tuple

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import get_config
from app.memory.session_store import (
    SessionAccessDeniedError,
    SessionNotFoundError,
    SessionStore,
)
from app.services.llm_transport import iter_chat_text_deltas
from app.services.prompt_render import (
    format_messages_markdown_for_prompt,
    render_chat_prompt,
    select_rounds_for_prompt,
)

logger = logging.getLogger(__name__)
# OpenAPI grouping: these routes exist only for the bundled Jinja2 UI, not as a public integration API.
router = APIRouter(tags=["internal_ui"])


class MessageItem(BaseModel):
    """Single chat message from client request body."""

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatStreamRequest(BaseModel):
    """POST payload for /api/chat/stream."""

    messages: List[MessageItem] = Field(min_length=1)
    backend: Optional[str] = None
    model: Optional[str] = None
    session_id: Optional[str] = None


def _sse(data: dict) -> str:
    """Format JSON payload as one SSE frame."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _try_persist_session_turn(
    request: Request,
    session_id: Optional[str],
    messages: List[dict],
    assistant_text: str,
    *,
    last_user_override: Optional[str] = None,
) -> None:
    """
    Append user + assistant messages to Redis when session_id and Redis are active.

    Failures are logged only; SSE must not fail because of Redis.
    """
    sid = (session_id or "").strip()
    if not sid:
        return
    rs = getattr(request.app.state, "redis_settings", None)
    client = getattr(request.app.state, "redis", None)
    if rs is None or not rs.enabled or client is None:
        return
    last_user = (last_user_override or "").strip()
    if not last_user:
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = (m.get("content") or "").strip()
                break
    if not last_user:
        return
    cfg = get_config()
    try:
        store = SessionStore(client, rs)
        store.append_turn(sid, cfg.user_id, last_user, assistant_text)
    except SessionNotFoundError:
        logger.warning("redis persist: session not found %s", sid)
    except SessionAccessDeniedError:
        logger.warning("redis persist: access denied %s", sid)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("redis persist failed: %s", exc)


def _messages_for_stream_and_persist_user(
    payload: ChatStreamRequest,
    request: Request,
) -> Tuple[List[dict], Optional[str]]:
    """
    Build message list for stream_chat and optional real user text for Redis persist.

    When CHAT_MODE=prompt_template and Redis session is available, replace payload
    with a single user message built from chat_prompt.md.

    Returns:
        (messages_for_llm, last_user_for_persist) — persist uses short user query when template.
    """
    cfg = get_config()
    base: List[dict] = [{"role": m.role, "content": m.content} for m in payload.messages]
    last_user = ""
    for m in reversed(base):
        if m.get("role") == "user":
            last_user = (m.get("content") or "").strip()
            break
    if cfg.chat_mode != "prompt_template":
        return base, None
    sid = (payload.session_id or "").strip()
    if not sid or not last_user:
        return base, None
    rs = getattr(request.app.state, "redis_settings", None)
    client = getattr(request.app.state, "redis", None)
    if rs is None or not rs.enabled or client is None:
        return base, None
    try:
        store = SessionStore(client, rs)
        stored = store.get_messages(sid, cfg.user_id)
        hist_subset = select_rounds_for_prompt(stored, cfg.memory_rounds)
        hist_md = format_messages_markdown_for_prompt(hist_subset)
        full = render_chat_prompt(current_query=last_user, historical_message=hist_md)
        return ([{"role": "user", "content": full}], last_user)
    except SessionNotFoundError:
        logger.warning("prompt_template: session not found %s", sid)
    except SessionAccessDeniedError:
        logger.warning("prompt_template: access denied %s", sid)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("prompt_template build failed, using client messages: %s", exc)
    return base, None


@router.post("/api/chat/stream")
async def chat_stream(payload: ChatStreamRequest, request: Request) -> StreamingResponse:
    """
    Stream assistant deltas as ``text/event-stream``.

    Intended for the built-in chat page only. Host applications should import
    ``app.integrations`` and call ``stream_chat(..., runtime=...)`` instead of
    proxying this endpoint.

    Response frames:
      - {\"delta\": \"...\"}
      - {\"done\": true}
      - {\"error\": \"...\"}
    """
    stream_messages, persist_user_override = _messages_for_stream_and_persist_user(payload, request)

    def event_generator() -> Generator[str, None, None]:
        assistant_text = ""
        try:
            for delta in iter_chat_text_deltas(
                stream_messages,
                backend=payload.backend,
                model_override=payload.model,
            ):
                assistant_text += delta
                yield _sse({"delta": delta})
            yield _sse({"done": True})
            _try_persist_session_turn(
                request,
                payload.session_id,
                stream_messages,
                assistant_text,
                last_user_override=persist_user_override,
            )
        except ValueError as exc:
            logger.warning("chat request validation failed: %s", exc)
            yield _sse({"error": str(exc)})
        except RuntimeError as exc:
            logger.warning("chat stream failed: %s", exc)
            yield _sse({"error": str(exc)})
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.exception("unexpected chat stream failure")
            yield _sse({"error": "Internal server error while processing chat stream."})

    if not stream_messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty list")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
