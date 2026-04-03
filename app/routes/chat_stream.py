"""
SSE streaming endpoint for chat messages.

Key points:
1) Validate request body via Pydantic models.
2) Delegate backend selection and execution to call_llm wrapper.
3) Stream JSON frames with SSE format for frontend incremental rendering.
"""

from __future__ import annotations

import json
import logging
from typing import Generator, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.call_llm import stream_chat

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


def _sse(data: dict) -> str:
    """Format JSON payload as one SSE frame."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/api/chat/stream")
async def chat_stream(payload: ChatStreamRequest) -> StreamingResponse:
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
    normalized_messages = [{"role": m.role, "content": m.content} for m in payload.messages]

    def event_generator() -> Generator[str, None, None]:
        try:
            # Centralized backend routing (deepseek / ollama) lives in call_llm.
            for delta in stream_chat(
                messages=normalized_messages,
                backend=payload.backend,
                model_override=payload.model,
            ):
                yield _sse({"delta": delta})
            yield _sse({"done": True})
        except ValueError as exc:
            logger.warning("chat request validation failed: %s", exc)
            yield _sse({"error": str(exc)})
        except RuntimeError as exc:
            logger.warning("chat stream failed: %s", exc)
            yield _sse({"error": str(exc)})
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.exception("unexpected chat stream failure")
            yield _sse({"error": "Internal server error while processing chat stream."})

    if not normalized_messages:
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
