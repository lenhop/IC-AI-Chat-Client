"""
FastAPI app exposing ``POST /v1/chat/stream`` for the UI process when ``LLM_TRANSPORT=http``.

Loads repository-root ``.env`` when present; validates only LLM backend credentials
(see :func:`app.config.validate_llm_worker_env`). Does not connect to Redis.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Generator, List, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import validate_llm_worker_env
from app.services.call_llm import normalize_messages, stream_chat

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = PROJECT_ROOT / ".env"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load dotenv then enforce LLM worker env before serving."""
    if _ENV_FILE.is_file():
        load_dotenv(_ENV_FILE, override=False)
    try:
        validate_llm_worker_env()
    except RuntimeError as exc:
        logger.error("LLM worker env invalid: %s", exc)
        raise
    yield


app = FastAPI(
    title="IC-AI LLM Service",
    version="0.1.0",
    description="Streams chat completions for IC-AI-Chat-Client UI over SSE.",
    lifespan=lifespan,
)


class _MessageItem(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class _StreamRequest(BaseModel):
    messages: List[_MessageItem] = Field(min_length=1)
    backend: Optional[str] = None
    model: Optional[str] = None


def _sse_frame(data: dict) -> str:
    """One SSE line with JSON body."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/v1/chat/stream")
async def chat_stream_v1(body: _StreamRequest) -> StreamingResponse:
    """
    Stream assistant deltas as ``text/event-stream`` (``delta`` / ``done`` / ``error`` frames).

    Request body matches the UI transport payload: ``messages``, optional ``backend``, ``model``.
    """
    raw_msgs = [{"role": m.role, "content": m.content} for m in body.messages]
    try:
        normalized = normalize_messages(raw_msgs)
    except ValueError as exc:
        logger.warning("llm_service: invalid messages: %s", exc)
        err_text = str(exc)

        def err_only() -> Generator[str, None, None]:
            # Capture message text outside except scope for Python 3.11+.
            yield _sse_frame({"error": err_text})

        return StreamingResponse(
            err_only(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    def event_generator() -> Generator[str, None, None]:
        assistant_text = ""
        try:
            for delta in stream_chat(
                normalized,
                backend=body.backend,
                model_override=body.model,
            ):
                assistant_text += delta
                yield _sse_frame({"delta": delta})
            yield _sse_frame({"done": True})
        except ValueError as exc:
            logger.warning("llm_service stream validation: %s", exc)
            yield _sse_frame({"error": str(exc)})
        except RuntimeError as exc:
            logger.warning("llm_service stream failed: %s", exc)
            yield _sse_frame({"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            logger.exception("llm_service unexpected failure")
            yield _sse_frame({"error": "Internal error while streaming."})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
