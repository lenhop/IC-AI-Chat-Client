"""
FastAPI application entry.

Key points:
1) Standalone: load only this repo's ``PROJECT_ROOT/.env`` (missing file -> raise).
2) Mount Gradio chat UI at ``/gradio``; ``GET /`` redirects to Gradio.
3) M3: optional Redis in ``lifespan`` for Gradio history/session persistence.

Library integration does not use this module: import ``app.integrations`` and pass
``RuntimeConfig`` to ``stream_chat`` / ``complete_chat``.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware

from app.config import get_config, get_redis_settings, validate_standalone_env
from app.memory.redis_pool import close_redis_client, create_sync_redis_client
from app.memory.redis_runtime import bind_redis_for_gradio, clear_redis_for_gradio
from app.routes.chat_pages import router as chat_pages_router
from app.routes.message_ingress import (
    MessageIngressRouteFacade,
    message_ingress_v1,
    router as message_ingress_router,
)
from app.services.message_ingress import MessageIngressResult
from app.ui.gradio_chat import mount_gradio_chat_app


def _sanitize_redis_url(url: str) -> str:
    """Hide credentials in redis URL for error messages."""
    u = (url or "").strip()
    if not u:
        return "(empty)"
    if "@" in u and "redis://" in u:
        try:
            hostpart = u.split("@", 1)[-1]
            return f"redis://***@{hostpart}"
        except (IndexError, ValueError):
            return "redis://***"
    return u


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
ENV_FILE = PROJECT_ROOT / ".env"

# Standalone policy: exactly one env file at repository root (no cwd / parent fallback).
if not ENV_FILE.is_file():
    raise RuntimeError(
        f"Standalone mode requires {ENV_FILE} to exist. "
        "Copy .env.example to .env and set variables for your LLM_BACKEND."
    )
load_dotenv(ENV_FILE, override=False)
validate_standalone_env()

# Read once at import so middleware + conditional routers match lifespan behavior.
_REDIS_BOOTSTRAP = get_redis_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Open Redis when enabled; bind client for Gradio callbacks; close on shutdown.
    """
    rs = get_redis_settings()
    app.state.redis_settings = rs
    app.state.redis = None
    client = None
    try:
        if rs.enabled:
            try:
                client = create_sync_redis_client(rs.url)
                client.ping()
            except Exception as exc:
                raise RuntimeError(
                    "REDIS_ENABLED=true but Redis is unreachable "
                    f"({_sanitize_redis_url(rs.url)}): {exc}"
                ) from exc
            app.state.redis = client
        bind_redis_for_gradio(app.state.redis, rs)
        yield
    finally:
        clear_redis_for_gradio()
        close_redis_client(client)
        app.state.redis = None


app = FastAPI(
    title="IC-AI-Chat-Client",
    version="0.4.0",
    description=(
        "FastAPI + Gradio chat UI with DeepSeek/Ollama; optional Redis persistence (M3). "
        f"Loaded env from {ENV_FILE}."
    ),
    lifespan=lifespan,
)

# Signed cookie for Gradio session id persistence (same-browser refresh).
if _REDIS_BOOTSTRAP.enabled:
    _secret = (os.getenv("SECRET_KEY") or "").strip() or "change-me-to-a-random-secret"
    app.add_middleware(SessionMiddleware, secret_key=_secret)

app.include_router(chat_pages_router)
app.include_router(message_ingress_router)
_cfg = get_config()
for alias_path in MessageIngressRouteFacade.resolve_alias_paths(_cfg.chat_ui_ingress_path):
    app.add_api_route(
        alias_path,
        message_ingress_v1,
        methods=["POST"],
        response_model=MessageIngressResult,
    )
mount_gradio_chat_app(app, path="/gradio")
