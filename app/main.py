"""
FastAPI application entry.

Key points:
1) Standalone: load only this repo's ``PROJECT_ROOT/.env`` (missing file -> raise).
2) Mount static assets for the legacy Jinja page at ``/legacy``.
3) Mount Gradio chat UI at ``/gradio``; ``GET /`` redirects to Gradio.
4) Register chat SSE route for the legacy UI (``internal_ui``).

Library integration does not use this module: import ``app.integrations`` and pass
``RuntimeConfig`` to ``stream_chat`` / ``complete_chat``.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import validate_standalone_env
from app.routes.chat_pages import router as chat_pages_router
from app.routes.chat_stream import router as chat_router
from app.ui.gradio_chat import build_gradio_chat_blocks


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
STATIC_DIR = BASE_DIR / "static"
ENV_FILE = PROJECT_ROOT / ".env"

# Standalone policy: exactly one env file at repository root (no cwd / parent fallback).
if not ENV_FILE.is_file():
    raise RuntimeError(
        f"Standalone mode requires {ENV_FILE} to exist. "
        "Copy .env.example to .env and set variables for your LLM_BACKEND."
    )
load_dotenv(ENV_FILE, override=False)
validate_standalone_env()

app = FastAPI(
    title="IC-AI-Chat-Client",
    version="0.3.0",
    description=(
        "M1 FastAPI + Gradio chat UI with DeepSeek/Ollama. "
        f"Loaded env from {ENV_FILE}."
    ),
)

# Legacy Jinja chat page uses /static assets.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(chat_pages_router)
app.include_router(chat_router)

gr.mount_gradio_app(app, build_gradio_chat_blocks(), path="/gradio")
