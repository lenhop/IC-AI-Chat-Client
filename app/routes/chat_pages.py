"""
Chat-related HTML routes: root redirect to Gradio and optional legacy Jinja UI.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.config import get_config

router = APIRouter()
TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/")
async def root_redirect() -> RedirectResponse:
    """Primary UX is Gradio at ``/gradio`` (m1_plan_v2)."""
    return RedirectResponse(url="/gradio", status_code=302)


@router.get("/legacy", response_class=HTMLResponse)
async def legacy_chat_page(request: Request) -> HTMLResponse:
    """
    Original Jinja2 + static JS chat page (transition / comparison).
    """
    cfg = get_config()
    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={
            "model_name": cfg.deepseek_llm_model if cfg.llm_backend == "deepseek" else cfg.ollama_generate_model,
            "user_id": cfg.user_id,
            "backend_name": cfg.llm_backend,
        },
    )
