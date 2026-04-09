"""Chat-related routes: keep ``/`` redirecting to Gradio main UI."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import RedirectResponse

router = APIRouter()


@router.get("/")
async def root_redirect() -> RedirectResponse:
    """Primary UX is Gradio at ``/gradio`` (m1_plan_v2)."""
    return RedirectResponse(url="/gradio", status_code=302)
