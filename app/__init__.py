"""Expose FastAPI app for importers (lazy to avoid importing Gradio on ``import app.config``)."""

from __future__ import annotations

from typing import Any

__all__ = ["app"]


def __getattr__(name: str) -> Any:
    if name == "app":
        from app.main import app as _app

        globals()["app"] = _app
        return _app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
