"""Gradio chat UI package (avoid eager import to prevent heavy Gradio init on ``import app.ui``)."""

from __future__ import annotations

from typing import Any, List

__all__: List[str] = ["build_gradio_chat_blocks"]


def __getattr__(name: str) -> Any:
    if name == "build_gradio_chat_blocks":
        from app.ui.gradio_chat import build_gradio_chat_blocks

        return build_gradio_chat_blocks
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
