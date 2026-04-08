"""
Process-local Redis handle for Gradio callbacks (set from FastAPI lifespan).

Gradio blocks are built at import time; the actual client is bound after startup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    import redis

    from app.config import RedisSettings

_redis_client: Optional["redis.Redis"] = None
_redis_settings: Optional["RedisSettings"] = None


def bind_redis_for_gradio(client: Optional["redis.Redis"], settings: "RedisSettings") -> None:
    """
    Store Redis client and settings for Gradio server-side callbacks.

    Args:
        client: Live client when Redis enabled, else None.
        settings: Parsed RedisSettings from environment.
    """
    global _redis_client, _redis_settings
    _redis_client = client
    _redis_settings = settings


def get_redis_for_gradio() -> Tuple[Optional["redis.Redis"], Optional["RedisSettings"]]:
    """Return (client, settings) tuple for Gradio hooks."""
    return _redis_client, _redis_settings


def clear_redis_for_gradio() -> None:
    """Clear references on shutdown (before closing the client)."""
    global _redis_client, _redis_settings
    _redis_client = None
    _redis_settings = None
