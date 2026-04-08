"""
Synchronous Redis client factory for FastAPI lifespan (matches sync stream_chat).

Async refactor is deferred; chat routes use run_in_threadpool for Redis I/O.
"""

from __future__ import annotations

import logging
from typing import Optional

import redis

logger = logging.getLogger(__name__)


def create_sync_redis_client(url: str) -> redis.Redis:
    """
    Create a decode_responses Redis client from URL.

    Args:
        url: redis:// host connection string.

    Returns:
        redis.Redis: connected client (caller should ping in startup).
    """
    client = redis.Redis.from_url(url, decode_responses=True)
    return client


def close_redis_client(client: Optional[redis.Redis]) -> None:
    """
    Best-effort close for app shutdown.

    Args:
        client: Active client or None.
    """
    if client is None:
        return
    try:
        client.close()
    except (redis.RedisError, OSError, RuntimeError) as exc:
        logger.debug("redis close ignored: %s", exc)
