"""
Stable import surface for embedding IC-AI-Chat-Client in another Python process.

Do not rely on ``app.main`` or HTTP routes for integration: construct
:class:`~app.runtime_config.RuntimeConfig`, pass it to :func:`stream_chat` or
:func:`complete_chat`, and manage your own transport (e.g. your app's REST or gRPC).

Example:

    from app.integrations import RuntimeConfig, stream_chat, normalize_messages

    cfg = RuntimeConfig(
        llm_backend="deepseek",
        deepseek_api_key="sk-...",
    )
    for delta in stream_chat(
        normalize_messages([{"role": "user", "content": "Hello"}]),
        runtime=cfg,
    ):
        print(delta, end="", flush=True)
"""

from __future__ import annotations

from app.runtime_config import RuntimeConfig, validate_runtime_config
from app.services.call_llm import complete_chat, normalize_messages, stream_chat

__all__ = [
    "RuntimeConfig",
    "complete_chat",
    "normalize_messages",
    "stream_chat",
    "validate_runtime_config",
]
