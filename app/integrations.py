"""
Stable import surface for embedding IC-AI-Chat-Client in another Python process.

Do not rely on ``app.main`` or HTTP routes for integration: construct
:class:`~app.runtime_config.RuntimeConfig`, pass it to :func:`stream_chat`,
:func:`stream_chat_chunks`, or :func:`complete_chat`, and manage your own transport.

For structured streams (reasoning / ``done`` markers), use :func:`stream_chat_chunks`
and :class:`~app.services.llm_chunks.ChatStreamChunk`. Use :func:`list_chat_model_names`
to populate model dropdowns (Ollama lists tags; DeepSeek returns the configured id).

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
from app.services.call_llm import complete_chat, normalize_messages, stream_chat, stream_chat_chunks
from app.services.llm_chunks import ChatStreamChunk
from app.services.llm_models import list_chat_model_names

__all__ = [
    "ChatStreamChunk",
    "RuntimeConfig",
    "complete_chat",
    "list_chat_model_names",
    "normalize_messages",
    "stream_chat",
    "stream_chat_chunks",
    "validate_runtime_config",
]
