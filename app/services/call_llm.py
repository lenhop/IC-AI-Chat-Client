"""
Unified LLM wrapper for DeepSeek and Ollama.

This is the only module that routes backend names, so API routes stay simple.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Iterator, List, Optional

from app.runtime_config import RuntimeConfig, validate_runtime_config
from app.services.call_deepseek import DeepSeekClient, DeepSeekConfig
from app.services.call_ollama import OllamaClient, OllamaConfig
from app.services.llm_chunks import ChatStreamChunk, iter_text_deltas

logger = logging.getLogger(__name__)


def _runtime_to_deepseek(cfg: RuntimeConfig) -> DeepSeekConfig:
    """Map validated RuntimeConfig fields to an immutable DeepSeek client config."""
    return DeepSeekConfig(
        api_key=(cfg.deepseek_api_key or "").strip(),
        llm_model=(cfg.deepseek_llm_model or "").strip(),
        base_url=(cfg.deepseek_base_url or "").strip().rstrip("/"),
        request_timeout=cfg.deepseek_request_timeout,
    )


def _runtime_to_ollama(cfg: RuntimeConfig) -> OllamaConfig:
    """Map validated RuntimeConfig fields to an immutable Ollama client config."""
    return OllamaConfig(
        base_url=cfg.ollama_base_url,
        generate_model=cfg.ollama_generate_model,
        request_timeout=cfg.ollama_request_timeout,
        embed_model=cfg.ollama_embed_model,
    )


def _normalize_backend(raw: Optional[str]) -> str:
    """Normalize backend label to 'deepseek' or 'ollama'."""
    text = (raw or "").strip().lower()
    if text in {"deepseek", "ds"}:
        return "deepseek"
    if text in {"ollama", "local"}:
        return "ollama"
    if text:
        logger.warning("Unknown backend label %r, fallback to env/default", raw)
    env_backend = (os.getenv("LLM_BACKEND") or "deepseek").strip().lower()
    return "ollama" if env_backend in {"ollama", "local"} else "deepseek"


def normalize_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Validate role/content schema used by both backends."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")

    normalized: List[Dict[str, str]] = []
    for idx, item in enumerate(messages):
        if not isinstance(item, dict):
            raise ValueError(f"messages[{idx}] must be an object")
        role = item.get("role")
        content = item.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"messages[{idx}].role must be system/user/assistant")
        if not isinstance(content, str):
            raise ValueError(f"messages[{idx}].content must be a string")
        if not content.strip():
            raise ValueError(f"messages[{idx}].content must not be empty")
        normalized.append({"role": role, "content": content})
    return normalized


def stream_chat_chunks(
    messages: List[Dict[str, str]],
    *,
    backend: Optional[str] = None,
    model_override: Optional[str] = None,
    runtime: Optional[RuntimeConfig] = None,
) -> Iterator[ChatStreamChunk]:
    """
    Stream structured chunks (content / reasoning / done) for the active backend.

    Same routing and DeepSeek->Ollama fallback semantics as :func:`stream_chat`.
    """
    normalized = normalize_messages(messages)

    if runtime is not None:
        validate_runtime_config(runtime)
        if runtime.llm_backend == "deepseek":
            yield from DeepSeekClient(config=_runtime_to_deepseek(runtime)).stream_chat_chunks(
                normalized,
                model_override=model_override,
            )
            return
        yield from OllamaClient(config=_runtime_to_ollama(runtime)).stream_chat_chunks(
            normalized,
            model_override=model_override,
        )
        return

    target = _normalize_backend(backend)

    if target == "deepseek":
        try:
            yield from DeepSeekClient().stream_chat_chunks(
                normalized,
                model_override=model_override,
            )
            return
        except ValueError as exc:
            logger.warning("DeepSeek env invalid (%s), fallback to Ollama", exc)
            target = "ollama"
        except RuntimeError as exc:
            logger.warning("DeepSeek stream failed (%s), fallback to Ollama", exc)
            target = "ollama"

    if target == "ollama":
        yield from OllamaClient().stream_chat_chunks(
            normalized,
            model_override=model_override,
        )
        return

    raise ValueError(f"Unsupported backend: {target}")


def stream_chat(
    messages: List[Dict[str, str]],
    *,
    backend: Optional[str] = None,
    model_override: Optional[str] = None,
    runtime: Optional[RuntimeConfig] = None,
) -> Iterator[str]:
    """
    Stream assistant text deltas using selected backend (concatenation of ``content_delta``).

    Implemented as a thin wrapper over :func:`stream_chat_chunks`.
    """
    yield from iter_text_deltas(
        stream_chat_chunks(
            messages,
            backend=backend,
            model_override=model_override,
            runtime=runtime,
        )
    )


def complete_chat(
    messages: List[Dict[str, str]],
    *,
    backend: Optional[str] = None,
    model_override: Optional[str] = None,
    runtime: Optional[RuntimeConfig] = None,
) -> str:
    """
    Non-stream unified completion interface.

    This is useful for future APIs that do not require token-level streaming.

    Args:
        runtime: If set, same semantics as :func:`stream_chat`.
    """
    normalized = normalize_messages(messages)

    if runtime is not None:
        validate_runtime_config(runtime)
        if runtime.llm_backend == "deepseek":
            return DeepSeekClient(config=_runtime_to_deepseek(runtime)).complete_chat(
                normalized,
                model_override=model_override,
            )
        prompt_lines = []
        for item in normalized:
            prompt_lines.append(f"### {item['role'].capitalize()}\n{item['content']}")
        prompt_lines.append("### Assistant")
        prompt = "\n\n".join(prompt_lines)
        return OllamaClient(config=_runtime_to_ollama(runtime)).generate(
            prompt,
            model_override=model_override,
        )

    target = _normalize_backend(backend)

    if target == "deepseek":
        try:
            return DeepSeekClient().complete_chat(
                normalized,
                model_override=model_override,
            )
        except ValueError as exc:
            logger.warning("DeepSeek env invalid (%s), fallback to Ollama", exc)
            target = "ollama"
        except RuntimeError as exc:
            logger.warning("DeepSeek completion failed (%s), fallback to Ollama", exc)
            target = "ollama"

    if target == "ollama":
        # For Ollama we map chat messages into /api/generate style prompt.
        prompt_lines = []
        for item in normalized:
            prompt_lines.append(f"### {item['role'].capitalize()}\n{item['content']}")
        prompt_lines.append("### Assistant")
        prompt = "\n\n".join(prompt_lines)
        return OllamaClient().generate(prompt, model_override=model_override)

    raise ValueError(f"Unsupported backend: {target}")
