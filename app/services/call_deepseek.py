"""
DeepSeek wrapper referenced from IC-RAG-Agent style implementation.

This module keeps DeepSeek-specific API handling isolated:
- Read config from environment.
- Validate message format.
- Provide streaming and non-stream chat methods.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

from openai import OpenAI

from app.services.llm_chunks import ChatStreamChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeepSeekConfig:
    """Immutable DeepSeek runtime configuration."""

    api_key: str
    llm_model: str
    base_url: str
    request_timeout: int


def get_deepseek_config() -> DeepSeekConfig:
    """
    Load DeepSeek configuration from environment variables.

    Required:
      - DEEPSEEK_API_KEY
    Optional:
      - DEEPSEEK_LLM_MODEL
      - DEEPSEEK_BASE_URL
      - DEEPSEEK_REQUEST_TIMEOUT
    """
    api_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY must be set for DeepSeek backend")

    model = (os.getenv("DEEPSEEK_LLM_MODEL") or "deepseek-chat").strip()
    base_url = (os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").strip().rstrip("/")
    timeout_raw = (os.getenv("DEEPSEEK_REQUEST_TIMEOUT") or "600").strip()
    try:
        timeout = int(timeout_raw)
    except ValueError as exc:
        raise ValueError("DEEPSEEK_REQUEST_TIMEOUT must be a positive integer") from exc
    if timeout <= 0:
        raise ValueError("DEEPSEEK_REQUEST_TIMEOUT must be positive")

    return DeepSeekConfig(
        api_key=api_key,
        llm_model=model,
        base_url=base_url,
        request_timeout=timeout,
    )


class DeepSeekClient:
    """DeepSeek chat client using OpenAI-compatible API."""

    def __init__(self, config: Optional[DeepSeekConfig] = None) -> None:
        """
        Args:
            config: If set, use this instead of ``get_deepseek_config()`` (library mode).
        """
        self._config_override = config

    def _resolve_config(self) -> DeepSeekConfig:
        """Pick explicit override or environment-backed configuration."""
        if self._config_override is not None:
            return self._config_override
        return get_deepseek_config()

    def _client(self) -> tuple[OpenAI, DeepSeekConfig]:
        cfg = self._resolve_config()
        client = OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            timeout=cfg.request_timeout,
        )
        return client, cfg

    def complete_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        model_override: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """Return one full assistant answer text from DeepSeek."""
        client, cfg = self._client()
        model = (model_override or "").strip() or cfg.llm_model
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
            )
            choice = response.choices[0] if response.choices else None
            content = ""
            if choice and choice.message and choice.message.content:
                content = str(choice.message.content)
            content = content.strip()
            if not content:
                raise RuntimeError("DeepSeek returned empty assistant content")
            return content
        except Exception as exc:
            logger.error("DeepSeek completion failed: %s", exc, exc_info=True)
            raise RuntimeError(f"DeepSeek request failed: {exc}") from exc

    def stream_chat_chunks(
        self,
        messages: List[Dict[str, str]],
        *,
        model_override: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> Iterator[ChatStreamChunk]:
        """
        Stream OpenAI-compatible chunks mapped to :class:`ChatStreamChunk`.

        Emits ``content_delta`` / optional ``reasoning_delta``, then a final
        ``done=True`` chunk when the HTTP stream completes without error.
        """
        client, cfg = self._client()
        model = (model_override or "").strip() or cfg.llm_model
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue
                # Visible assistant text (standard chat models).
                text = getattr(delta, "content", None)
                if text:
                    yield ChatStreamChunk(content_delta=str(text))
                # Optional reasoning stream (some OpenAI-compatible / reasoning models).
                reasoning = getattr(delta, "reasoning_content", None) or getattr(
                    delta, "reasoning", None
                )
                if reasoning:
                    yield ChatStreamChunk(reasoning_delta=str(reasoning))
            yield ChatStreamChunk(done=True)
        except Exception as exc:
            logger.error("DeepSeek stream failed: %s", exc, exc_info=True)
            raise RuntimeError(f"DeepSeek stream failed: {exc}") from exc

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        model_override: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """Yield assistant token deltas from DeepSeek streaming API."""
        for ch in self.stream_chat_chunks(
            messages,
            model_override=model_override,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            if ch.content_delta:
                yield ch.content_delta
