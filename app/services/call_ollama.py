"""
Ollama wrapper referenced from IC-RAG-Agent style implementation.

This module provides:
- generate() and embed() helpers
- stream_chat() for /api/chat NDJSON streaming
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

import httpx

from app.services.llm_chunks import ChatStreamChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OllamaConfig:
    """Immutable Ollama runtime configuration."""

    base_url: str
    generate_model: str
    request_timeout: int
    embed_model: str


def get_ollama_config() -> OllamaConfig:
    """Load and validate Ollama environment variables."""
    base_url = (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").strip().rstrip("/")
    generate_model = (os.getenv("OLLAMA_GENERATE_MODEL") or "qwen3:1.7b").strip()
    embed_model = (os.getenv("OLLAMA_EMBED_MODEL") or "all-minilm:latest").strip()
    timeout_raw = (os.getenv("OLLAMA_REQUEST_TIMEOUT") or "600").strip()

    if not base_url:
        raise ValueError("OLLAMA_BASE_URL is not set")
    if not generate_model:
        raise ValueError("OLLAMA_GENERATE_MODEL is not set")
    if not embed_model:
        raise ValueError("OLLAMA_EMBED_MODEL is not set")

    try:
        timeout = int(timeout_raw)
    except ValueError as exc:
        raise ValueError("OLLAMA_REQUEST_TIMEOUT must be a positive integer") from exc
    if timeout <= 0:
        raise ValueError("OLLAMA_REQUEST_TIMEOUT must be positive")

    return OllamaConfig(
        base_url=base_url,
        generate_model=generate_model,
        request_timeout=timeout,
        embed_model=embed_model,
    )


class OllamaClient:
    """Ollama HTTP client with generate/embed/stream chat APIs."""

    def __init__(self, config: Optional[OllamaConfig] = None) -> None:
        """
        Args:
            config: If set, use this instead of ``get_ollama_config()`` (library mode).
        """
        self._config_override = config

    def _resolve_config(self) -> OllamaConfig:
        """Pick explicit override or environment-backed configuration."""
        if self._config_override is not None:
            return self._config_override
        return get_ollama_config()

    @staticmethod
    def strip_markdown_fences(text: str) -> str:
        """Remove optional triple-backtick code fences from model output."""
        raw = (text or "").strip()
        if not raw.startswith("```"):
            return raw
        lines = raw.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
        return raw

    def generate(
        self,
        prompt: str,
        *,
        model_override: Optional[str] = None,
    ) -> str:
        """Call Ollama /api/generate and return final text."""
        cfg = self._resolve_config()
        url = f"{cfg.base_url}/api/generate"
        model = (model_override or "").strip() or cfg.generate_model
        payload = {"model": model, "prompt": prompt or "", "stream": False}
        timeout = httpx.Timeout(float(cfg.request_timeout), connect=15.0)

        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
            text = (data.get("response") or "").strip()
            if not text:
                raise RuntimeError("Ollama returned empty text")
            return self.strip_markdown_fences(text)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Ollama generate failed: {exc}") from exc
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Ollama invalid generate response: {exc}") from exc

    def embed(self, inputs: List[str], *, model_override: Optional[str] = None) -> List[List[float]]:
        """Call Ollama /api/embed and return embedding vectors."""
        cfg = self._resolve_config()
        if not isinstance(inputs, list) or not inputs:
            raise ValueError("inputs must be a non-empty list of strings")

        url = f"{cfg.base_url}/api/embed"
        model = (model_override or "").strip() or cfg.embed_model
        timeout = httpx.Timeout(float(cfg.request_timeout), connect=15.0)
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, json={"model": model, "input": inputs})
                response.raise_for_status()
                data = response.json()
            embeddings = data.get("embeddings") or []
            if len(embeddings) != len(inputs):
                raise ValueError(
                    f"Ollama embed expected {len(inputs)} vectors, got {len(embeddings)}"
                )
            return embeddings
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Ollama embed failed: {exc}") from exc
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Ollama invalid embed response: {exc}") from exc

    def stream_chat_chunks(
        self,
        messages: List[Dict[str, str]],
        *,
        model_override: Optional[str] = None,
    ) -> Iterator[ChatStreamChunk]:
        """
        Stream Ollama /api/chat NDJSON lines as :class:`ChatStreamChunk`.

        Normalizes cumulative vs incremental ``message.content`` to true deltas,
        then emits ``done=True`` after a successful stream.
        """
        cfg = self._resolve_config()
        url = f"{cfg.base_url}/api/chat"
        model = (model_override or "").strip() or cfg.generate_model
        payload = {"model": model, "messages": messages, "stream": True}
        timeout = httpx.Timeout(float(cfg.request_timeout), connect=15.0)
        accumulated = ""

        try:
            with httpx.Client(timeout=timeout) as client:
                with client.stream("POST", url, json=payload) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            logger.debug("skip non-json line from ollama stream")
                            continue
                        if obj.get("done"):
                            break

                        message = obj.get("message") or {}
                        piece = message.get("content") or ""
                        if not piece:
                            continue

                        if accumulated and piece.startswith(accumulated):
                            delta = piece[len(accumulated) :]
                            accumulated = piece
                        elif not accumulated:
                            delta = piece
                            accumulated = piece
                        else:
                            delta = piece
                            accumulated += piece
                        if delta:
                            yield ChatStreamChunk(content_delta=delta)
            yield ChatStreamChunk(done=True)
        except httpx.HTTPStatusError as exc:
            detail = self._extract_error_body(exc.response)
            raise RuntimeError(detail or f"Ollama HTTP {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            raise RuntimeError(
                "Cannot reach Ollama. Check OLLAMA_BASE_URL and that ollama serve is running."
            ) from exc

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        model_override: Optional[str] = None,
    ) -> Iterator[str]:
        """
        Stream chat token deltas from Ollama /api/chat.

        Delegates to :meth:`stream_chat_chunks` and yields text deltas only.
        """
        for ch in self.stream_chat_chunks(messages, model_override=model_override):
            if ch.content_delta:
                yield ch.content_delta

    @staticmethod
    def _extract_error_body(response: Optional[httpx.Response]) -> str:
        """Best-effort parse JSON error payload."""
        if response is None:
            return ""
        try:
            data = response.json()
            if isinstance(data, dict):
                return str(data.get("error") or data.get("message") or "")
        except (ValueError, TypeError):
            pass
        try:
            return response.text[:500]
        except Exception:
            return ""
