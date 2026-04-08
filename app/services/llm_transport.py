"""
Unified streaming entry for UI and SSE: local ``call_llm`` vs HTTP LLM microservice.

When ``LLM_TRANSPORT=http``, deltas are read from ``POST {LLM_SERVICE_URL}/v1/chat/stream``
(SSE). ``runtime=`` injection always uses the in-process backend (library embed mode).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterator, List, Optional

import httpx

from app.config import AppConfig, get_config
from app.runtime_config import RuntimeConfig
from app.services.call_llm import stream_chat

logger = logging.getLogger(__name__)


def _iter_http_stream_deltas(
    cfg: AppConfig,
    messages: List[Dict[str, str]],
    *,
    backend: Optional[str],
    model_override: Optional[str],
) -> Iterator[str]:
    """
    Stream assistant text chunks from the remote LLM service (SSE ``data:`` JSON lines).

    Args:
        cfg: Resolved app config (``llm_service_url``, timeout, optional API key).
        messages: OpenAI-style role/content dicts (already normalized upstream).
        backend: Optional backend override for the worker.
        model_override: Optional model id for the worker.

    Yields:
        Text deltas from frames that contain a ``delta`` field.

    Raises:
        RuntimeError: On HTTP errors, non-2xx responses, or ``error`` SSE frames.
    """
    base = (cfg.llm_service_url or "").strip().rstrip("/")
    if not base:
        raise RuntimeError("llm_service_url is empty (LLM_SERVICE_URL)")

    url = f"{base}/v1/chat/stream"
    headers: Dict[str, str] = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    api_key = (cfg.llm_service_api_key or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: Dict[str, Any] = {
        "messages": messages,
        "backend": backend,
        "model": model_override,
    }

    read_s = float(cfg.llm_service_timeout_seconds)
    if read_s <= 0:
        read_s = 120.0
    client_timeout = httpx.Timeout(connect=30.0, read=read_s, write=30.0, pool=30.0)

    try:
        with httpx.Client(timeout=client_timeout) as client:
            with client.stream(
                "POST",
                url,
                json=payload,
                headers=headers,
                timeout=client_timeout,
            ) as resp:
                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise RuntimeError(
                        f"LLM service HTTP {resp.status_code}: {resp.text[:500]}"
                    ) from exc

                for line in resp.iter_lines():
                    if not line:
                        continue
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:].strip()
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        logger.debug("llm_transport: skip non-JSON SSE line")
                        continue
                    if isinstance(obj, dict) and obj.get("error") is not None:
                        raise RuntimeError(str(obj.get("error")))
                    if isinstance(obj, dict) and "delta" in obj:
                        d = obj.get("delta")
                        if isinstance(d, str) and d:
                            yield d
    except httpx.HTTPError as exc:
        raise RuntimeError(f"LLM HTTP transport request failed: {exc}") from exc


def iter_chat_text_deltas(
    messages: List[Dict[str, str]],
    *,
    backend: Optional[str] = None,
    model_override: Optional[str] = None,
    runtime: Optional[RuntimeConfig] = None,
) -> Iterator[str]:
    """
    Yield assistant-visible text deltas (same contract as :func:`stream_chat`).

    Args:
        messages: Chat turns in OpenAI shape.
        backend: Optional backend hint (ignored when ``runtime`` is set).
        model_override: Optional model id (ignored when ``runtime`` is set).
        runtime: When set, always use in-process streaming with this config.

    Yields:
        Concatenable assistant text fragments.

    Raises:
        ValueError: When message normalization fails (local path only).
        RuntimeError: When HTTP transport or upstream LLM fails.
    """
    if runtime is not None:
        yield from stream_chat(
            messages,
            backend=backend,
            model_override=model_override,
            runtime=runtime,
        )
        return

    cfg = get_config()
    transport = (cfg.llm_transport or "local").strip().lower()
    if transport == "http":
        yield from _iter_http_stream_deltas(
            cfg,
            messages,
            backend=backend,
            model_override=model_override,
        )
        return

    yield from stream_chat(
        messages,
        backend=backend,
        model_override=model_override,
    )
