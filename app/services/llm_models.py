"""
List deployable model names per backend (M2).

Ollama: ``GET /api/tags``. DeepSeek: no stable public list API in this client;
return the configured chat model id as a single-element list.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

import httpx

from app.config import AppConfig, get_config
from app.runtime_config import RuntimeConfig, validate_runtime_config

logger = logging.getLogger(__name__)


def list_ollama_model_names(base_url: str, *, timeout_seconds: float = 15.0) -> List[str]:
    """
    Fetch tag names from Ollama ``GET /api/tags``.

    Args:
        base_url: Ollama root URL without trailing slash.
        timeout_seconds: HTTP timeout budget.

    Returns:
        Sorted unique model names, or empty list on transport/parse errors (logged).
    """
    root = (base_url or "").strip().rstrip("/")
    if not root:
        return []
    url = f"{root}/api/tags"
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout_seconds, connect=10.0)) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()
    except (httpx.HTTPError, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("list_ollama_model_names failed: %s", exc)
        return []

    models = data.get("models") or []
    names: List[str] = []
    for item in models:
        if isinstance(item, dict):
            name = item.get("name")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
    return sorted(set(names))


def list_deepseek_configured_models(cfg: AppConfig) -> List[str]:
    """Return the configured DeepSeek chat model as the only selectable id."""
    model = (cfg.deepseek_llm_model or "").strip()
    return [model] if model else []


def list_deepseek_configured_models_runtime(cfg: RuntimeConfig) -> List[str]:
    """Return the runtime DeepSeek chat model as the only selectable id."""
    model = (cfg.deepseek_llm_model or "").strip()
    return [model] if model else []


def list_chat_model_names(
    *,
    backend: Optional[str] = None,
    runtime: Optional[RuntimeConfig] = None,
) -> List[str]:
    """
    List model names for the resolved backend (env or explicit ``runtime``).

    Args:
        backend: Optional ``deepseek`` / ``ollama`` label; ignored when ``runtime`` is set.
        runtime: If set, use this backend and config fields only.

    Returns:
        Non-empty list when possible; may be empty if Ollama is unreachable.
    """
    if runtime is not None:
        validate_runtime_config(runtime)
        if runtime.llm_backend == "deepseek":
            return list_deepseek_configured_models_runtime(runtime)
        return list_ollama_model_names(
            runtime.ollama_base_url,
            timeout_seconds=float(runtime.ollama_request_timeout),
        )

    app_cfg = get_config()
    text = (backend or "").strip().lower()
    if text in {"deepseek", "ds"}:
        return list_deepseek_configured_models(app_cfg)
    if text in {"ollama", "local"}:
        return list_ollama_model_names(
            app_cfg.ollama_base_url,
            timeout_seconds=float(app_cfg.ollama_request_timeout),
        )

    if app_cfg.llm_backend == "deepseek":
        return list_deepseek_configured_models(app_cfg)
    return list_ollama_model_names(
        app_cfg.ollama_base_url,
        timeout_seconds=float(app_cfg.ollama_request_timeout),
    )
