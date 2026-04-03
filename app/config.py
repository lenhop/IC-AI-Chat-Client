"""
FastAPI runtime configuration loader.

Key points:
1) Read all backend settings from environment variables (.env supported).
2) Keep one normalized config snapshot in memory via ``get_config()``.
3) ``validate_standalone_env()`` runs after dotenv in ``app.main`` (fail fast).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import List


@dataclass(frozen=True)
class AppConfig:
    """Immutable app config used by routes and service wrappers."""

    user_id: str
    session_id: str
    llm_backend: str
    ollama_base_url: str
    ollama_generate_model: str
    ollama_request_timeout: int
    ollama_embed_model: str
    deepseek_api_key: str
    deepseek_llm_model: str
    deepseek_base_url: str
    deepseek_request_timeout: int


def _read_positive_int(name: str, default_value: int) -> int:
    """Read a positive integer from env; fallback to default on invalid value."""
    raw = (os.getenv(name) or str(default_value)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = default_value
    return value if value > 0 else default_value


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """
    Build an immutable config object from process environment.

    Returns:
        AppConfig: validated runtime settings for FastAPI + LLM adapters.
    """
    llm_backend = (os.getenv("LLM_BACKEND") or "deepseek").strip().lower()
    if llm_backend not in {"deepseek", "ollama"}:
        llm_backend = "deepseek"

    return AppConfig(
        user_id=(os.getenv("USER_ID") or "local-dev").strip(),
        session_id=(os.getenv("SESSION_ID") or "").strip(),
        llm_backend=llm_backend,
        ollama_base_url=(os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").strip().rstrip("/"),
        ollama_generate_model=(os.getenv("OLLAMA_GENERATE_MODEL") or "qwen3:1.7b").strip(),
        ollama_request_timeout=_read_positive_int("OLLAMA_REQUEST_TIMEOUT", 600),
        ollama_embed_model=(os.getenv("OLLAMA_EMBED_MODEL") or "all-minilm:latest").strip(),
        deepseek_api_key=(os.getenv("DEEPSEEK_API_KEY") or "").strip(),
        deepseek_llm_model=(os.getenv("DEEPSEEK_LLM_MODEL") or "deepseek-chat").strip(),
        deepseek_base_url=(os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").strip().rstrip("/"),
        deepseek_request_timeout=_read_positive_int("DEEPSEEK_REQUEST_TIMEOUT", 600),
    )


def validate_standalone_env() -> None:
    """
    Check required variables for LLM_BACKEND in os.environ (Standalone app only).

    Raises:
        RuntimeError: If any required variable is missing or blank.
    """
    backend = (os.getenv("LLM_BACKEND") or "deepseek").strip().lower()
    if backend not in {"deepseek", "ollama"}:
        backend = "deepseek"

    missing: List[str] = []

    if backend == "deepseek":
        if not (os.getenv("DEEPSEEK_API_KEY") or "").strip():
            missing.append("DEEPSEEK_API_KEY")
        _optional_positive_int("DEEPSEEK_REQUEST_TIMEOUT", missing)
    else:
        if not (os.getenv("OLLAMA_BASE_URL") or "").strip():
            missing.append("OLLAMA_BASE_URL")
        if not (os.getenv("OLLAMA_GENERATE_MODEL") or "").strip():
            missing.append("OLLAMA_GENERATE_MODEL")
        if not (os.getenv("OLLAMA_EMBED_MODEL") or "").strip():
            missing.append("OLLAMA_EMBED_MODEL")
        _optional_positive_int("OLLAMA_REQUEST_TIMEOUT", missing)

    if missing:
        raise RuntimeError(
            "Standalone .env is missing required variables for "
            f"LLM_BACKEND={backend!r}: {', '.join(missing)}"
        )

    _validate_gradio_ui_theme_env()


def _validate_gradio_ui_theme_env() -> None:
    """Reject invalid ``GRADIO_UI_THEME`` when explicitly set (optional env)."""
    raw = (os.getenv("GRADIO_UI_THEME") or "").strip()
    if not raw:
        return
    if raw.lower() not in {"business", "warm", "minimal"}:
        raise RuntimeError(
            "GRADIO_UI_THEME must be one of: business, warm, minimal "
            f"(got {raw!r})"
        )


def get_gradio_ui_theme() -> str:
    """
    Gradio client skin from env (default ``business``).

    Returns:
        One of ``business``, ``warm``, ``minimal``.
    """
    raw = (os.getenv("GRADIO_UI_THEME") or "business").strip().lower()
    if raw not in {"business", "warm", "minimal"}:
        return "business"
    return raw


def validate_app_config_for_ui(cfg: AppConfig) -> None:
    """
    Ensure ``AppConfig`` can support chat (LLM fields) for Gradio / display paths.

    Raises:
        ValueError: On missing or inconsistent fields.
    """
    backend = (cfg.llm_backend or "").strip().lower()
    if backend not in {"deepseek", "ollama"}:
        raise ValueError(f"AppConfig.llm_backend must be deepseek or ollama, got {cfg.llm_backend!r}")

    if backend == "deepseek":
        if not (cfg.deepseek_api_key or "").strip():
            raise ValueError("AppConfig: deepseek_api_key is required when llm_backend is deepseek")
        if cfg.deepseek_request_timeout <= 0:
            raise ValueError("AppConfig: deepseek_request_timeout must be positive")
    else:
        if not (cfg.ollama_base_url or "").strip():
            raise ValueError("AppConfig: ollama_base_url is required when llm_backend is ollama")
        if not (cfg.ollama_generate_model or "").strip():
            raise ValueError("AppConfig: ollama_generate_model is required when llm_backend is ollama")
        if not (cfg.ollama_embed_model or "").strip():
            raise ValueError("AppConfig: ollama_embed_model is required when llm_backend is ollama")
        if cfg.ollama_request_timeout <= 0:
            raise ValueError("AppConfig: ollama_request_timeout must be positive")


def _optional_positive_int(name: str, missing: List[str]) -> None:
    """If set, must be a positive integer; if unset, client code may use its own default."""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return
    try:
        value = int(raw)
    except ValueError:
        missing.append(f"{name} (not a positive integer)")
        return
    if value <= 0:
        missing.append(f"{name} (must be > 0)")
