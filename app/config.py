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
from urllib.parse import urlparse
from typing import List


@dataclass(frozen=True)
class RedisSettings:
    """Redis options for M3 session storage (browser never connects directly)."""

    enabled: bool
    url: str
    key_prefix: str
    session_ttl_seconds: int


@dataclass(frozen=True)
class AppConfig:
    """Immutable app config used by routes and service wrappers."""

    user_id: str
    session_id: str
    # Last N conversation rounds for Gradio hydrate / prompt history; 0 = unlimited.
    memory_rounds: int
    # messages: OpenAI-style multi-turn; prompt_template: chat_prompt.md + single user blob (needs Redis).
    chat_mode: str
    llm_backend: str
    ollama_base_url: str
    ollama_generate_model: str
    ollama_request_timeout: int
    ollama_embed_model: str
    deepseek_api_key: str
    deepseek_llm_model: str
    deepseek_base_url: str
    deepseek_request_timeout: int
    # Gradio: hide optional message types (query/answer always shown).
    clarification_message_display_enable: bool = True
    rewriting_message_display_enable: bool = True
    classification_message_display_enable: bool = True
    plan_message_display_enable: bool = True
    reason_message_display_enable: bool = True
    context_message_display_enable: bool = True
    dispatcher_message_display_enable: bool = True
    # UI / SSE: local process calls DeepSeek/Ollama; http = delegate to LLM microservice.
    llm_transport: str = "local"
    llm_service_url: str = ""
    llm_service_timeout_seconds: int = 120
    llm_service_api_key: str = ""
    # v3.5 message ingress/forward settings (UI side).
    chat_ui_ingress_path: str = "/v1/messages/test"
    chat_ui_forward_url: str = "http://127.0.0.1:8001/v1/chat/stream"
    chat_ui_forward_timeout_seconds: int = 30
    chat_ui_forward_api_key: str = ""


@dataclass(frozen=True)
class MessageDisplayOptions:
    """Subset of AppConfig used when building Gradio history rows (avoids coupling to full config)."""

    clarification_message_display_enable: bool
    rewriting_message_display_enable: bool
    classification_message_display_enable: bool
    plan_message_display_enable: bool
    reason_message_display_enable: bool
    context_message_display_enable: bool
    dispatcher_message_display_enable: bool

    @classmethod
    def from_app_config(cls, cfg: AppConfig) -> MessageDisplayOptions:
        """Build options from a resolved ``AppConfig`` snapshot."""
        return cls(
            clarification_message_display_enable=cfg.clarification_message_display_enable,
            rewriting_message_display_enable=cfg.rewriting_message_display_enable,
            classification_message_display_enable=cfg.classification_message_display_enable,
            plan_message_display_enable=cfg.plan_message_display_enable,
            reason_message_display_enable=cfg.reason_message_display_enable,
            context_message_display_enable=cfg.context_message_display_enable,
            dispatcher_message_display_enable=cfg.dispatcher_message_display_enable,
        )

    @classmethod
    def all_enabled(cls) -> MessageDisplayOptions:
        """Default when no config is passed (e.g. tests): show every optional type."""
        return cls(True, True, True, True, True, True, True)


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse common truthy/falsey env strings."""
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


@lru_cache(maxsize=1)
def get_redis_settings() -> RedisSettings:
    """
    Build Redis settings from environment (after dotenv load in standalone main).

    Returns:
        RedisSettings: enabled flag, URL, key prefix, and session TTL seconds.
    """
    enabled = _env_bool("REDIS_ENABLED", default=False)
    url = (os.getenv("REDIS_URL") or "").strip()
    prefix = (os.getenv("REDIS_KEY_PREFIX") or "icai:").strip()
    if not prefix.endswith(":"):
        prefix = prefix + ":"
    ttl = _read_positive_int("REDIS_SESSION_TTL_SECONDS", 2592000)
    return RedisSettings(
        enabled=enabled,
        url=url,
        key_prefix=prefix,
        session_ttl_seconds=ttl,
    )


def _read_positive_int(name: str, default_value: int) -> int:
    """Read a positive integer from env; fallback to default on invalid value."""
    raw = (os.getenv(name) or str(default_value)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = default_value
    return value if value > 0 else default_value


def _read_non_negative_int(name: str, default_value: int) -> int:
    """Read int >= 0 from env; invalid or negative falls back to default."""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default_value
    try:
        value = int(raw)
    except ValueError:
        return default_value
    return value if value >= 0 else default_value


def _parse_chat_mode() -> str:
    """CHAT_MODE: messages | prompt_template (invalid -> messages)."""
    raw = (os.getenv("CHAT_MODE") or "messages").strip().lower()
    if raw in {"messages", "prompt_template"}:
        return raw
    return "messages"


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

    lt_raw = (os.getenv("LLM_TRANSPORT") or "local").strip().lower()
    llm_transport = lt_raw if lt_raw in {"local", "http"} else "local"

    return AppConfig(
        user_id=(os.getenv("USER_ID") or "local-dev").strip(),
        session_id=(os.getenv("SESSION_ID") or "").strip(),
        memory_rounds=_read_non_negative_int("MEMORY_ROUNDS", 3),
        chat_mode=_parse_chat_mode(),
        llm_backend=llm_backend,
        ollama_base_url=(os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").strip().rstrip("/"),
        ollama_generate_model=(os.getenv("OLLAMA_GENERATE_MODEL") or "qwen3:1.7b").strip(),
        ollama_request_timeout=_read_positive_int("OLLAMA_REQUEST_TIMEOUT", 600),
        ollama_embed_model=(os.getenv("OLLAMA_EMBED_MODEL") or "all-minilm:latest").strip(),
        deepseek_api_key=(os.getenv("DEEPSEEK_API_KEY") or "").strip(),
        deepseek_llm_model=(os.getenv("DEEPSEEK_LLM_MODEL") or "deepseek-chat").strip(),
        deepseek_base_url=(os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").strip().rstrip("/"),
        deepseek_request_timeout=_read_positive_int("DEEPSEEK_REQUEST_TIMEOUT", 600),
        clarification_message_display_enable=_env_bool(
            "CLARIFICATION_MESSAGE_DISPLAY_ENABLE", default=True
        ),
        rewriting_message_display_enable=_env_bool(
            "REWRITING_MESSAGE_DISPLAY_ENABLE", default=True
        ),
        classification_message_display_enable=_env_bool(
            "CLASSIFICATION_MESSAGE_DISPLAY_ENABLE", default=True
        ),
        plan_message_display_enable=_env_bool("PLAN_MESSAGE_DISPLAY_ENABLE", default=True),
        reason_message_display_enable=_env_bool("REASON_MESSAGE_DISPLAY_ENABLE", default=True),
        context_message_display_enable=_env_bool("CONTEXT_MESSAGE_DISPLAY_ENABLE", default=True),
        dispatcher_message_display_enable=_env_bool(
            "DISPATCHER_MESSAGE_DISPLAY_ENABLE", default=True
        ),
        llm_transport=llm_transport,
        llm_service_url=(os.getenv("LLM_SERVICE_URL") or "").strip().rstrip("/"),
        llm_service_timeout_seconds=_read_positive_int("LLM_SERVICE_TIMEOUT_SECONDS", 120),
        llm_service_api_key=(os.getenv("LLM_SERVICE_API_KEY") or "").strip(),
        chat_ui_ingress_path=(os.getenv("CHAT_UI_INGRESS_PATH") or "/v1/messages/test").strip(),
        chat_ui_forward_url=(
            os.getenv("CHAT_UI_FORWARD_URL") or "http://127.0.0.1:8001/v1/chat/stream"
        ).strip().rstrip("/"),
        chat_ui_forward_timeout_seconds=_read_positive_int("CHAT_UI_FORWARD_TIMEOUT_SECONDS", 30),
        chat_ui_forward_api_key=(os.getenv("CHAT_UI_FORWARD_API_KEY") or "").strip(),
    )


def validate_llm_worker_env() -> None:
    """
    Validate env for the standalone LLM microservice (keys / Ollama only).

    Raises:
        RuntimeError: When LLM_BACKEND credentials are missing or invalid.
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
            "LLM worker .env is missing required variables for "
            f"LLM_BACKEND={backend!r}: {', '.join(missing)}"
        )


def validate_standalone_env() -> None:
    """
    Check required variables for standalone UI + optional HTTP LLM transport.

    Raises:
        RuntimeError: If any required variable is missing or blank.
    """
    transport_raw = (os.getenv("LLM_TRANSPORT") or "").strip()
    if transport_raw and transport_raw.lower() not in {"local", "http"}:
        raise RuntimeError(
            "LLM_TRANSPORT must be local or http "
            f"(got {transport_raw!r})"
        )
    transport = (transport_raw or "local").strip().lower()
    if transport not in {"local", "http"}:
        transport = "local"

    if transport == "http":
        if not (os.getenv("LLM_SERVICE_URL") or "").strip():
            raise RuntimeError(
                "LLM_TRANSPORT=http requires a non-empty LLM_SERVICE_URL."
            )
        timeout_missing: List[str] = []
        _optional_positive_int("LLM_SERVICE_TIMEOUT_SECONDS", timeout_missing)
        if timeout_missing:
            raise RuntimeError(
                "LLM_SERVICE_TIMEOUT_SECONDS must be a positive integer when set."
            )
        _validate_redis_env()
        _validate_chat_mode_env()
        _validate_gradio_ui_theme_env()
        _validate_chat_message_ingress_env()
        return

    validate_llm_worker_env()

    _validate_redis_env()
    _validate_chat_mode_env()
    _validate_gradio_ui_theme_env()
    _validate_chat_message_ingress_env()


def _validate_redis_env() -> None:
    """Require REDIS_URL when REDIS_ENABLED is true."""
    rs = get_redis_settings()
    if not rs.enabled:
        return
    if not rs.url:
        raise RuntimeError(
            "REDIS_ENABLED is true but REDIS_URL is missing or empty."
        )


def _validate_chat_mode_env() -> None:
    """prompt_template mode requires Redis."""
    if _parse_chat_mode() != "prompt_template":
        return
    if not get_redis_settings().enabled:
        raise RuntimeError(
            "CHAT_MODE=prompt_template requires REDIS_ENABLED=true and a working Redis URL."
        )


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


def _validate_path_env(name: str, default_value: str) -> None:
    """Validate ingress path-like env; must start with ``/``."""
    raw = (os.getenv(name) or default_value).strip()
    if not raw.startswith("/"):
        raise RuntimeError(f"{name} must start with '/', got {raw!r}")


def _is_valid_http_url(url: str) -> bool:
    """Return whether a URL uses http/https and has a non-empty netloc."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    return bool(parsed.netloc.strip())


def _validate_http_url_env(name: str, default_value: str, *, allow_empty: bool = False) -> None:
    """
    Validate URL-like env values with explicit diagnostics.

    Rules:
    - when ``allow_empty=False``: value must be non-empty and valid http/https URL.
    - when ``allow_empty=True``: empty is allowed; non-empty value must still be valid.
    """
    raw = (os.getenv(name) or default_value).strip()
    if not raw:
        if allow_empty:
            return
        raise RuntimeError(f"{name} must be non-empty.")
    if not _is_valid_http_url(raw):
        raise RuntimeError(
            f"{name} must be a valid http/https URL with host "
            f"(got {raw!r})"
        )


def _validate_chat_message_ingress_env() -> None:
    """Validate v3.5 UI ingress and forwarding env values."""
    _validate_path_env("CHAT_UI_INGRESS_PATH", "/v1/messages/test")

    timeout_missing: List[str] = []
    _optional_positive_int("CHAT_UI_FORWARD_TIMEOUT_SECONDS", timeout_missing)
    if timeout_missing:
        raise RuntimeError("; ".join(timeout_missing))

    _validate_http_url_env(
        "CHAT_UI_FORWARD_URL",
        "http://127.0.0.1:8001/v1/chat/stream",
        allow_empty=False
    )


def validate_message_ingress_env() -> None:
    """
    Public wrapper for v3.5 ingress/forward env validation.

    Compatibility note:
        Kept as a stable public entrypoint for tests and external scripts that
        validate only ingress-related env variables.
    """
    _validate_chat_message_ingress_env()


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
    if cfg.memory_rounds < 0:
        raise ValueError("AppConfig: memory_rounds must be >= 0")
    cm = (cfg.chat_mode or "").strip().lower()
    if cm not in {"messages", "prompt_template"}:
        raise ValueError("AppConfig: chat_mode must be messages or prompt_template")

    lt = (cfg.llm_transport or "local").strip().lower()
    if lt == "http":
        if not (cfg.llm_service_url or "").strip():
            raise ValueError("AppConfig: llm_service_url is required when llm_transport is http")
        if cfg.llm_service_timeout_seconds <= 0:
            raise ValueError("AppConfig: llm_service_timeout_seconds must be positive")
    elif lt != "local":
        raise ValueError("AppConfig: llm_transport must be local or http")
    else:
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

    if not (cfg.chat_ui_ingress_path or "").startswith("/"):
        raise ValueError("AppConfig: chat_ui_ingress_path must start with '/'")
    ui_forward_url = (cfg.chat_ui_forward_url or "").strip()
    if not ui_forward_url:
        raise ValueError("AppConfig: chat_ui_forward_url must be non-empty")
    if not _is_valid_http_url(ui_forward_url):
        raise ValueError("AppConfig: chat_ui_forward_url must be a valid http/https URL")
    if cfg.chat_ui_forward_timeout_seconds <= 0:
        raise ValueError("AppConfig: chat_ui_forward_timeout_seconds must be positive")


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
