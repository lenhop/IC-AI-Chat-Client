"""
Runtime LLM configuration for Python import integration (not from process .env).

Hosts construct RuntimeConfig and pass it to stream_chat(..., runtime=cfg).
Never log fields that contain secrets.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class RuntimeConfig(BaseModel):
    """
    Explicit LLM settings when IC-AI-Chat-Client is used as an imported library.

    Attributes:
        llm_backend: Which provider to use for this call chain.
        deepseek_*: Used when llm_backend is deepseek.
        ollama_*: Used when llm_backend is ollama.
    """

    llm_backend: Literal["deepseek", "ollama"] = Field(
        ...,
        description="Chat backend: deepseek or ollama",
    )
    deepseek_api_key: Optional[str] = Field(
        default=None,
        description="Required when llm_backend=deepseek",
    )
    deepseek_llm_model: str = Field(default="deepseek-chat")
    deepseek_base_url: str = Field(default="https://api.deepseek.com")
    deepseek_request_timeout: int = Field(default=600, ge=1)

    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_generate_model: str = Field(default="qwen3:1.7b")
    ollama_request_timeout: int = Field(default=600, ge=1)
    ollama_embed_model: str = Field(default="all-minilm:latest")

    @field_validator("deepseek_base_url", "ollama_base_url", mode="before")
    @classmethod
    def strip_trailing_slash(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip().rstrip("/")
        return v


def validate_runtime_config(cfg: RuntimeConfig) -> None:
    """
    Ensure all required fields are present for the selected backend.

    Args:
        cfg: Parsed runtime configuration.

    Raises:
        ValueError: With a list of missing or invalid field names.
    """
    missing: list[str] = []

    if cfg.llm_backend == "deepseek":
        if not (cfg.deepseek_api_key or "").strip():
            missing.append("deepseek_api_key")
        if not (cfg.deepseek_llm_model or "").strip():
            missing.append("deepseek_llm_model")
        if not (cfg.deepseek_base_url or "").strip():
            missing.append("deepseek_base_url")
    else:
        if not (cfg.ollama_base_url or "").strip():
            missing.append("ollama_base_url")
        if not (cfg.ollama_generate_model or "").strip():
            missing.append("ollama_generate_model")
        if not (cfg.ollama_embed_model or "").strip():
            missing.append("ollama_embed_model")

    if missing:
        raise ValueError(
            f"RuntimeConfig invalid for backend={cfg.llm_backend!r}; "
            f"missing or empty: {', '.join(missing)}"
        )
