"""Unit tests for v3.5 config validation paths."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.config import AppConfig, validate_app_config_for_ui, validate_message_ingress_env


def _valid_cfg() -> AppConfig:
    """Build a valid base config and mutate one field per test."""
    return AppConfig(
        user_id="u1",
        session_id="",
        memory_rounds=3,
        chat_mode="messages",
        llm_backend="deepseek",
        ollama_base_url="http://localhost:11434",
        ollama_generate_model="qwen3:1.7b",
        ollama_request_timeout=600,
        ollama_embed_model="all-minilm:latest",
        deepseek_api_key="k",
        deepseek_llm_model="deepseek-chat",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_request_timeout=600,
        llm_transport="local",
        llm_service_url="",
        llm_service_timeout_seconds=120,
        llm_service_api_key="",
        chat_ui_ingress_path="/v1/messages/test",
        chat_ui_forward_url="http://127.0.0.1:8001/v1/chat/stream",
        chat_ui_forward_timeout_seconds=30,
        chat_ui_forward_api_key="",
    )


class ConfigV35ValidationTests(unittest.TestCase):
    """Ensure new ingress/forward config fields fail fast on invalid values."""

    def test_invalid_ui_ingress_path_raises(self) -> None:
        cfg = _valid_cfg()
        cfg = cfg.__class__(**{**cfg.__dict__, "chat_ui_ingress_path": "v1/messages/in"})
        with self.assertRaises(ValueError):
            validate_app_config_for_ui(cfg)

    def test_invalid_ui_forward_url_raises(self) -> None:
        cfg = _valid_cfg()
        cfg = cfg.__class__(**{**cfg.__dict__, "chat_ui_forward_url": "not-a-url"})
        with self.assertRaises(ValueError):
            validate_app_config_for_ui(cfg)

    def test_non_positive_forward_timeout_raises(self) -> None:
        cfg = _valid_cfg()
        cfg = cfg.__class__(**{**cfg.__dict__, "chat_ui_forward_timeout_seconds": 0})
        with self.assertRaises(ValueError):
            validate_app_config_for_ui(cfg)

    def test_env_validation_rejects_invalid_ui_forward_url(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "CHAT_UI_INGRESS_PATH": "/v1/messages/test",
                "CHAT_UI_FORWARD_URL": "bad-url",
            },
            clear=False,
        ):
            with self.assertRaises(RuntimeError):
                validate_message_ingress_env()

if __name__ == "__main__":
    unittest.main()

