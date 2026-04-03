"""Unit tests for M2 streaming chunk helpers (no live LLM calls)."""

from __future__ import annotations

import unittest

from app.config import AppConfig
from app.services.llm_chunks import ChatStreamChunk, iter_text_deltas
from app.services.llm_models import list_deepseek_configured_models


class TestIterTextDeltas(unittest.TestCase):
    """``iter_text_deltas`` must mirror legacy ``stream_chat`` text-only semantics."""

    def test_yields_only_content_delta(self) -> None:
        def gen():
            yield ChatStreamChunk(content_delta="hel")
            yield ChatStreamChunk(content_delta="lo")
            yield ChatStreamChunk(reasoning_delta="think")
            yield ChatStreamChunk(done=True)

        self.assertEqual(list(iter_text_deltas(gen())), ["hel", "lo"])

    def test_skips_empty_content(self) -> None:
        def gen():
            yield ChatStreamChunk(content_delta="")
            yield ChatStreamChunk(content_delta="x")

        self.assertEqual(list(iter_text_deltas(gen())), ["x"])


class TestListDeepseekModels(unittest.TestCase):
    """DeepSeek has no list API in this client; expose configured model id only."""

    def test_returns_single_configured_name(self) -> None:
        cfg = AppConfig(
            user_id="u",
            session_id="",
            llm_backend="deepseek",
            ollama_base_url="http://localhost:11434",
            ollama_generate_model="m",
            ollama_request_timeout=600,
            ollama_embed_model="e",
            deepseek_api_key="k",
            deepseek_llm_model="deepseek-chat",
            deepseek_base_url="https://api.deepseek.com",
            deepseek_request_timeout=600,
        )
        self.assertEqual(list_deepseek_configured_models(cfg), ["deepseek-chat"])

    def test_empty_when_model_blank(self) -> None:
        cfg = AppConfig(
            user_id="u",
            session_id="",
            llm_backend="deepseek",
            ollama_base_url="http://localhost:11434",
            ollama_generate_model="m",
            ollama_request_timeout=600,
            ollama_embed_model="e",
            deepseek_api_key="k",
            deepseek_llm_model="",
            deepseek_base_url="https://api.deepseek.com",
            deepseek_request_timeout=600,
        )
        self.assertEqual(list_deepseek_configured_models(cfg), [])


if __name__ == "__main__":
    unittest.main()
