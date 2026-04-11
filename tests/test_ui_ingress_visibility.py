"""Integration test for ingress persistence and Gradio-readable history."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import fakeredis

from app.config import AppConfig, MessageDisplayOptions, RedisSettings
from app.memory.redis_runtime import bind_redis_for_gradio, clear_redis_for_gradio
from app.memory.session_store import SessionStore, gradio_history_from_stored
from app.messages.message_envelope import MessageEnvelope
from app.messages.message_ingress_service import MessageIngressService


def _visibility_cfg() -> AppConfig:
    """Build deterministic config for ingress -> storage -> UI visibility tests."""
    return AppConfig(
        user_id="u-visible",
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
        llm_transport="http",
        llm_service_url="http://127.0.0.1:8001",
        llm_service_timeout_seconds=120,
        llm_service_api_key="",
        chat_ui_ingress_path="/v1/messages/test",
        chat_ui_forward_url="http://127.0.0.1:8001/v1/chat/stream",
        chat_ui_forward_timeout_seconds=15,
        chat_ui_forward_api_key="",
    )


class UiIngressVisibilityTests(unittest.TestCase):
    """Prove ingress messages become Gradio-readable history rows."""

    def setUp(self) -> None:
        """Bind fake Redis runtime for deterministic storage assertions."""
        self.client = fakeredis.FakeRedis(decode_responses=True)
        self.redis_settings = RedisSettings(
            enabled=True,
            url="redis://fake/0",
            key_prefix="icai:",
            session_ttl_seconds=3600,
        )
        bind_redis_for_gradio(self.client, self.redis_settings)

    def tearDown(self) -> None:
        """Clean process-local Redis bindings after each test."""
        clear_redis_for_gradio()

    @patch("app.messages.message_ingress_service.get_config")
    @patch("app.messages.message_ingress_service.MessageIngressService._forward_message")
    def test_ingress_messages_are_visible_in_gradio_history(
        self,
        mock_forward,
        mock_get_config,
    ) -> None:
        """Verify ingress query/answer/clarification become UI-readable rows."""
        cfg = _visibility_cfg()
        mock_get_config.return_value = cfg
        mock_forward.return_value = {
            "payload": {
                "status": "ok",
                "downstream": {
                    "envelope": {
                        "message_id": "m-ans-visible",
                        "session_id": "s-visible",
                        "turn_id": "t-visible",
                        "type": "answer",
                        "content": "visible answer",
                        "source": "chat_llm_stream",
                        "target": "chat_ui",
                        "timestamp": "2026-01-01T00:00:01+00:00",
                        "metadata": {"reply_to_message_id": "m-query-visible"},
                    }
                },
            },
            "status_code": 200,
        }

        query = MessageEnvelope(
            message_id="m-query-visible",
            session_id="s-visible",
            turn_id="t-visible",
            type="query",
            content="visible question",
            source="chat_ui",
            target="chat_llm",
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={"scene": "visibility"},
        )
        clarification = MessageEnvelope(
            message_id="m-clarification-visible",
            session_id="s-visible",
            turn_id="t-visible",
            type="clarification",
            content="extra context",
            source="chat_ui",
            target="chat_llm",
            timestamp="2026-01-01T00:00:02+00:00",
            metadata={
                "scene": "visibility",
                "clarification_backend": "deepseek",
                "clarification_status": "Complete",
                "clarification_time_ms": "15",
            },
        )

        query_result = MessageIngressService.handle_ui_ingress(query)
        clarification_result = MessageIngressService.handle_ui_ingress(clarification)

        self.assertTrue(query_result.forwarded)
        self.assertFalse(clarification_result.forwarded)

        stored_messages = SessionStore(self.client, self.redis_settings).get_messages(
            "s-visible",
            cfg.user_id,
        )
        self.assertEqual([m["type"] for m in stored_messages], ["query", "answer", "clarification"])

        history_rows = gradio_history_from_stored(
            stored_messages,
            MessageDisplayOptions.all_enabled(),
        )
        self.assertEqual(history_rows[0]["role"], "user")
        self.assertEqual(history_rows[0]["content"], "visible question")
        self.assertIn("## Answer", history_rows[1]["content"])
        self.assertIn("visible answer", history_rows[1]["content"])
        self.assertIn("## Clarification", history_rows[2]["content"])
        self.assertIn("extra context", history_rows[2]["content"])
        self.assertIn("Clarification backend: deepseek", history_rows[2]["content"])
        self.assertIn("Clarification status: Complete", history_rows[2]["content"])
        self.assertIn("Clarification time: 15 ms", history_rows[2]["content"])


if __name__ == "__main__":
    unittest.main()
