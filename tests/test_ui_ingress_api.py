"""Route-level tests for UI message ingress query/non-query behavior."""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.config import AppConfig
from app.main import app


def _cfg() -> AppConfig:
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
        llm_transport="http",
        llm_service_url="http://127.0.0.1:8001",
        llm_service_timeout_seconds=120,
        llm_service_api_key="",
        chat_ui_ingress_path="/v1/messages/test",
        chat_ui_forward_url="http://127.0.0.1:8001/v1/chat/stream",
        chat_ui_forward_timeout_seconds=15,
        chat_ui_forward_api_key="",
    )


class UiIngressApiTests(unittest.TestCase):
    """Verify UI ingress keeps query-only forwarding contract."""

    @patch("app.services.message_ingress.get_config")
    @patch("app.services.message_ingress.MessageIngressService._store_envelope")
    @patch("app.services.message_ingress.MessageIngressService._forward_message")
    def test_non_query_request_not_forwarded(
        self,
        mock_forward: MagicMock,
        mock_store: MagicMock,
        mock_cfg: MagicMock,
    ) -> None:
        mock_cfg.return_value = _cfg()
        mock_store.return_value = True
        body = {
            "message_id": "m-plan",
            "session_id": "s1",
            "turn_id": "t1",
            "type": "plan",
            "content": "do steps",
            "source": "chat_ui",
            "target": "chat_llm",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "metadata": {},
        }
        with TestClient(app) as client:
            resp = client.post("/v1/messages/test", json=body)
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(resp.json()["forwarded"])
        mock_forward.assert_not_called()

    @patch("app.services.message_ingress.get_config")
    @patch("app.services.message_ingress.MessageIngressService._store_envelope")
    @patch("app.services.message_ingress.MessageIngressService._forward_message")
    def test_query_request_forwarded(
        self,
        mock_forward: MagicMock,
        mock_store: MagicMock,
        mock_cfg: MagicMock,
    ) -> None:
        mock_cfg.return_value = _cfg()
        mock_store.return_value = True
        mock_forward.return_value = {
            "payload": {
                "status": "ok",
                "downstream": {
                    "envelope": {
                        "message_id": "m-ans",
                        "session_id": "s1",
                        "turn_id": "t1",
                        "type": "answer",
                        "content": "ok",
                        "source": "chat_llm_stream",
                        "target": "chat_ui",
                        "timestamp": "2026-01-01T00:00:01+00:00",
                        "metadata": {"reply_to_message_id": "m-query"},
                    }
                }
            },
            "status_code": 200,
        }
        body = {
            "message_id": "m-query",
            "session_id": "s1",
            "turn_id": "t1",
            "type": "query",
            "content": "hello",
            "source": "chat_ui",
            "target": "chat_llm",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "metadata": {},
        }
        with TestClient(app) as client:
            resp = client.post("/v1/messages/test", json=body)
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["forwarded"])
        down_env = resp.json()["downstream"]["downstream"]["envelope"]
        self.assertEqual(down_env["session_id"], "s1")
        self.assertEqual(down_env["turn_id"], "t1")
        self.assertEqual(down_env["metadata"]["reply_to_message_id"], "m-query")
        mock_forward.assert_called_once()

    @patch("app.services.message_ingress.get_config")
    @patch("app.services.message_ingress.MessageIngressService._store_envelope")
    @patch("app.services.message_ingress.MessageIngressService._forward_message")
    def test_query_uses_configured_forward_url(
        self,
        mock_forward: MagicMock,
        mock_store: MagicMock,
        mock_cfg: MagicMock,
    ) -> None:
        cfg = _cfg()
        cfg = cfg.__class__(
            **{
                **cfg.__dict__,
                "chat_ui_forward_url": "http://127.0.0.1:8201/v1/messages/test",
            }
        )
        mock_cfg.return_value = cfg
        mock_store.return_value = True
        mock_forward.return_value = {"payload": {"status": "ok"}, "status_code": 200}
        body = {
            "message_id": "m-route",
            "session_id": "s1",
            "turn_id": "t1",
            "type": "query",
            "content": "hello",
            "source": "chat_ui",
            "target": "chat_llm",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "metadata": {},
        }
        with TestClient(app) as client:
            resp = client.post("/v1/messages/test", json=body)
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["forwarded"])
        self.assertEqual(
            mock_forward.call_args.kwargs["url"],
            "http://127.0.0.1:8201/v1/messages/test",
        )

    @patch("app.services.message_ingress.get_config")
    @patch("app.services.message_ingress.MessageIngressService._store_envelope")
    @patch("app.services.message_ingress.MessageIngressService._forward_message")
    def test_query_forward_error_maps_to_502(
        self,
        mock_forward: MagicMock,
        mock_store: MagicMock,
        mock_cfg: MagicMock,
    ) -> None:
        """Forwarding failure should map to HTTP 502 with explicit reason."""
        mock_cfg.return_value = _cfg()
        mock_store.return_value = True
        mock_forward.side_effect = RuntimeError("downstream timeout")
        body = {
            "message_id": "m-query-fail",
            "session_id": "s1",
            "turn_id": "t1",
            "type": "query",
            "content": "hello",
            "source": "chat_ui",
            "target": "chat_llm",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "metadata": {},
        }
        with TestClient(app) as client:
            resp = client.post("/v1/messages/test", json=body)
        self.assertEqual(resp.status_code, 502)
        self.assertIn("Failed to forward query message", resp.json()["detail"])

    @patch("app.services.message_ingress.get_config")
    @patch("app.services.message_ingress.MessageIngressService._store_envelope")
    @patch("app.services.message_ingress.MessageIngressService._forward_message")
    def test_legacy_alias_paths_still_available(
        self,
        mock_forward: MagicMock,
        mock_store: MagicMock,
        mock_cfg: MagicMock,
    ) -> None:
        """Legacy ingress aliases remain functional for backward compatibility."""
        mock_cfg.return_value = _cfg()
        mock_store.return_value = True
        mock_forward.return_value = {"payload": {"status": "ok"}, "status_code": 200}
        body = {
            "message_id": "m-legacy",
            "session_id": "s1",
            "turn_id": "t1",
            "type": "query",
            "content": "hello",
            "source": "chat_ui",
            "target": "chat_llm",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "metadata": {},
        }
        with TestClient(app) as client:
            resp_in = client.post("/v1/messages/in", json=body)
            resp_receive = client.post("/v1/messages/receive", json=body)
        self.assertEqual(resp_in.status_code, 200)
        self.assertEqual(resp_receive.status_code, 200)

    @patch("app.services.message_ingress.get_config")
    @patch("app.services.message_ingress.MessageIngressService._store_envelope")
    @patch("app.services.message_ingress.MessageIngressService._forward_message")
    def test_custom_env_ingress_path_route_is_reachable(
        self,
        mock_forward: MagicMock,
        mock_store: MagicMock,
        mock_cfg: MagicMock,
    ) -> None:
        """
        Validate custom ingress route reachability when env sets /v1/messages/custom.

        This test is intended for explicit re-validation command:
        CHAT_UI_INGRESS_PATH=/v1/messages/custom python -m unittest tests.test_ui_ingress_api -v
        """
        custom_path = (os.getenv("CHAT_UI_INGRESS_PATH") or "").strip()
        if custom_path != "/v1/messages/custom":
            self.skipTest("custom route test requires CHAT_UI_INGRESS_PATH=/v1/messages/custom")

        cfg = _cfg()
        cfg = cfg.__class__(**{**cfg.__dict__, "chat_ui_ingress_path": custom_path})
        mock_cfg.return_value = cfg
        mock_store.return_value = True
        mock_forward.return_value = {"payload": {"status": "ok"}, "status_code": 200}
        query_body = {
            "message_id": "m-custom-query",
            "session_id": "s1",
            "turn_id": "t1",
            "type": "query",
            "content": "hello",
            "source": "chat_ui",
            "target": "chat_llm",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "metadata": {},
        }
        non_query_body = {
            "message_id": "m-custom-plan",
            "session_id": "s1",
            "turn_id": "t2",
            "type": "plan",
            "content": "step-by-step",
            "source": "chat_ui",
            "target": "chat_llm",
            "timestamp": "2026-01-01T00:00:01+00:00",
            "metadata": {},
        }
        with TestClient(app) as client:
            custom_query_resp = client.post(custom_path, json=query_body)
            custom_non_query_resp = client.post(custom_path, json=non_query_body)
            canonical_resp = client.post("/v1/messages/test", json=query_body)
            legacy_in_resp = client.post("/v1/messages/in", json=query_body)
            legacy_receive_resp = client.post("/v1/messages/receive", json=query_body)

        self.assertEqual(custom_query_resp.status_code, 200)
        self.assertTrue(custom_query_resp.json()["forwarded"])
        self.assertEqual(custom_non_query_resp.status_code, 200)
        self.assertFalse(custom_non_query_resp.json()["forwarded"])
        self.assertEqual(canonical_resp.status_code, 200)
        self.assertEqual(legacy_in_resp.status_code, 200)
        self.assertEqual(legacy_receive_resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()

