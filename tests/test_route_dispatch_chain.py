"""Integration-style test for v3.5 scene2 round-trip via configured forward URL."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.config import AppConfig
from app.messages.message_envelope import MessageEnvelope
from app.messages.message_ingress_service import MessageIngressService


def _scene2_cfg() -> AppConfig:
    """Build scene2 config with a mock route entrypoint as UI forward URL."""
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
        chat_ui_forward_url="http://mock-route/v1/messages/test",
        chat_ui_forward_timeout_seconds=15,
        chat_ui_forward_api_key="",
    )


class _MockResponse:
    """Minimal response object compatible with message_ingress _forward_message."""

    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"mock status {self.status_code}")

    def json(self) -> dict:
        return self._payload


class _MockRouteDispatcherClient:
    """Mock httpx client that simulates route -> dispatcher -> llm -> dispatcher chain."""

    def __init__(self, *args, **kwargs):
        self.hops = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    @classmethod
    def _route_service(cls, env: dict) -> dict:
        """Scene2: route service forwards to dispatcher."""
        return cls._dispatcher_pre(env)

    @classmethod
    def _dispatcher_pre(cls, env: dict) -> dict:
        """Dispatcher pre-process step before LLM."""
        dispatch_in = {**env, "source": "dispatcher_pre", "target": "chat_llm"}
        llm_out = cls._chat_llm(dispatch_in)
        return cls._dispatcher_post(llm_out)

    @classmethod
    def _chat_llm(cls, env: dict) -> dict:
        """Mock chat LLM generates one answer with same trace IDs."""
        return {
            "message_id": "ans-from-llm",
            "session_id": env["session_id"],
            "turn_id": env["turn_id"],
            "type": "answer",
            "content": "mock answer from llm",
            "source": "chat_llm",
            "target": "dispatcher",
            "timestamp": env["timestamp"],
            "metadata": {"reply_to_message_id": env["message_id"]},
        }

    @classmethod
    def _dispatcher_post(cls, env: dict) -> dict:
        """Dispatcher post-process step returns answer to UI."""
        out = dict(env)
        out["source"] = "dispatcher_post"
        out["target"] = "chat_ui"
        out["metadata"] = {
            **(env.get("metadata") or {}),
            "chain": "route->dispatcher->llm->dispatcher",
        }
        return out

    def post(self, url: str, json: dict, headers: dict) -> _MockResponse:
        """Dispatch by URL to mock route service and return downstream envelope payload."""
        if url != "http://mock-route/v1/messages/test":
            return _MockResponse({"error": "unexpected route"}, status_code=404)
        final_env = self._route_service(json)
        payload = {
            "status": "ok",
            "downstream": {
                "envelope": final_env,
            },
        }
        return _MockResponse(payload, status_code=200)


class RouteDispatchChainTests(unittest.TestCase):
    """Verify scene2 full round-trip and answer persistence contract."""

    @patch("app.messages.message_ingress_service.get_config")
    @patch("app.messages.message_ingress_service.httpx.Client", new=_MockRouteDispatcherClient)
    def test_scene2_round_trip_answer_returned_and_persisted(
        self,
        mock_get_cfg,
    ) -> None:
        mock_get_cfg.return_value = _scene2_cfg()
        stored_rows = []

        def _store_capture(envelope: MessageEnvelope, cfg: AppConfig) -> bool:
            stored_rows.append(envelope)
            return True

        query = MessageEnvelope(
            message_id="m-query-1",
            session_id="s-chain-1",
            turn_id="t-chain-1",
            type="query",
            content="route dispatch test",
            source="chat_ui",
            target="route_llm",
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={"scene": "route_dispatch"},
        )

        with patch.object(MessageIngressService, "_store_envelope", side_effect=_store_capture):
            out = MessageIngressService.handle_ui_ingress(query)

        self.assertTrue(out.forwarded)
        self.assertEqual(out.status, "ok")
        self.assertEqual(len(stored_rows), 2, "query + downstream answer should be stored")

        # Trace continuity assertions across round-trip.
        answer_env = out.downstream["downstream"]["envelope"]
        self.assertEqual(answer_env["session_id"], query.session_id)
        self.assertEqual(answer_env["turn_id"], query.turn_id)
        self.assertEqual(answer_env["type"], "answer")
        self.assertEqual(answer_env["target"], "chat_ui")
        self.assertEqual(answer_env["metadata"]["reply_to_message_id"], query.message_id)


if __name__ == "__main__":
    unittest.main()
