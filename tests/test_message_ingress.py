"""Unit tests for v3.5 message ingress rules and envelope routing."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from app.config import AppConfig
from app.messages.message_envelope import MessageEnvelope
from app.services.message_ingress import MessageIngressService


def _base_cfg() -> AppConfig:
    """Build a deterministic config snapshot for ingress unit tests."""
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


def _envelope(mtype: str = "query") -> MessageEnvelope:
    """Create a valid message envelope for ingress branch tests."""
    return MessageEnvelope(
        message_id="m-1",
        session_id="s-1",
        turn_id="t-1",
        type=mtype,
        content="hello",
        source="chat_ui",
        target="chat_llm",
        timestamp="2026-01-01T00:00:00+00:00",
        metadata={"scene": "test"},
    )


class _MockSseStreamResponse:
    """Minimal streaming response object for SSE forwarding tests."""

    def __init__(self, lines: list[str], status_code: int = 200):
        self._lines = lines
        self.status_code = status_code
        self.text = "\n".join(lines)

    def raise_for_status(self) -> None:
        """Raise runtime error for HTTP status simulation."""
        if self.status_code >= 400:
            raise RuntimeError(f"mock status {self.status_code}")

    def iter_lines(self):
        """Iterate pre-defined SSE lines."""
        for line in self._lines:
            yield line

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _MockSseHttpClient:
    """Context-managed httpx.Client replacement for SSE tests."""

    response_lines: list[str] = []

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def stream(self, method: str, url: str, json: dict, headers: dict) -> _MockSseStreamResponse:
        """Return one deterministic streaming response."""
        return _MockSseStreamResponse(self.response_lines, status_code=200)


class MessageIngressTests(unittest.TestCase):
    """Cover query/non-query branches and downstream forwarding behavior."""

    @patch("app.services.message_ingress.get_config")
    @patch("app.services.message_ingress.MessageIngressService._store_envelope")
    @patch("app.services.message_ingress.MessageIngressService._forward_message")
    def test_ui_non_query_is_not_forwarded(
        self,
        mock_forward: MagicMock,
        mock_store: MagicMock,
        mock_get_cfg: MagicMock,
    ) -> None:
        mock_get_cfg.return_value = _base_cfg()
        mock_store.return_value = True

        out = MessageIngressService.handle_ui_ingress(_envelope("plan"))
        self.assertFalse(out.forwarded)
        self.assertTrue(out.stored)
        self.assertEqual(out.type, "plan")
        mock_forward.assert_not_called()

    @patch("app.services.message_ingress.get_config")
    @patch("app.services.message_ingress.MessageIngressService._store_envelope")
    @patch("app.services.message_ingress.MessageIngressService._forward_message")
    def test_ui_query_is_forwarded(
        self,
        mock_forward: MagicMock,
        mock_store: MagicMock,
        mock_get_cfg: MagicMock,
    ) -> None:
        mock_get_cfg.return_value = _base_cfg()
        mock_store.return_value = True
        mock_forward.return_value = {
            "payload": {
                "status": "ok",
                "downstream": {
                    "envelope": {
                        "message_id": "m-2",
                        "session_id": "s-1",
                        "turn_id": "t-1",
                        "type": "answer",
                        "content": "world",
                        "source": "chat_llm_stream",
                        "target": "chat_ui",
                        "timestamp": "2026-01-01T00:00:01+00:00",
                        "metadata": {"reply_to_message_id": "m-1"},
                    }
                }
            },
            "status_code": 200,
        }

        out = MessageIngressService.handle_ui_ingress(_envelope("query"))
        self.assertTrue(out.forwarded)
        self.assertTrue(out.stored)
        self.assertEqual(out.type, "query")
        self.assertEqual(mock_store.call_count, 2)
        mock_forward.assert_called_once()
        answer_env = out.downstream["downstream"]["envelope"]
        self.assertEqual(answer_env["session_id"], "s-1")
        self.assertEqual(answer_env["turn_id"], "t-1")
        self.assertEqual(answer_env["metadata"]["reply_to_message_id"], "m-1")

    @patch("app.services.message_ingress.get_config")
    @patch("app.services.message_ingress.MessageIngressService._store_envelope")
    @patch("app.services.message_ingress.MessageIngressService._forward_message")
    def test_ui_uses_configured_forward_target(
        self,
        mock_forward: MagicMock,
        mock_store: MagicMock,
        mock_get_cfg: MagicMock,
    ) -> None:
        cfg = _base_cfg()
        cfg = cfg.__class__(
            **{
                **cfg.__dict__,
                "chat_ui_forward_url": "http://127.0.0.1:8201/v1/messages/test",
            }
        )
        mock_get_cfg.return_value = cfg
        mock_store.return_value = True
        mock_forward.return_value = {"payload": {"status": "ok"}, "status_code": 200}

        out = MessageIngressService.handle_ui_ingress(_envelope("query"))
        self.assertTrue(out.forwarded)
        called_kwargs = mock_forward.call_args.kwargs
        self.assertEqual(called_kwargs["url"], "http://127.0.0.1:8201/v1/messages/test")

    @patch("app.services.message_ingress.get_config")
    @patch("app.services.message_ingress.MessageIngressService._store_envelope")
    @patch("app.services.message_ingress.MessageIngressService._forward_message")
    def test_ui_query_forward_failure_raises_runtime_error(
        self,
        mock_forward: MagicMock,
        mock_store: MagicMock,
        mock_get_cfg: MagicMock,
    ) -> None:
        """Downstream forwarding errors must be surfaced as runtime failures."""
        mock_get_cfg.return_value = _base_cfg()
        mock_store.return_value = True
        mock_forward.side_effect = RuntimeError("downstream exploded")

        with self.assertRaises(RuntimeError) as ctx:
            MessageIngressService.handle_ui_ingress(_envelope("query"))
        self.assertIn("Failed to forward query message", str(ctx.exception))

    @patch("app.services.message_ingress.logger.info")
    @patch("app.services.message_ingress.get_config")
    @patch("app.services.message_ingress.MessageIngressService._store_envelope")
    @patch("app.services.message_ingress.MessageIngressService._forward_message")
    def test_non_query_done_log_contains_completion_fields(
        self,
        mock_forward: MagicMock,
        mock_store: MagicMock,
        mock_get_cfg: MagicMock,
        mock_logger_info: MagicMock,
    ) -> None:
        """non-query branch should emit completion log with trace and status fields."""
        mock_get_cfg.return_value = _base_cfg()
        mock_store.return_value = True

        out = MessageIngressService.handle_ui_ingress(_envelope("clarification"))
        self.assertFalse(out.forwarded)
        mock_forward.assert_not_called()

        done_call = None
        for call in mock_logger_info.call_args_list:
            if len(call.args) > 1 and call.args[1] == "ui_ingress_non_query_done":
                done_call = call
                break
        self.assertIsNotNone(done_call)
        assert done_call is not None
        self.assertEqual(done_call.args[8], 200)
        self.assertGreaterEqual(done_call.args[9], 0)
        self.assertTrue(done_call.args[10])

    @patch("app.services.message_ingress.httpx.Client", new=_MockSseHttpClient)
    @patch("app.services.message_ingress.logger.warning")
    def test_sse_invalid_frame_over_threshold_raises(self, mock_warning: MagicMock) -> None:
        """Too many invalid SSE data frames should fail fast with diagnostics."""
        _MockSseHttpClient.response_lines = [
            "data: {bad-1}",
            "data: {bad-2}",
            "data: {bad-3}",
            "data: {bad-4}",
        ]
        with self.assertRaises(RuntimeError) as ctx:
            MessageIngressService._forward_message(
                url="http://127.0.0.1:8001/v1/chat/stream",
                timeout_seconds=10,
                api_key="",
                envelope=_envelope("query"),
            )
        self.assertIn("too many invalid JSON frames", str(ctx.exception))
        self.assertGreaterEqual(mock_warning.call_count, 4)

    @patch("app.services.message_ingress.httpx.Client", new=_MockSseHttpClient)
    @patch("app.services.message_ingress.logger.warning")
    def test_sse_single_invalid_frame_logs_and_recovers(self, mock_warning: MagicMock) -> None:
        """One invalid SSE frame should be logged while valid deltas still pass."""
        envelope = _envelope("query")
        _MockSseHttpClient.response_lines = [
            "data: {broken}",
            "data: {\"delta\": \"hello\"}",
            "data: {\"done\": true}",
        ]
        out = MessageIngressService._forward_message(
            url="http://127.0.0.1:8001/v1/chat/stream",
            timeout_seconds=10,
            api_key="",
            envelope=envelope,
        )
        envelope_payload = out["payload"]["downstream"]["envelope"]
        self.assertEqual(envelope_payload["type"], "answer")
        self.assertEqual(envelope_payload["content"], "hello")
        self.assertGreaterEqual(mock_warning.call_count, 1)
        warning_args = mock_warning.call_args_list[0].args
        self.assertIn("ui_ingress_sse_invalid_frame", warning_args[0])
        self.assertEqual(warning_args[1], envelope.message_id)
        self.assertEqual(warning_args[2], envelope.session_id)
        self.assertEqual(warning_args[3], envelope.turn_id)


if __name__ == "__main__":
    unittest.main()

