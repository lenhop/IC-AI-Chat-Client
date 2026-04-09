"""Route-level tests for LLM service SSE stream API."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.llm_service.main import app


class LlmServiceStreamApiTests(unittest.TestCase):
    """Verify /v1/chat/stream SSE frames without real LLM backends."""

    @patch("app.llm_service.main.validate_llm_worker_env", return_value=None)
    @patch("app.llm_service.main.stream_chat")
    @patch("app.llm_service.main.normalize_messages")
    def test_stream_success_emits_deltas_then_done(
        self,
        mock_normalize: MagicMock,
        mock_stream_chat: MagicMock,
        _mock_validate: MagicMock,
    ) -> None:
        # Keep payload normalized and stream deterministic deltas.
        mock_normalize.return_value = [{"role": "user", "content": "hello"}]
        mock_stream_chat.return_value = iter(["hi", " there"])

        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/stream",
                json={
                    "messages": [{"role": "user", "content": "hello"}],
                    "backend": "deepseek",
                    "model": "m1",
                },
            )

        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/event-stream", resp.headers.get("content-type", ""))
        self.assertIn('data: {"delta": "hi"}', resp.text)
        self.assertIn('data: {"delta": " there"}', resp.text)
        self.assertIn('data: {"done": true}', resp.text)
        self.assertLess(resp.text.find('"delta": "hi"'), resp.text.find('"done": true'))

    @patch("app.llm_service.main.validate_llm_worker_env", return_value=None)
    @patch("app.llm_service.main.stream_chat")
    @patch("app.llm_service.main.normalize_messages")
    def test_stream_validation_error_emits_error_frame(
        self,
        mock_normalize: MagicMock,
        mock_stream_chat: MagicMock,
        _mock_validate: MagicMock,
    ) -> None:
        # Reject malformed message list before stream starts.
        mock_normalize.side_effect = ValueError("bad messages")

        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/stream",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )

        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/event-stream", resp.headers.get("content-type", ""))
        self.assertIn('data: {"error": "bad messages"}', resp.text)
        mock_stream_chat.assert_not_called()

    @patch("app.llm_service.main.validate_llm_worker_env", return_value=None)
    @patch("app.llm_service.main.stream_chat")
    @patch("app.llm_service.main.normalize_messages")
    def test_stream_runtime_error_emits_error_frame(
        self,
        mock_normalize: MagicMock,
        mock_stream_chat: MagicMock,
        _mock_validate: MagicMock,
    ) -> None:
        # Simulate upstream runtime failure during stream.
        mock_normalize.return_value = [{"role": "user", "content": "hello"}]

        def _raise_runtime(*_args, **_kwargs):
            raise RuntimeError("upstream failed")
            yield ""

        mock_stream_chat.side_effect = _raise_runtime

        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/stream",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )

        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/event-stream", resp.headers.get("content-type", ""))
        self.assertIn('data: {"error": "upstream failed"}', resp.text)

    @patch("app.llm_service.main.validate_llm_worker_env", return_value=None)
    @patch("app.llm_service.main.stream_chat")
    @patch("app.llm_service.main.normalize_messages")
    def test_stream_unknown_error_emits_internal_error_frame(
        self,
        mock_normalize: MagicMock,
        mock_stream_chat: MagicMock,
        _mock_validate: MagicMock,
    ) -> None:
        # Unknown exceptions are mapped to stable internal-error message.
        mock_normalize.return_value = [{"role": "user", "content": "hello"}]

        def _raise_unknown(*_args, **_kwargs):
            raise Exception("unexpected")
            yield ""

        mock_stream_chat.side_effect = _raise_unknown

        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/stream",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )

        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/event-stream", resp.headers.get("content-type", ""))
        self.assertIn('data: {"error": "Internal error while streaming."}', resp.text)


if __name__ == "__main__":
    unittest.main()
