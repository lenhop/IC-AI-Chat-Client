"""Unit tests for ``llm_transport`` (mocked HTTP, no live LLM)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from app.config import AppConfig
from app.services.llm_transport import iter_chat_text_deltas, validate_or_normalize_messages


def _cfg_http(url: str = "http://127.0.0.1:9") -> AppConfig:
    return AppConfig(
        user_id="u",
        session_id="",
        memory_rounds=3,
        chat_mode="messages",
        llm_backend="deepseek",
        ollama_base_url="http://localhost:11434",
        ollama_generate_model="m",
        ollama_request_timeout=600,
        ollama_embed_model="e",
        deepseek_api_key="k",
        deepseek_llm_model="deepseek-chat",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_request_timeout=600,
        llm_transport="http",
        llm_service_url=url,
        llm_service_timeout_seconds=120,
        llm_service_api_key="",
    )


class TestIterChatTextDeltas(unittest.TestCase):
    """Local vs HTTP selection."""

    @patch("app.services.llm_transport.stream_chat")
    @patch("app.services.llm_transport.get_config")
    def test_local_delegates_to_stream_chat(self, mock_gc: MagicMock, mock_sc: MagicMock) -> None:
        mock_gc.return_value = MagicMock(llm_transport="local")

        def _gen(*_a, **_k):
            yield "ab"

        mock_sc.side_effect = _gen
        out = list(
            iter_chat_text_deltas(
                [{"role": "user", "content": "hi"}],
            )
        )
        self.assertEqual(out, ["ab"])
        mock_sc.assert_called_once()

    @patch("app.services.llm_transport.httpx.Client")
    @patch("app.services.llm_transport.get_config")
    def test_http_reads_sse_deltas(self, mock_gc: MagicMock, mock_client_cls: MagicMock) -> None:
        mock_gc.return_value = _cfg_http("http://worker.example")

        class _StreamCtx:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def raise_for_status(self) -> None:
                return None

            def iter_lines(self):
                yield 'data: {"delta": "x"}'
                yield 'data: {"done": true}'

        stream_cm = MagicMock()
        stream_cm.__enter__.return_value = _StreamCtx()
        stream_cm.__exit__.return_value = None
        mock_http = MagicMock()
        mock_http.stream.return_value = stream_cm
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_http
        mock_client.__exit__.return_value = None
        mock_client_cls.return_value = mock_client

        deltas = list(
            iter_chat_text_deltas(
                [{"role": "user", "content": "hi"}],
            )
        )
        self.assertEqual(deltas, ["x"])
        mock_http.stream.assert_called_once()
        args, _kwargs = mock_http.stream.call_args
        self.assertIn("/v1/chat/stream", str(args[1]))

    @patch("app.services.llm_transport.httpx.Client")
    @patch("app.services.llm_transport.get_config")
    def test_http_stage_callback_keeps_delta_contract(
        self,
        mock_gc: MagicMock,
        mock_client_cls: MagicMock,
    ) -> None:
        mock_gc.return_value = _cfg_http("http://worker.example")

        class _StreamCtx:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def raise_for_status(self) -> None:
                return None

            def iter_lines(self):
                yield 'data: {"type": "plan", "content": "draft plan"}'
                yield 'data: {"delta": "x"}'
                yield 'data: {"stage": "reason", "content": "because"}'
                yield 'data: {"done": true}'

        stream_cm = MagicMock()
        stream_cm.__enter__.return_value = _StreamCtx()
        stream_cm.__exit__.return_value = None
        mock_http = MagicMock()
        mock_http.stream.return_value = stream_cm
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_http
        mock_client.__exit__.return_value = None
        mock_client_cls.return_value = mock_client

        stage_rows = []

        deltas = list(
            iter_chat_text_deltas(
                [{"role": "user", "content": "hi"}],
                on_stage_message=lambda mt, c: stage_rows.append((mt, c)),
            )
        )
        self.assertEqual(deltas, ["x"])
        self.assertEqual(
            stage_rows,
            [("plan", "draft plan"), ("reason", "because")],
        )

    @patch("app.services.llm_transport._normalize_messages")
    def test_validate_or_normalize_messages_success(self, mock_norm: MagicMock) -> None:
        mock_norm.return_value = [{"role": "user", "content": "hi"}]
        out = validate_or_normalize_messages([{"role": "user", "content": "hi"}])
        self.assertEqual(out, [{"role": "user", "content": "hi"}])
        mock_norm.assert_called_once()

    @patch("app.services.llm_transport._normalize_messages")
    def test_validate_or_normalize_messages_logs_and_raises(self, mock_norm: MagicMock) -> None:
        mock_norm.side_effect = ValueError("bad")
        with self.assertLogs("app.services.llm_transport", level="WARNING") as logs:
            with self.assertRaises(ValueError):
                validate_or_normalize_messages([{"role": "user", "content": ""}])
        self.assertTrue(any("invalid messages" in line for line in logs.output))


if __name__ == "__main__":
    unittest.main()
