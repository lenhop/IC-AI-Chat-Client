"""Unit tests for Gradio handler service key paths."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import fakeredis

from app.config import RedisSettings
from app.memory.session_store import SessionStore
from app.ui.gradio_handlers import GradioHandlerService
from app.ui.gradio_persistence import GRADIO_SESSION_KEY


def _rs() -> RedisSettings:
    return RedisSettings(
        enabled=True,
        url="redis://localhost:6379/0",
        key_prefix="gh:",
        session_ttl_seconds=3600,
    )


class _InnerRequest:
    def __init__(self, session: dict) -> None:
        self.session = session


class _GradioRequestStub:
    def __init__(self, session: dict) -> None:
        self.request = _InnerRequest(session)


class GradioChatHandlerTests(unittest.TestCase):
    """Cover query persist, success answer persist, and failure fallback paths."""

    def setUp(self) -> None:
        self.redis_client = fakeredis.FakeRedis(decode_responses=True)
        self.redis_settings = _rs()
        self.store = SessionStore(self.redis_client, self.redis_settings)
        self.session_id = self.store.create_session("u1", "deepseek")
        self.session_map: dict = {GRADIO_SESSION_KEY: self.session_id}
        self.request = _GradioRequestStub(self.session_map)

    def test_user_turn_persists_query(self) -> None:
        """Submitting a user line should append a query row with active turn_id."""
        with patch(
            "app.ui.gradio_persistence.get_redis_for_gradio",
            return_value=(self.redis_client, self.redis_settings),
        ):
            _, history, _ = GradioHandlerService.handle_user_turn(
                "hello",
                [],
                self.session_id,
                request=self.request,
                user_id="u1",
            )
        self.assertEqual(history[-1]["role"], "user")
        self.assertEqual(history[-1]["content"], "hello")
        rows = self.store.get_messages(self.session_id, "u1")
        self.assertEqual([r.get("type") for r in rows], ["query"])
        self.assertTrue(str(rows[0].get("turn_id") or "").strip())

    def test_stream_success_persists_answer_same_turn_id(self) -> None:
        """Successful stream should append answer and keep query/answer turn_id一致."""
        with patch(
            "app.ui.gradio_persistence.get_redis_for_gradio",
            return_value=(self.redis_client, self.redis_settings),
        ):
            _, history, _ = GradioHandlerService.handle_user_turn(
                "hello",
                [],
                self.session_id,
                request=self.request,
                user_id="u1",
            )
            with patch(
                "app.ui.gradio_handlers.iter_chat_text_deltas",
                return_value=iter(["A", "B"]),
            ):
                list(
                    GradioHandlerService.stream_assistant(
                        history,
                        self.session_id,
                        request=self.request,
                        user_id="u1",
                        chat_mode="messages",
                        memory_rounds=3,
                    )
                )

        rows = self.store.get_messages(self.session_id, "u1")
        self.assertEqual([r.get("type") for r in rows], ["query", "answer"])
        self.assertEqual(rows[0].get("content"), "hello")
        self.assertEqual(rows[1].get("content"), "AB")
        self.assertEqual(rows[0].get("turn_id"), rows[1].get("turn_id"))

    def test_stream_runtime_error_does_not_persist_answer(self) -> None:
        """Runtime failure should render error text and avoid answer persistence."""

        def _raise_runtime(*_args, **_kwargs):
            raise RuntimeError("boom")
            yield ""

        with patch(
            "app.ui.gradio_persistence.get_redis_for_gradio",
            return_value=(self.redis_client, self.redis_settings),
        ):
            _, history, _ = GradioHandlerService.handle_user_turn(
                "hello",
                [],
                self.session_id,
                request=self.request,
                user_id="u1",
            )
            with patch(
                "app.ui.gradio_handlers.iter_chat_text_deltas",
                side_effect=_raise_runtime,
            ):
                states = list(
                    GradioHandlerService.stream_assistant(
                        history,
                        self.session_id,
                        request=self.request,
                        user_id="u1",
                        chat_mode="messages",
                        memory_rounds=3,
                    )
                )
        self.assertIn("[错误] boom", states[-1][-1]["content"])
        rows = self.store.get_messages(self.session_id, "u1")
        self.assertNotIn("answer", [str(r.get("type")) for r in rows])


if __name__ == "__main__":
    unittest.main()
