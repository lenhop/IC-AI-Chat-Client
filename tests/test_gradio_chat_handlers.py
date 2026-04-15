"""Unit tests for Gradio handler service key paths."""

from __future__ import annotations

import asyncio
import gc
import unittest
import warnings
from unittest.mock import patch

import fakeredis

from app.config import RedisSettings
from app.memory.session_store import SessionStore
from app.ui.gradio_handlers import GradioHandlerService
from app.ui.gradio_layout import GradioLayoutService
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


def _close_stray_asyncio_event_loops() -> None:
    """Close event loops left open by third-party code (e.g. Gradio) during tests."""
    for obj in gc.get_objects():
        if isinstance(obj, asyncio.AbstractEventLoop) and not obj.is_closed():
            try:
                obj.close()
            except (RuntimeError, OSError):
                pass
    try:
        asyncio.set_event_loop(None)
    except (RuntimeError, OSError):
        pass


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
        """Successful stream should append answer with the same turn id."""
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
        self.assertIn("[Error] boom", states[-1][-1]["content"])
        rows = self.store.get_messages(self.session_id, "u1")
        self.assertNotIn("answer", [str(r.get("type")) for r in rows])

    def test_clone_message_history_filters_invalid_rows(self) -> None:
        """History normalization keeps only valid role/content rows for Chatbot."""
        history = [
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": 123},
            {"role": "tool", "content": "drop"},
            "bad-row",
            {"content": "missing role"},
        ]
        rows = GradioHandlerService.clone_message_history(history)
        self.assertEqual(
            rows,
            [
                {"role": "user", "content": "ok"},
                {"role": "assistant", "content": "123"},
            ],
        )


class GradioLayoutRequestForwardingTests(unittest.TestCase):
    """Ensure Gradio layout wrappers forward request to downstream callbacks."""

    def test_wrapped_callbacks_forward_same_request_object(self) -> None:
        """Load/user/stream/clear wrapper callbacks should all pass request through."""
        observed: dict = {}

        def _on_load(request=None):
            observed["load"] = request
            return "sid", []

        def _on_user_turn(message, history, session_id, request=None):
            observed["user_turn"] = request
            return "", history, session_id

        def _on_stream_assistant(history, session_id, request=None):
            observed["stream"] = request
            yield history

        def _on_clear_chat(session_id, request=None):
            observed["clear"] = request
            return [], "", session_id

        # Gradio may create asyncio loops during block wiring; suppress known ResourceWarning noise.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            blocks = GradioLayoutService.build_blocks(
                page_title="x",
                theme_key="minimal",
                backend_label="DeepSeek",
                model_label="deepseek-chat",
                user_label="local-dev",
                on_load=_on_load,
                on_user_turn=_on_user_turn,
                on_stream_assistant=_on_stream_assistant,
                on_clear_chat=_on_clear_chat,
            )
            request_stub = object()

            for _, wrapped in blocks.fns.items():
                name = wrapped.fn.__name__
                if name == "_on_load_wrapped":
                    wrapped.fn(request_stub)
                elif name == "_on_user_turn_wrapped":
                    wrapped.fn("hello", [], "sid", request_stub)
                elif name == "_on_stream_assistant_wrapped":
                    list(wrapped.fn([], "sid", request_stub))
                elif name == "_on_clear_chat_wrapped":
                    wrapped.fn("sid", request_stub)

        self.assertIs(observed.get("load"), request_stub)
        self.assertIs(observed.get("user_turn"), request_stub)
        self.assertIs(observed.get("stream"), request_stub)
        self.assertIs(observed.get("clear"), request_stub)
        _close_stray_asyncio_event_loops()


if __name__ == "__main__":
    unittest.main()
