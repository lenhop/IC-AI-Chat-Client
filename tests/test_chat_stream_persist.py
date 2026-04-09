"""Tests for chat_stream Redis persistence path (fakeredis, no live LLM)."""

from __future__ import annotations

import types
import unittest
from unittest.mock import patch

import fakeredis
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import RedisSettings
from app.memory.session_store import SessionStore
from app.routes.chat_stream import router


def _rs() -> RedisSettings:
    return RedisSettings(
        enabled=True,
        url="redis://localhost:6379/0",
        key_prefix="cs:",
        session_ttl_seconds=3600,
    )


def _cfg_stub() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        user_id="u1",
        chat_mode="messages",
        memory_rounds=3,
    )


class ChatStreamPersistTests(unittest.TestCase):
    """Ensure successful stream persists query/answer with one turn_id."""

    def setUp(self) -> None:
        self.client_redis = fakeredis.FakeRedis(decode_responses=True)
        self.rs = _rs()
        self.store = SessionStore(self.client_redis, self.rs)
        self.session_id = self.store.create_session("u1", "deepseek")

        app = FastAPI()
        app.state.redis = self.client_redis
        app.state.redis_settings = self.rs
        app.include_router(router)
        self.client = TestClient(app)

    def test_success_persists_query_then_answer_same_turn_id(self) -> None:
        with patch("app.routes.chat_stream.get_config", return_value=_cfg_stub()):
            with patch(
                "app.routes.chat_stream.iter_chat_text_deltas",
                return_value=iter(["A", "B"]),
            ):
                resp = self.client.post(
                    "/api/chat/stream",
                    json={
                        "messages": [{"role": "user", "content": "hello"}],
                        "session_id": self.session_id,
                    },
                )
        self.assertEqual(resp.status_code, 200)
        self.assertIn('"done": true', resp.text)

        rows = self.store.get_messages(self.session_id, "u1")
        self.assertEqual([row.get("type") for row in rows], ["query", "answer"])
        self.assertEqual(rows[0].get("content"), "hello")
        self.assertEqual(rows[1].get("content"), "AB")
        self.assertEqual(rows[0].get("turn_id"), rows[1].get("turn_id"))
        self.assertTrue(str(rows[0].get("turn_id") or "").strip())

    def test_failed_stream_does_not_persist_answer(self) -> None:
        def _raise_runtime(*_args, **_kwargs):
            raise RuntimeError("boom")
            yield ""

        with patch("app.routes.chat_stream.get_config", return_value=_cfg_stub()):
            with patch("app.routes.chat_stream.iter_chat_text_deltas", side_effect=_raise_runtime):
                resp = self.client.post(
                    "/api/chat/stream",
                    json={
                        "messages": [{"role": "user", "content": "hello"}],
                        "session_id": self.session_id,
                    },
                )
        self.assertEqual(resp.status_code, 200)
        self.assertIn('"error": "boom"', resp.text)

        rows = self.store.get_messages(self.session_id, "u1")
        self.assertNotIn("answer", [str(row.get("type")) for row in rows])


if __name__ == "__main__":
    unittest.main()
