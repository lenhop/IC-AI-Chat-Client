"""
Unit tests for M3 Redis session store (fakeredis, no real server).
"""

from __future__ import annotations

import unittest

import fakeredis

from app.config import RedisSettings
from app.memory.session_store import (
    SessionAccessDeniedError,
    SessionNotFoundError,
    SessionStore,
    gradio_history_from_stored,
    messages_for_openai_payload,
    normalize_stored_message,
)

def _fake_settings() -> RedisSettings:
    return RedisSettings(
        enabled=True,
        url="redis://localhost:6379/0",
        key_prefix="test:",
        session_ttl_seconds=3600,
    )


class SessionStoreTests(unittest.TestCase):
    """Exercise create, get, append, clear, and access control."""

    def setUp(self) -> None:
        self._client = fakeredis.FakeRedis(decode_responses=True)
        self._store = SessionStore(self._client, _fake_settings())

    def test_create_and_get_empty_messages(self) -> None:
        sid = self._store.create_session("u1", "deepseek")
        self.assertTrue(sid)
        msgs = self._store.get_messages(sid, "u1")
        self.assertEqual(msgs, [])

    def test_append_turn_round_trip(self) -> None:
        sid = self._store.create_session("u1", "ollama")
        self._store.append_turn(sid, "u1", "hello", "world")
        raw = self._store.get_messages(sid, "u1")
        self.assertEqual(len(raw), 2)
        self.assertEqual(raw[0].get("type"), "query")
        self.assertEqual(raw[0].get("content"), "hello")
        self.assertEqual(raw[0].get("user_id"), "u1")
        self.assertEqual(raw[0].get("session_id"), sid)
        self.assertTrue(raw[0].get("timestamp"))
        self.assertEqual(raw[1].get("type"), "answer")
        self.assertEqual(raw[1].get("content"), "world")
        self.assertEqual(raw[0].get("turn_id"), raw[1].get("turn_id"))
        openai = messages_for_openai_payload(raw)
        self.assertEqual(len(openai), 2)
        gradio_rows = gradio_history_from_stored(raw)
        self.assertEqual(len(gradio_rows), 2)

    def test_legacy_normalize_role_based(self) -> None:
        sid = "s-leg"
        n = normalize_stored_message(
            {"role": "user", "content": "hi", "ts": 1700000000},
            sid,
            "u1",
        )
        self.assertIsNotNone(n)
        assert n is not None
        self.assertEqual(n["type"], "query")
        self.assertEqual(n["content"], "hi")

    def test_wrong_user_denied(self) -> None:
        sid = self._store.create_session("owner", "deepseek")
        with self.assertRaises(SessionAccessDeniedError):
            self._store.get_messages(sid, "other")

    def test_missing_session(self) -> None:
        with self.assertRaises(SessionNotFoundError):
            self._store.get_messages("00000000-0000-0000-0000-000000000000", "u1")

    def test_clear_messages(self) -> None:
        sid = self._store.create_session("u1", "deepseek")
        self._store.append_turn(sid, "u1", "a", "b")
        self._store.clear_messages(sid, "u1")
        self.assertEqual(self._store.get_messages(sid, "u1"), [])


if __name__ == "__main__":
    unittest.main()
