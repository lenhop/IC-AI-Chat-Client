"""
Unit tests for M3 Redis session store (fakeredis, no real server).
"""

from __future__ import annotations

import json
import unittest

import fakeredis

from app.config import MessageDisplayOptions, RedisSettings
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

    def test_append_turn_with_explicit_turn_id(self) -> None:
        """Optional turn_id keeps query/answer on one id (Gradio pre-persist path)."""
        sid = self._store.create_session("u1", "deepseek")
        fixed = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        self._store.append_turn(sid, "u1", "q", "a", turn_id=fixed)
        raw = self._store.get_messages(sid, "u1")
        self.assertEqual(raw[0].get("turn_id"), fixed)
        self.assertEqual(raw[1].get("turn_id"), fixed)

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

    def test_historical_role_based_skipped_in_get_messages(self) -> None:
        """Option B: historical role/ts rows are not normalized and are dropped on read."""
        sid = self._store.create_session("u1", "deepseek")
        msg_k = f"test:session:{sid}:messages"
        legacy = json.dumps({"role": "user", "content": "old", "ts": 1700000000})
        self._client.rpush(msg_k, legacy)
        msgs = self._store.get_messages(sid, "u1")
        self.assertEqual(msgs, [])

    def test_normalize_requires_type_and_timestamp(self) -> None:
        self.assertIsNone(normalize_stored_message({"content": "x"}, "s", "u"))
        self.assertIsNone(
            normalize_stored_message(
                {"type": "query", "content": "x"},
                "s",
                "u",
            )
        )
        n = normalize_stored_message(
            {
                "type": "query",
                "content": "hi",
                "timestamp": "2020-01-01 00:00:00 UTC",
            },
            "s1",
            "u1",
        )
        self.assertIsNotNone(n)
        assert n is not None
        self.assertEqual(n["type"], "query")
        self.assertEqual(n["user_id"], "u1")
        self.assertEqual(n["session_id"], "s1")

    def test_append_memory_message_round_trip(self) -> None:
        sid = self._store.create_session("u1", "deepseek")
        self._store.append_memory_message(
            sid,
            "u1",
            message_type="plan",
            content="do things",
        )
        msgs = self._store.get_messages(sid, "u1")
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["type"], "plan")
        self.assertEqual(msgs[0]["content"], "do things")

    def test_gradio_history_honors_display_off(self) -> None:
        msgs = [
            {
                "user_id": "u",
                "session_id": "s",
                "type": "query",
                "content": "q",
                "timestamp": "t",
                "turn_id": "",
            },
            {
                "user_id": "u",
                "session_id": "s",
                "type": "plan",
                "content": "hidden",
                "timestamp": "t",
                "turn_id": "",
            },
            {
                "user_id": "u",
                "session_id": "s",
                "type": "answer",
                "content": "a",
                "timestamp": "t",
                "turn_id": "",
            },
        ]
        off = MessageDisplayOptions(
            clarification_message_display_enable=True,
            rewriting_message_display_enable=True,
            classification_message_display_enable=True,
            plan_message_display_enable=False,
            reason_message_display_enable=True,
            context_message_display_enable=True,
            dispatcher_message_display_enable=True,
        )
        rows = gradio_history_from_stored(msgs, off)
        roles = [r["role"] for r in rows]
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)
        combined = " ".join(str(r.get("content", "")) for r in rows)
        self.assertNotIn("hidden", combined)

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
