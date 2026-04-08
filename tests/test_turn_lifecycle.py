"""
M3 v3.2 turn lifecycle: Starlette session ``turn_id`` + Redis ``append_memory_message`` pairs.
"""

from __future__ import annotations

import unittest

import fakeredis

from app.config import RedisSettings
from app.memory.session_store import SessionStore
from app.ui.gradio_session_turn import ACTIVE_TURN_ID_SESSION_KEY, GradioSessionTurn


def _rs() -> RedisSettings:
    return RedisSettings(
        enabled=True,
        url="redis://localhost:6379/0",
        key_prefix="tl:",
        session_ttl_seconds=3600,
    )


class TurnLifecycleTests(unittest.TestCase):
    """Session dict + SessionStore behavior without Gradio runtime."""

    def test_ensure_and_clear_session_turn(self) -> None:
        s: dict = {}
        a = GradioSessionTurn.ensure_active_turn_id(s)
        b = GradioSessionTurn.ensure_active_turn_id(s)
        self.assertEqual(a, b)
        self.assertEqual(s.get(ACTIVE_TURN_ID_SESSION_KEY), a)
        GradioSessionTurn.clear_active_turn_id(s)
        self.assertNotIn(ACTIVE_TURN_ID_SESSION_KEY, s)
        self.assertEqual(GradioSessionTurn.get_active_turn_id(s), "")

    def test_query_then_answer_same_turn_id(self) -> None:
        client = fakeredis.FakeRedis(decode_responses=True)
        store = SessionStore(client, _rs())
        sid = store.create_session("u1", "deepseek")
        sess: dict = {}
        tid = GradioSessionTurn.ensure_active_turn_id(sess)
        store.append_memory_message(
            sid,
            "u1",
            message_type="query",
            content="hello",
            turn_id=tid,
        )
        store.append_memory_message(
            sid,
            "u1",
            message_type="answer",
            content="world",
            turn_id=tid,
        )
        GradioSessionTurn.clear_active_turn_id(sess)
        self.assertEqual(GradioSessionTurn.get_active_turn_id(sess), "")
        rows = store.get_messages(sid, "u1")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].get("turn_id"), rows[1].get("turn_id"))


if __name__ == "__main__":
    unittest.main()
