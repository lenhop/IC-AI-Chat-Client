"""Unit tests for Gradio stage-message Redis persistence (fakeredis only)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import fakeredis

from app.config import RedisSettings
from app.memory.session_store import SessionStore
from app.ui.gradio_persistence import GradioPersistenceService
from app.ui.gradio_session_turn import GradioSessionTurn


def _rs() -> RedisSettings:
    return RedisSettings(
        enabled=True,
        url="redis://localhost:6379/0",
        key_prefix="gs:",
        session_ttl_seconds=3600,
    )


class _InnerRequest:
    def __init__(self, session: dict) -> None:
        self.session = session


class _GradioRequestStub:
    def __init__(self, session: dict) -> None:
        self.request = _InnerRequest(session)


class GradioStagePersistTests(unittest.TestCase):
    """Validate stage rows are appended in-order on the active turn."""

    def setUp(self) -> None:
        self.client = fakeredis.FakeRedis(decode_responses=True)
        self.store = SessionStore(self.client, _rs())
        self.session_id = self.store.create_session("u1", "deepseek")
        self.session = {}
        self.turn_id = GradioSessionTurn.ensure_active_turn_id(self.session)
        self.request = _GradioRequestStub(self.session)

    def _append_query(self, text: str) -> None:
        self.store.append_memory_message(
            self.session_id,
            "u1",
            message_type="query",
            content=text,
            turn_id=self.turn_id,
        )

    def _append_answer(self, text: str) -> None:
        self.store.append_memory_message(
            self.session_id,
            "u1",
            message_type="answer",
            content=text,
            turn_id=self.turn_id,
        )

    def test_same_turn_id_order_query_plan_reason_answer(self) -> None:
        with patch(
            "app.ui.gradio_persistence.get_redis_for_gradio",
            return_value=(self.client, _rs()),
        ):
            self._append_query("user asks")
            GradioPersistenceService.persist_stage_message(
                self.session_id,
                self.request,
                user_id="u1",
                message_type="plan",
                content="  first plan  ",
            )
            GradioPersistenceService.persist_stage_message(
                self.session_id,
                self.request,
                user_id="u1",
                message_type="reason",
                content="because",
            )
            self._append_answer("final answer")

        rows = self.store.get_messages(self.session_id, "u1")
        self.assertEqual(
            [row.get("type") for row in rows],
            ["query", "plan", "reason", "answer"],
        )
        self.assertEqual(
            {str(row.get("turn_id")) for row in rows},
            {self.turn_id},
        )

    def test_stage_messages_exist_before_answer(self) -> None:
        with patch(
            "app.ui.gradio_persistence.get_redis_for_gradio",
            return_value=(self.client, _rs()),
        ):
            self._append_query("user asks")
            GradioPersistenceService.persist_stage_message(
                self.session_id,
                self.request,
                user_id="u1",
                message_type="plan",
                content="draft plan",
            )
            GradioPersistenceService.persist_stage_message(
                self.session_id,
                self.request,
                user_id="u1",
                message_type="reason",
                content="analysis",
            )
            mid_rows = self.store.get_messages(self.session_id, "u1")
            self.assertEqual(
                [row.get("type") for row in mid_rows],
                ["query", "plan", "reason"],
            )
            self.assertNotIn("answer", [row.get("type") for row in mid_rows])

    def test_empty_stage_content_is_ignored(self) -> None:
        with patch(
            "app.ui.gradio_persistence.get_redis_for_gradio",
            return_value=(self.client, _rs()),
        ):
            self._append_query("user asks")
            GradioPersistenceService.persist_stage_message(
                self.session_id,
                self.request,
                user_id="u1",
                message_type="plan",
                content="   ",
            )
            GradioPersistenceService.persist_stage_message(
                self.session_id,
                self.request,
                user_id="u1",
                message_type="reason",
                content="\n\t",
            )
            rows = self.store.get_messages(self.session_id, "u1")
            self.assertEqual([row.get("type") for row in rows], ["query"])

    def test_persist_answer_finishes_turn_and_clears_active_turn_id(self) -> None:
        """Successful answer persistence should reuse current turn and clear lifecycle state."""
        with patch(
            "app.ui.gradio_persistence.get_redis_for_gradio",
            return_value=(self.client, _rs()),
        ):
            self._append_query("user asks")
            GradioPersistenceService.persist_answer_and_finish_turn(
                self.session_id,
                self.request,
                user_id="u1",
                assistant_text="final answer",
            )

        rows = self.store.get_messages(self.session_id, "u1")
        self.assertEqual([row.get("type") for row in rows], ["query", "answer"])
        self.assertEqual(rows[0].get("turn_id"), rows[1].get("turn_id"))
        self.assertEqual(GradioSessionTurn.get_active_turn_id(self.session), "")

    def test_persist_answer_without_active_turn_id_is_ignored(self) -> None:
        """Answer persistence should be skipped when there is no active turn id."""
        no_turn_session: dict = {}
        request_without_turn = _GradioRequestStub(no_turn_session)
        with patch(
            "app.ui.gradio_persistence.get_redis_for_gradio",
            return_value=(self.client, _rs()),
        ):
            self._append_query("user asks")
            GradioPersistenceService.persist_answer_and_finish_turn(
                self.session_id,
                request_without_turn,
                user_id="u1",
                assistant_text="final answer",
            )

        rows = self.store.get_messages(self.session_id, "u1")
        self.assertEqual([row.get("type") for row in rows], ["query"])


if __name__ == "__main__":
    unittest.main()
