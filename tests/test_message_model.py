"""Unit tests for Gradio message formatting (no Gradio runtime)."""

from __future__ import annotations

import re
import unittest

from app.config import MessageDisplayOptions
from app.ui.message_model import GradioMessageFormatter


def _canon(
    mtype: str,
    content: str,
    *,
    ts: str = "2020-01-01 00:00:00 UTC",
) -> dict:
    return {
        "user_id": "u",
        "session_id": "s",
        "type": mtype,
        "content": content,
        "timestamp": ts,
        "turn_id": "",
    }


class TestGradioMessageFormatter(unittest.TestCase):
    """Role/content rows and display toggles."""

    def test_query_plain_text(self) -> None:
        row = GradioMessageFormatter.to_chat_row(_canon("query", "hello"))
        assert row is not None
        self.assertEqual(row["role"], "user")
        self.assertEqual(row["content"], "hello")

    def test_answer_has_heading(self) -> None:
        row = GradioMessageFormatter.to_chat_row(_canon("answer", "done"))
        assert row is not None
        self.assertEqual(row["role"], "assistant")
        self.assertIn("Final answer", row["content"])
        self.assertIn("done", row["content"])

    def test_classification_no_ordered_list_marker(self) -> None:
        row = GradioMessageFormatter.to_chat_row(_canon("classification", "intent=x"))
        assert row is not None
        self.assertNotRegex(row["content"], re.compile(r"^\s*\d+\.\s", re.MULTILINE))

    def test_dispatcher_section(self) -> None:
        row = GradioMessageFormatter.to_chat_row(_canon("dispatcher", "step"))
        assert row is not None
        self.assertIn("Dispatcher", row["content"])

    def test_unknown_type_fallback(self) -> None:
        row = GradioMessageFormatter.to_chat_row(_canon("custom_type", "body"))
        assert row is not None
        self.assertIn("custom_type", row["content"])

    def test_should_display_respects_flags(self) -> None:
        off = MessageDisplayOptions(
            clarification_message_display_enable=False,
            rewriting_message_display_enable=True,
            classification_message_display_enable=True,
            plan_message_display_enable=True,
            reason_message_display_enable=True,
            context_message_display_enable=True,
            dispatcher_message_display_enable=True,
        )
        self.assertTrue(GradioMessageFormatter.should_display_type("query", off))
        self.assertTrue(GradioMessageFormatter.should_display_type("answer", off))
        self.assertFalse(GradioMessageFormatter.should_display_type("clarification", off))
        self.assertTrue(GradioMessageFormatter.should_display_type("plan", off))


if __name__ == "__main__":
    unittest.main()
