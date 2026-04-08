"""Unit tests for prompt round splitting and Markdown formatting."""

from __future__ import annotations

import unittest

from app.services.prompt_render import (
    format_messages_markdown_for_prompt,
    select_rounds_for_prompt,
    select_rounds_for_ui,
    split_messages_into_rounds,
)


def _msg(
    mtype: str,
    content: str,
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


class TestSplitAndSelect(unittest.TestCase):
    """Round detection and memory windowing."""

    def test_split_empty(self) -> None:
        self.assertEqual(split_messages_into_rounds([]), [])

    def test_select_prompt_last_complete_round(self) -> None:
        msgs = [
            _msg("query", "q1"),
            _msg("answer", "a1"),
            _msg("query", "q2"),
            _msg("answer", "a2"),
        ]
        flat = select_rounds_for_prompt(msgs, memory_rounds=1)
        self.assertEqual(len(flat), 2)
        self.assertEqual(flat[0]["content"], "q2")

    def test_select_ui_includes_incomplete_tail(self) -> None:
        msgs = [
            _msg("query", "q1"),
            _msg("answer", "a1"),
            _msg("query", "q2"),
        ]
        flat = select_rounds_for_ui(msgs, memory_rounds=10)
        self.assertEqual(len(flat), 3)

    def test_format_markdown_non_empty(self) -> None:
        msgs = [_msg("query", "hi"), _msg("answer", "yo")]
        md = format_messages_markdown_for_prompt(msgs)
        self.assertIn("Turn 1", md)
        self.assertIn("query", md)
        self.assertIn("hi", md)


if __name__ == "__main__":
    unittest.main()
