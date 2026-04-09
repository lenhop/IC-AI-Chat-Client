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
    turn_id: str = "",
) -> dict:
    return {
        "user_id": "u",
        "session_id": "s",
        "type": mtype,
        "content": content,
        "timestamp": ts,
        "turn_id": turn_id,
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

    def test_split_by_turn_id_groups_multi_query_round(self) -> None:
        """Same turn_id: multiple query rows stay one round (v3.2)."""
        tid = "11111111-1111-1111-1111-111111111111"
        msgs = [
            _msg("query", "q1", turn_id=tid),
            _msg("query", "q2", turn_id=tid),
            _msg("answer", "a1", turn_id=tid),
        ]
        rounds = split_messages_into_rounds(msgs)
        self.assertEqual(len(rounds), 1)
        self.assertEqual(len(rounds[0]), 3)

    def test_split_mixed_turn_id_falls_back_to_query_boundaries(self) -> None:
        """Blank ``turn_id`` on any row => historical compatibility query-started rounds."""
        tid = "22222222-2222-2222-2222-222222222222"
        msgs = [
            _msg("query", "q1", turn_id=""),
            _msg("answer", "a1", turn_id=""),
            _msg("query", "q2", turn_id=tid),
        ]
        rounds = split_messages_into_rounds(msgs)
        self.assertEqual(len(rounds), 2)
        self.assertEqual(rounds[1][0]["content"], "q2")


if __name__ == "__main__":
    unittest.main()
