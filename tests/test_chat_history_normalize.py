"""Unit tests for shared Gradio chat row normalization."""

from __future__ import annotations

import unittest

from app.ui.chat_history_normalize import normalize_chat_history, normalize_chat_row


class ChatHistoryNormalizeTests(unittest.TestCase):
    """Ensure layout and handlers share identical row filtering rules."""

    def test_normalize_chat_row_accepts_standard_roles(self) -> None:
        self.assertEqual(
            normalize_chat_row({"role": "user", "content": "hi"}),
            {"role": "user", "content": "hi"},
        )
        self.assertEqual(
            normalize_chat_row({"role": "assistant", "content": 42}),
            {"role": "assistant", "content": "42"},
        )

    def test_normalize_chat_row_rejects_unknown_roles(self) -> None:
        self.assertIsNone(normalize_chat_row({"role": "tool", "content": "x"}))
        self.assertIsNone(normalize_chat_row({"content": "no role"}))

    def test_normalize_chat_history_filters_mixed_input(self) -> None:
        history = [
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "yes"},
            {"role": "tool", "content": "drop"},
            "bad",
        ]
        self.assertEqual(
            normalize_chat_history(history),
            [
                {"role": "user", "content": "ok"},
                {"role": "assistant", "content": "yes"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
