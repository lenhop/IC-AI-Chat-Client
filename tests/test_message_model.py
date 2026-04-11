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
        self.assertIn("## Answer", row["content"])
        self.assertIn("done", row["content"])

    def test_answer_body_heading_is_downgraded_for_readability(self) -> None:
        row = GradioMessageFormatter.to_chat_row(_canon("answer", "## Section\n\nbody"))
        assert row is not None
        self.assertIn("### Section", row["content"])
        self.assertNotIn("\n## Section", row["content"])

    def test_classification_no_ordered_list_marker(self) -> None:
        row = GradioMessageFormatter.to_chat_row(_canon("classification", "intent=x"))
        assert row is not None
        self.assertNotRegex(row["content"], re.compile(r"^\s*\d+\.\s", re.MULTILINE))
        self.assertIn("Classification time:", row["content"])

    def test_dispatcher_section(self) -> None:
        row = GradioMessageFormatter.to_chat_row(_canon("dispatcher", "step"))
        assert row is not None
        self.assertIn("Dispatcher", row["content"])
        self.assertIn("Plan build:", row["content"])

    def test_rewritten_template_shows_backend_status_and_time(self) -> None:
        msg = _canon("rewriting", "normalized query")
        msg["metadata"] = {
            "integrate_rounds": "2",
            "historical_text_length": "120",
            "normalize_status": "Completed",
            "rewrite_backend": "deepseek",
            "rewrite_time_ms": "2659",
        }
        row = GradioMessageFormatter.to_chat_row(msg)
        assert row is not None
        self.assertIn("Integrate short-term memory: 2 rounds (text length: 120 chars)", row["content"])
        self.assertIn("Normalize: Completed", row["content"])
        self.assertIn("Rewrite backend: deepseek", row["content"])
        self.assertIn("Rewrite time: 2659 ms", row["content"])

    def test_classification_template_shows_workflow_and_time(self) -> None:
        msg = _canon("classification", "sp_api")
        msg["metadata"] = {
            "intent_input": "get sku status",
            "workflow": "sp_api",
            "classification_result": "sp_api",
            "classification_time_ms": "18",
        }
        row = GradioMessageFormatter.to_chat_row(msg)
        assert row is not None
        self.assertIn("Intent classification list: get sku status", row["content"])
        self.assertIn("Workflow: sp_api", row["content"])
        self.assertIn("Intent classification result: sp_api", row["content"])
        self.assertIn("Classification time: 18 ms", row["content"])

    def test_dispatcher_template_shows_plan_metrics(self) -> None:
        msg = _canon("dispatcher", "task detail fallback")
        msg["metadata"] = {
            "plan_build_ms": "2",
            "execute_plan_ms": "2529",
            "plan_type": "single_domain",
            "task_groups": "1",
            "planned_tasks": "1",
            "results_completed": "1",
            "results_failed": "0",
            "results_skipped": "0",
            "task_detail": "Task 1: workflow sp_api completed",
        }
        row = GradioMessageFormatter.to_chat_row(msg)
        assert row is not None
        self.assertIn("Plan build: 2 ms", row["content"])
        self.assertIn("Execute plan (workers): 2529 ms", row["content"])
        self.assertIn("Plan type: single_domain", row["content"])
        self.assertIn("Results: 1 completed, 0 failed, 0 skipped", row["content"])

    def test_clarification_includes_backend_status_time(self) -> None:
        msg = _canon("clarification", "please provide details")
        msg["metadata"] = {
            "clarification_backend": "deepseek",
            "clarification_status": "Complete",
            "clarification_time_ms": "12",
        }
        row = GradioMessageFormatter.to_chat_row(msg)
        assert row is not None
        self.assertIn("Clarification backend: deepseek", row["content"])
        self.assertIn("Clarification status: Complete", row["content"])
        self.assertIn("Clarification time: 12 ms", row["content"])
        self.assertIn("please provide details", row["content"])

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
