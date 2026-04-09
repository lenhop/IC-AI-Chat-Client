"""Dependency-boundary tests for Gradio UI module imports."""

from __future__ import annotations

from pathlib import Path
import unittest


class TestGradioChatDependencies(unittest.TestCase):
    """Ensure UI module only uses transport/facade for LLM calls."""

    def test_gradio_chat_does_not_import_call_llm_normalize(self) -> None:
        file_path = Path(__file__).resolve().parents[1] / "app" / "ui" / "gradio_chat.py"
        source = file_path.read_text(encoding="utf-8")
        # Keep UI decoupled from low-level LLM implementation details.
        self.assertNotIn("from app.services.call_llm import normalize_messages", source)
        self.assertIn("from app.services.llm_transport import", source)
        self.assertIn("validate_or_normalize_messages", source)


if __name__ == "__main__":
    unittest.main()
