"""
Gradio Chatbot row formatters for canonical Redis session messages.

Maps each ``type`` (query, answer, clarification, …) to a ``{role, content}`` dict
suitable for ``gr.Chatbot(type='messages')``. Content uses Markdown sections without
ordered-list step numbering (product requirement). Does not read Redis.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from app.config import MessageDisplayOptions


class GradioMessageFormatter:
    """
    Build Gradio chat rows from normalized message dicts.

    Uses classmethods only for a single, consistent entry style (per project AI dev rules).
    """

    @classmethod
    def to_chat_row(cls, message: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Map one canonical message to a Gradio ``type='messages'`` row.

        Args:
            message: Dict with ``type``, ``content``, and optional metadata.

        Returns:
            ``{"role": "user"|"assistant", "content": str}`` or ``None`` if type unknown/empty.

        Example:
            >>> GradioMessageFormatter.to_chat_row({"type": "query", "content": "hi"})
            {'role': 'user', 'content': 'hi'}
        """
        mtype = (message.get("type") or "").strip()
        content = str(message.get("content") or "")
        if not mtype:
            return None
        if mtype == "query":
            return cls._format_query(content)
        if mtype == "answer":
            return cls._format_answer(message)
        if mtype == "clarification":
            return cls._format_clarification(message)
        if mtype == "rewriting":
            return cls._format_rewriting(message)
        if mtype == "classification":
            return cls._format_classification(message)
        if mtype == "reason":
            return cls._format_reason(content)
        if mtype == "plan":
            return cls._format_plan(content)
        if mtype == "context":
            return cls._format_context(content)
        if mtype == "dispatcher":
            return cls._format_dispatcher(message)
        return cls._format_fallback(mtype, content)

    @classmethod
    def should_display_type(cls, message_type: str, display: MessageDisplayOptions) -> bool:
        """
        Whether a message type may appear in the Gradio chat (query/answer always True).

        Args:
            message_type: Normalized ``type`` string.
            display: Parsed display toggles from ``AppConfig``.

        Returns:
            True if the type should be rendered.
        """
        mt = (message_type or "").strip()
        if mt in ("query", "answer"):
            return True
        if mt == "clarification":
            return display.clarification_message_display_enable
        if mt == "rewriting":
            return display.rewriting_message_display_enable
        if mt == "classification":
            return display.classification_message_display_enable
        if mt == "plan":
            return display.plan_message_display_enable
        if mt == "reason":
            return display.reason_message_display_enable
        if mt == "context":
            return display.context_message_display_enable
        if mt == "dispatcher":
            return display.dispatcher_message_display_enable
        return True

    @classmethod
    def _format_query(cls, content: str) -> Dict[str, str]:
        return {"role": "user", "content": content}

    @classmethod
    def _format_answer(cls, message: Dict[str, Any]) -> Dict[str, str]:
        """Render answer content with a stable title."""
        content = str(message.get("content") or "")
        body = cls._downgrade_markdown_headings(content.strip())
        block = f"## Answer\n\n{body}" if body else "## Answer\n\n—"
        return {"role": "assistant", "content": block}

    @classmethod
    def _format_clarification(cls, message: Dict[str, Any]) -> Dict[str, str]:
        """
        Render clarification template with operational diagnostics.

        The clarification card keeps backward compatibility: when metadata fields
        are missing, fallback placeholders are displayed.
        """
        content = str(message.get("content") or "")
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        backend = cls._pick_metadata_text(
            metadata,
            ("clarification_backend", "backend"),
            default="unknown",
        )
        status = cls._pick_metadata_text(
            metadata,
            ("clarification_status", "status"),
            default="unknown",
        )
        time_value = cls._pick_metadata_text(
            metadata,
            ("clarification_time", "clarification_time_ms", "time_ms"),
            default="0 ms",
        )
        time_text = time_value if time_value.lower().endswith("ms") else f"{time_value} ms"
        body = content.strip() or "—"
        block = (
            "## Clarification\n\n"
            f"- Clarification backend: {backend}\n"
            f"- Clarification status: {status}\n"
            f"- Clarification time: {time_text}\n\n"
            f"{body}"
        )
        return {"role": "assistant", "content": block}

    @classmethod
    def _pick_metadata_text(
        cls,
        metadata: Dict[str, Any],
        keys: tuple[str, ...],
        *,
        default: str,
    ) -> str:
        """Return first non-empty metadata string for a key set."""
        for key in keys:
            if key not in metadata:
                continue
            value = str(metadata.get(key) or "").strip()
            if value:
                return value
        return default

    @classmethod
    def _pick_time_text(
        cls,
        metadata: Dict[str, Any],
        keys: tuple[str, ...],
        *,
        default: str,
    ) -> str:
        """Return normalized time text ending with ``ms``."""
        raw = cls._pick_metadata_text(metadata, keys, default=default)
        text = raw.strip()
        if not text:
            return default
        if text.lower().endswith("ms"):
            return text
        return f"{text} ms"

    @classmethod
    def _downgrade_markdown_headings(cls, text: str) -> str:
        """
        Downgrade markdown heading level by one for answer body readability.

        Example:
        - ``## Topic`` -> ``### Topic``
        - ``##### Deep`` -> ``###### Deep``
        """
        if not text:
            return text

        def _replace(match: re.Match[str]) -> str:
            hashes = match.group(1)
            title = match.group(2)
            if len(hashes) >= 6:
                return f"{hashes} {title}"
            return f"{hashes}# {title}"

        return re.sub(r"^(#{1,6})\s+(.+)$", _replace, text, flags=re.MULTILINE)

    @classmethod
    def _format_rewriting(cls, message: Dict[str, Any]) -> Dict[str, str]:
        """Render rewritten output with backend/status/time diagnostics."""
        content = str(message.get("content") or "")
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        rounds = cls._pick_metadata_text(
            metadata,
            ("integrate_rounds", "historical_rounds", "memory_rounds"),
            default="0",
        )
        text_len = cls._pick_metadata_text(
            metadata,
            ("historical_text_length", "text_length_chars", "memory_text_length"),
            default="0",
        )
        normalize_status = cls._pick_metadata_text(
            metadata,
            ("normalize_status", "rewriting_status", "status"),
            default="unknown",
        )
        backend = cls._pick_metadata_text(
            metadata,
            ("rewrite_backend", "rewriting_backend", "backend"),
            default="unknown",
        )
        rewrite_time = cls._pick_time_text(
            metadata,
            ("rewrite_time", "rewrite_time_ms", "rewriting_time_ms", "time_ms"),
            default="0 ms",
        )
        body = content.strip() or "—"
        block = (
            "## Rewritten\n\n"
            f"- Integrate short-term memory: {rounds} rounds (text length: {text_len} chars)\n"
            f"- Normalize: {normalize_status}\n"
            f"- Rewritten query: {body}\n"
            f"- Rewrite backend: {backend}\n"
            f"- Rewrite time: {rewrite_time}"
        )
        return {"role": "assistant", "content": block}

    @classmethod
    def _format_classification(cls, message: Dict[str, Any]) -> Dict[str, str]:
        """Render classification output with workflow and timing fields."""
        content = str(message.get("content") or "")
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        intent_input = cls._pick_metadata_text(
            metadata,
            ("intent_input", "query", "user_query"),
            default=content.strip() or "—",
        )
        workflow = cls._pick_metadata_text(
            metadata,
            ("workflow", "classification_workflow"),
            default="unknown",
        )
        result = cls._pick_metadata_text(
            metadata,
            ("classification_result", "intent_result", "result"),
            default=content.strip() or "unknown",
        )
        cls_time = cls._pick_time_text(
            metadata,
            ("classification_time", "classification_time_ms", "time_ms"),
            default="0 ms",
        )
        block = (
            "## Classification\n\n"
            f"- Intent classification list: {intent_input}\n"
            f"- Workflow: {workflow}\n"
            f"- Intent classification result: {result}\n"
            f"- Classification time: {cls_time}"
        )
        return {"role": "assistant", "content": block}

    @classmethod
    def _format_reason(cls, content: str) -> Dict[str, str]:
        body = content.strip() or "—"
        block = f"### Reasoning\n\n{body}"
        return {"role": "assistant", "content": block}

    @classmethod
    def _format_plan(cls, content: str) -> Dict[str, str]:
        body = content.strip() or "—"
        block = f"### Plan\n\n{body}"
        return {"role": "assistant", "content": block}

    @classmethod
    def _format_context(cls, content: str) -> Dict[str, str]:
        body = content.strip() or "—"
        block = f"### Context summary\n\n{body}"
        return {"role": "assistant", "content": block}

    @classmethod
    def _format_dispatcher(cls, message: Dict[str, Any]) -> Dict[str, str]:
        """Render dispatcher summary with plan metrics and task diagnostics."""
        content = str(message.get("content") or "")
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        plan_build = cls._pick_time_text(
            metadata,
            ("plan_build_ms", "plan_build_time_ms"),
            default="0 ms",
        )
        execute_plan = cls._pick_time_text(
            metadata,
            ("execute_plan_ms", "execute_time_ms"),
            default="0 ms",
        )
        plan_type = cls._pick_metadata_text(metadata, ("plan_type",), default="unknown")
        task_groups = cls._pick_metadata_text(metadata, ("task_groups",), default="0")
        planned_tasks = cls._pick_metadata_text(metadata, ("planned_tasks",), default="0")
        results_completed = cls._pick_metadata_text(
            metadata,
            ("results_completed", "completed"),
            default="0",
        )
        results_failed = cls._pick_metadata_text(
            metadata,
            ("results_failed", "failed"),
            default="0",
        )
        results_skipped = cls._pick_metadata_text(
            metadata,
            ("results_skipped", "skipped"),
            default="0",
        )
        task_detail = cls._pick_metadata_text(
            metadata,
            ("task_detail", "task_details"),
            default=content.strip() or "—",
        )
        block = (
            "## Dispatcher\n\n"
            f"- Plan build: {plan_build}\n"
            f"- Execute plan (workers): {execute_plan}\n"
            f"- Plan type: {plan_type}\n"
            f"- Task groups: {task_groups} | Planned tasks: {planned_tasks}\n"
            f"- Results: {results_completed} completed, {results_failed} failed, {results_skipped} skipped\n"
            f"- Task detail: {task_detail}"
        )
        return {"role": "assistant", "content": block}

    @classmethod
    def _format_fallback(cls, mtype: str, content: str) -> Dict[str, str]:
        body = content.strip() or "—"
        block = f"### {mtype}\n\n{body}"
        return {"role": "assistant", "content": block}
