"""
Gradio Chatbot row formatters for canonical Redis session messages.

Maps each ``type`` (query, answer, clarification, …) to a ``{role, content}`` dict
suitable for ``gr.Chatbot(type='messages')``. Content uses Markdown sections without
ordered-list step numbering (product requirement). Does not read Redis.
"""

from __future__ import annotations

from typing import Any, Dict

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
            return cls._format_answer(content)
        if mtype == "clarification":
            return cls._format_clarification(content)
        if mtype == "rewriting":
            return cls._format_rewriting(content)
        if mtype == "classification":
            return cls._format_classification(content)
        if mtype == "reason":
            return cls._format_reason(content)
        if mtype == "plan":
            return cls._format_plan(content)
        if mtype == "context":
            return cls._format_context(content)
        if mtype == "dispatcher":
            return cls._format_dispatcher(content)
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
    def _format_answer(cls, content: str) -> Dict[str, str]:
        body = content.strip()
        block = f"### Final answer\n\n{body}" if body else "### Final answer\n\n—"
        return {"role": "assistant", "content": block}

    @classmethod
    def _format_clarification(cls, content: str) -> Dict[str, str]:
        body = content.strip() or "—"
        block = f"### Clarification\n\n{body}"
        return {"role": "assistant", "content": block}

    @classmethod
    def _format_rewriting(cls, content: str) -> Dict[str, str]:
        body = content.strip() or "—"
        block = f"### Rewritten query\n\n{body}"
        return {"role": "assistant", "content": block}

    @classmethod
    def _format_classification(cls, content: str) -> Dict[str, str]:
        body = content.strip() or "—"
        block = f"### Classification\n\n**Result**\n\n{body}"
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
    def _format_dispatcher(cls, content: str) -> Dict[str, str]:
        body = content.strip() or "—"
        block = f"### Dispatcher\n\n{body}"
        return {"role": "assistant", "content": block}

    @classmethod
    def _format_fallback(cls, mtype: str, content: str) -> Dict[str, str]:
        body = content.strip() or "—"
        block = f"### {mtype}\n\n{body}"
        return {"role": "assistant", "content": block}
