"""
Shared normalization for Gradio Chatbot ``messages`` rows.

``GradioLayoutService`` and ``GradioHandlerService`` both need the same
last-mile coercion so malformed rows from Gradio callbacks do not break
the UI or downstream API calls. Keep a single implementation here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def normalize_chat_row(item: Any) -> Optional[Dict[str, str]]:
    """
    Normalize one history item to Gradio ``messages`` shape ``{role, content}``.

    Args:
        item: A dict-like row or object with ``role`` / ``content`` attributes.

    Returns:
        A valid row dict, or ``None`` if the role is not allowed or data is unusable.
    """
    role: Any = None
    content: Any = None
    if isinstance(item, dict):
        role = item.get("role")
        content = item.get("content")
    else:
        role = getattr(item, "role", None)
        content = getattr(item, "content", None)

    role_text = str(role or "").strip()
    if role_text not in {"system", "user", "assistant"}:
        return None
    return {"role": role_text, "content": str(content or "")}


def normalize_chat_history(history: Any) -> List[Dict[str, str]]:
    """
    Normalize any history-like value to a list of safe Chatbot message rows.

    Args:
        history: List of rows or another iterable; non-lists are treated as empty.

    Returns:
        Only rows that pass :func:`normalize_chat_row`.
    """
    rows = history if isinstance(history, list) else (history or [])
    out: List[Dict[str, str]] = []
    for row in rows:
        item = normalize_chat_row(row)
        if item is not None:
            out.append(item)
    return out
