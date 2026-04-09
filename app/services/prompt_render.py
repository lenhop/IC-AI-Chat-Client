"""
Load chat_prompt.md and render {current_query} / {historical_message}.

Historical block is Markdown built from normalized session messages (type, content, timestamp).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _all_messages_have_nonempty_turn_id(messages: List[Dict[str, Any]]) -> bool:
    """True when every message carries a non-blank ``turn_id`` (M3 v3.2 grouping)."""
    if not messages:
        return False
    for m in messages:
        tid = str(m.get("turn_id") or "").strip()
        if not tid:
            return False
    return True

CHAT_PROMPT_FILENAME = "chat_prompt.md"


def _prompt_file_path() -> Path:
    """Resolve chat_prompt.md next to this module."""
    return Path(__file__).resolve().parent / CHAT_PROMPT_FILENAME


def load_chat_prompt_template() -> str:
    """
    Read UTF-8 template from disk; fail fast if missing.

    Returns:
        Raw template string with placeholders.

    Raises:
        RuntimeError: If file is missing or empty.
    """
    path = _prompt_file_path()
    if not path.is_file():
        raise RuntimeError(f"Chat prompt template missing: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Cannot read chat prompt template: {path}") from exc
    if not text.strip():
        raise RuntimeError(f"Chat prompt template is empty: {path}")
    return text


def split_messages_into_rounds(messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Split normalized messages into conversation rounds.

    When every row has a non-empty ``turn_id``, group consecutive rows that share
    the same ``turn_id`` (multi-query clarifications stay one round). Otherwise
    fall back to historical-data compatibility behavior: each new ``type=query`` starts a round.

    Args:
        messages: Normalized records (ordered).

    Returns:
        List of rounds; each round is a non-empty list of message dicts.
    """
    if not messages:
        return []
    if _all_messages_have_nonempty_turn_id(messages):
        rounds: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = [messages[0]]
        cur_tid = str(messages[0].get("turn_id") or "").strip()
        for m in messages[1:]:
            tid = str(m.get("turn_id") or "").strip()
            if tid == cur_tid:
                current.append(m)
            else:
                rounds.append(current)
                current = [m]
                cur_tid = tid
        rounds.append(current)
        return rounds
    rounds_q: List[List[Dict[str, Any]]] = []
    current_q: List[Dict[str, Any]] = []
    for m in messages:
        mtype = (m.get("type") or "").strip()
        if mtype == "query" and current_q:
            rounds_q.append(current_q)
            current_q = []
        current_q.append(m)
    if current_q:
        rounds_q.append(current_q)
    return rounds_q


def _round_is_complete_for_prompt(round_msgs: List[Dict[str, Any]]) -> bool:
    """A round is complete if it contains an ``answer`` (assistant reply stored)."""
    return any((m.get("type") or "") == "answer" for m in round_msgs)


def select_rounds_for_prompt(
    messages: List[Dict[str, Any]],
    memory_rounds: int,
) -> List[Dict[str, Any]]:
    """
    Keep only complete rounds, then take the last ``memory_rounds`` rounds (flattened).

    Args:
        messages: Normalized session messages from Redis (no pending user line).
        memory_rounds: Max completed rounds; ``<= 0`` means all complete rounds.

    Returns:
        Flat list of messages to format into Markdown.
    """
    rounds = split_messages_into_rounds(messages)
    complete = [r for r in rounds if _round_is_complete_for_prompt(r)]
    if memory_rounds <= 0:
        chosen = complete
    else:
        chosen = complete[-memory_rounds:]
    out: List[Dict[str, Any]] = []
    for r in chosen:
        out.extend(r)
    return out


def select_rounds_for_ui(
    messages: List[Dict[str, Any]],
    memory_rounds: int,
) -> List[Dict[str, Any]]:
    """
    Last ``memory_rounds`` rounds for UI hydrate (includes incomplete trailing round).

    Args:
        messages: Normalized list.
        memory_rounds: ``<= 0`` means entire list.

    Returns:
        Flattened messages for Gradio / display mapping.
    """
    rounds = split_messages_into_rounds(messages)
    if memory_rounds <= 0:
        chosen = rounds
    else:
        chosen = rounds[-memory_rounds:]
    out: List[Dict[str, Any]] = []
    for r in chosen:
        out.extend(r)
    return out


def format_messages_markdown_for_prompt(messages: List[Dict[str, Any]]) -> str:
    """
    Format selected messages as Markdown (type, timestamp, content per line).

    Args:
        messages: Flat list of normalized dicts.

    Returns:
        Markdown string (empty if no messages).
    """
    if not messages:
        return "(No prior messages in this session.)"
    lines: List[str] = ["## Historical Conversation", ""]
    rounds = split_messages_into_rounds(messages)
    for idx, round_msgs in enumerate(rounds, start=1):
        ts = (round_msgs[0].get("timestamp") or "").strip() or "—"
        lines.append(f"### Turn {idx} (started at {ts})")
        lines.append("")
        for m in round_msgs:
            mtype = (m.get("type") or "unknown").strip()
            content = (m.get("content") or "").strip() or "—"
            mts = (m.get("timestamp") or "").strip() or "—"
            lines.append(f"- **type:** `{mtype}`  **at:** {mts}")
            lines.append(f"  **content:** {content}")
        lines.append("")
        if idx < len(rounds):
            lines.append("---")
            lines.append("")
    return "\n".join(lines).strip()


def render_chat_prompt(*, current_query: str, historical_message: str) -> str:
    """
    Fill template placeholders (simple replace; order-safe if keys unique).

    Args:
        current_query: Latest user text.
        historical_message: Markdown block from format_messages_markdown_for_prompt.

    Returns:
        Full prompt string for a single user message to the LLM.
    """
    template = load_chat_prompt_template()
    # Replace longer key first if we ever add overlapping names.
    out = template.replace("{historical_message}", historical_message)
    out = out.replace("{current_query}", current_query)
    return out.strip()
