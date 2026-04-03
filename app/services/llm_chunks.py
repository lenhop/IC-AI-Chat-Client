"""
Unified streaming chunks for LLM adapters (project_goal §2.5).

Maps provider-specific stream events to a small common shape so gateways
and M4+ step UIs can consume ``stream_chat_chunks`` without parsing raw SSE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass(frozen=True)
class ChatStreamChunk:
    """
    One logical piece of a chat completion stream.

    Attributes:
        content_delta: Visible assistant text increment (may be empty on meta-only chunks).
        reasoning_delta: Optional chain-of-thought style increment when the API exposes it.
        done: True when the provider stream has finished successfully (after all deltas).
        error: Set when the stream terminates with a logical error (adapters may still raise).
    """

    content_delta: Optional[str] = None
    reasoning_delta: Optional[str] = None
    done: bool = False
    error: Optional[str] = None


def iter_text_deltas(chunks: Iterator[ChatStreamChunk]) -> Iterator[str]:
    """Yield only non-empty ``content_delta`` strings (``stream_chat`` compatibility)."""
    for ch in chunks:
        if ch.content_delta:
            yield ch.content_delta
