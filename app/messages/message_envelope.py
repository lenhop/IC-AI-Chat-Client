"""
Unified message envelope schema and validators for v3.5 ingress APIs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class MessageEnvelope(BaseModel):
    """
    Canonical message payload shared by UI and downstream message services.

    Required fields are frozen by the v3.5 implementation plan:
    ``message_id/session_id/turn_id/type/content/source/target/timestamp/metadata``.
    """

    message_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    session_id: str = Field(min_length=1)
    turn_id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    content: str = Field(default="")
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        min_length=1,
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("message_id", "session_id", "turn_id", "type", "source", "target", "timestamp")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        """Normalize required string fields and reject blank values."""
        text = str(value or "").strip()
        if not text:
            raise ValueError("required field cannot be blank")
        return text

    @field_validator("content", mode="before")
    @classmethod
    def _content_to_text(cls, value: Any) -> str:
        """Force content into string for stable transport and storage."""
        if value is None:
            return ""
        return str(value)

    @field_validator("metadata", mode="before")
    @classmethod
    def _metadata_fallback(cls, value: Any) -> Dict[str, Any]:
        """Ensure metadata is always a dictionary."""
        if isinstance(value, dict):
            return value
        return {}

    @classmethod
    def build_answer(
        cls,
        *,
        query_envelope: MessageEnvelope,
        answer_text: str,
        source: str,
        target: str,
    ) -> MessageEnvelope:
        """Construct an answer envelope that keeps session/turn continuity."""
        return cls(
            session_id=query_envelope.session_id,
            turn_id=query_envelope.turn_id,
            type="answer",
            content=answer_text,
            source=source,
            target=target,
            metadata={
                "reply_to_message_id": query_envelope.message_id,
            },
        )

