"""UI-facing message ingress route for v3.5."""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, HTTPException

from app.messages.message_envelope import MessageEnvelope
from app.services.message_ingress import MessageIngressResult, MessageIngressService

logger = logging.getLogger(__name__)

router = APIRouter()
MESSAGE_INGRESS_CANONICAL_PATH = "/v1/messages/test"
MESSAGE_INGRESS_LEGACY_PATHS = ("/v1/messages/in", "/v1/messages/receive")


class MessageIngressRouteFacade:
    """Route-layer facade for ingress protocol handling and alias registration."""

    @classmethod
    def resolve_alias_paths(cls, configured_path: str) -> List[str]:
        """Return deduplicated non-canonical paths kept for compatibility."""
        configured = (configured_path or "").strip()
        ordered = list(MESSAGE_INGRESS_LEGACY_PATHS)
        if configured:
            ordered.append(configured)
        aliases: List[str] = []
        for path in ordered:
            if path == MESSAGE_INGRESS_CANONICAL_PATH:
                continue
            if not path.startswith("/"):
                continue
            if path not in aliases:
                aliases.append(path)
        return aliases

    @classmethod
    async def handle_ingress(cls, body: MessageEnvelope) -> MessageIngressResult:
        """Receive envelope and map unexpected failures to stable 502 responses."""
        try:
            return MessageIngressService.handle_ui_ingress(body)
        except Exception as exc:  # noqa: BLE001
            logger.exception("ui ingress failed")
            raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post(MESSAGE_INGRESS_CANONICAL_PATH, response_model=MessageIngressResult)
async def message_ingress_v1(body: MessageEnvelope) -> MessageIngressResult:
    """Canonical v3.5 test ingress endpoint for message envelopes."""
    return await MessageIngressRouteFacade.handle_ingress(body)

