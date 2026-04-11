"""Message ingress services for UI endpoint forwarding in v3.5."""

from __future__ import annotations

import json
import logging
from json import JSONDecodeError
from time import perf_counter
from typing import Any, Dict

import httpx
from pydantic import BaseModel, Field

from app.config import AppConfig, get_config
from app.memory.redis_runtime import get_redis_for_gradio
from app.memory.session_store import SessionStore
from app.messages.message_envelope import MessageEnvelope

logger = logging.getLogger(__name__)


class MessageIngressResult(BaseModel):
    """Stable ingress response payload (never returns null fields)."""

    status: str = "ok"
    message: str = ""
    message_id: str
    session_id: str
    turn_id: str
    type: str
    forwarded: bool = False
    stored: bool = False
    downstream: Dict[str, Any] = Field(default_factory=dict)


class MessageIngressService:
    """Facade for UI ingress handling, forwarding, and persistence."""
    INVALID_SSE_FRAME_LIMIT = 3

    @classmethod
    def _log_path(
        cls,
        *,
        event: str,
        envelope: MessageEnvelope,
        target: str,
        detail: str,
        status_code: int = 0,
        latency_ms: int = 0,
        stored: bool = False,
    ) -> None:
        """Emit structured log lines required by v3.5 traceability."""
        logger.info(
            (
                "event=%s message_id=%s session_id=%s turn_id=%s type=%s "
                "source=%s target=%s status_code=%s latency_ms=%s stored=%s detail=%s"
            ),
            event,
            envelope.message_id,
            envelope.session_id,
            envelope.turn_id,
            envelope.type,
            envelope.source,
            target,
            status_code,
            latency_ms,
            stored,
            detail,
        )

    @classmethod
    def _store_envelope(cls, envelope: MessageEnvelope, cfg: AppConfig) -> bool:
        """Persist ingress message to Redis when storage is enabled."""
        client, redis_settings = get_redis_for_gradio()
        if redis_settings is None or not redis_settings.enabled or client is None:
            return False

        store = SessionStore(client, redis_settings)
        store.ensure_session_exists(
            envelope.session_id,
            cfg.user_id,
            cfg.llm_backend,
        )
        store.append_memory_message(
            envelope.session_id,
            cfg.user_id,
            message_type=envelope.type,
            content=envelope.content,
            turn_id=envelope.turn_id,
            timestamp=envelope.timestamp,
            metadata=envelope.metadata,
        )
        return True

    @classmethod
    def _build_non_query_result(cls, envelope: MessageEnvelope, stored: bool) -> MessageIngressResult:
        """Create response payload for non-query envelopes without forwarding."""
        return MessageIngressResult(
            message="Non-query message stored without forwarding.",
            message_id=envelope.message_id,
            session_id=envelope.session_id,
            turn_id=envelope.turn_id,
            type=envelope.type,
            forwarded=False,
            stored=stored,
            downstream={},
        )

    @classmethod
    def _persist_downstream_answer(
        cls,
        *,
        downstream_payload: Dict[str, Any],
        cfg: AppConfig,
    ) -> None:
        """Persist downstream answer envelope when payload follows expected schema."""
        answer_payload = downstream_payload.get("downstream", {})
        if not isinstance(answer_payload, dict):
            return
        maybe_answer = answer_payload.get("envelope")
        if not isinstance(maybe_answer, dict):
            return
        try:
            answer_env = MessageEnvelope.model_validate(maybe_answer)
            cls._store_envelope(answer_env, cfg)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ui_ingress answer persistence skipped: %s", exc)

    @classmethod
    def _forward_message(
        cls,
        *,
        url: str,
        timeout_seconds: int,
        api_key: str,
        envelope: MessageEnvelope,
    ) -> Dict[str, Any]:
        """
        Forward one query envelope downstream.

        If target is ``/v1/chat/stream``, call SSE endpoint and normalize to
        one answer envelope JSON payload. Otherwise assume downstream returns
        JSON directly.
        """
        headers = {"Content-Type": "application/json"}
        token = (api_key or "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"

        timeout = httpx.Timeout(
            connect=10.0,
            read=float(timeout_seconds),
            write=10.0,
            pool=10.0,
        )
        stream_path = "/v1/chat/stream"
        if url.rstrip("/").endswith(stream_path):
            stream_headers = dict(headers)
            stream_headers["Accept"] = "text/event-stream"
            payload = {"messages": [{"role": "user", "content": envelope.content}]}
            accumulated = ""
            invalid_sse_frames = 0
            status_code = 200
            try:
                with httpx.Client(timeout=timeout) as client:
                    with client.stream(
                        "POST",
                        url,
                        json=payload,
                        headers=stream_headers,
                    ) as resp:
                        status_code = int(resp.status_code)
                        resp.raise_for_status()
                        for line in resp.iter_lines():
                            if not line or not line.startswith("data: "):
                                continue
                            raw = line[6:].strip()
                            if not raw:
                                continue
                            try:
                                obj = json.loads(raw)
                            except JSONDecodeError as exc:
                                invalid_sse_frames += 1
                                logger.warning(
                                    (
                                        "event=ui_ingress_sse_invalid_frame "
                                        "message_id=%s session_id=%s turn_id=%s "
                                        "invalid_frames=%s raw=%s detail=%s"
                                    ),
                                    envelope.message_id,
                                    envelope.session_id,
                                    envelope.turn_id,
                                    invalid_sse_frames,
                                    raw[:200],
                                    str(exc),
                                )
                                if invalid_sse_frames > cls.INVALID_SSE_FRAME_LIMIT:
                                    raise RuntimeError(
                                        "Downstream SSE contains too many invalid JSON frames."
                                    ) from exc
                                continue
                            if isinstance(obj, dict) and obj.get("error") is not None:
                                raise RuntimeError(str(obj.get("error")))
                            if isinstance(obj, dict) and isinstance(obj.get("delta"), str):
                                accumulated += obj["delta"]
            except httpx.TimeoutException as exc:
                raise RuntimeError(f"Downstream timeout: {exc}") from exc
            except httpx.HTTPStatusError as exc:
                body = exc.response.text[:500]
                raise RuntimeError(f"Downstream HTTP {exc.response.status_code}: {body}") from exc
            except httpx.HTTPError as exc:
                raise RuntimeError(f"Downstream request failed: {exc}") from exc

            answer_env = MessageEnvelope.build_answer(
                query_envelope=envelope,
                answer_text=accumulated,
                source="chat_llm_stream",
                target=envelope.source,
            )
            payload_out = {
                "status": "ok",
                "downstream": {"envelope": answer_env.model_dump(mode="json")},
            }
            return {"payload": payload_out, "status_code": status_code}

        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, json=envelope.model_dump(mode="json"), headers=headers)
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, dict):
                    raise RuntimeError("Downstream response must be a JSON object.")
                return {"payload": data, "status_code": int(resp.status_code)}
        except httpx.TimeoutException as exc:
            raise RuntimeError(f"Downstream timeout: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500]
            raise RuntimeError(f"Downstream HTTP {exc.response.status_code}: {body}") from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Downstream request failed: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError(f"Downstream JSON decode failed: {exc}") from exc

    @classmethod
    def handle_ui_ingress(cls, envelope: MessageEnvelope) -> MessageIngressResult:
        """
        Handle UI ingress with the business rule: only ``type=query`` is forwarded.
        """
        started = perf_counter()
        cfg = get_config()
        stored = cls._store_envelope(envelope, cfg)
        cls._log_path(
            event="ui_ingress_received",
            envelope=envelope,
            target=envelope.target,
            detail=f"stored={stored}",
            stored=stored,
        )
        if envelope.type != "query":
            latency_ms = int((perf_counter() - started) * 1000)
            cls._log_path(
                event="ui_ingress_non_query_done",
                envelope=envelope,
                target=envelope.target,
                detail="forwarded=false",
                status_code=200,
                latency_ms=latency_ms,
                stored=stored,
            )
            return cls._build_non_query_result(envelope, stored)

        target_url = cfg.chat_ui_forward_url
        cls._log_path(
            event="ui_ingress_forward_start",
            envelope=envelope,
            target=target_url,
            detail="forward_query=true",
        )
        started = perf_counter()
        try:
            downstream_result = cls._forward_message(
                url=target_url,
                timeout_seconds=cfg.chat_ui_forward_timeout_seconds,
                api_key=cfg.chat_ui_forward_api_key,
                envelope=envelope,
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = int((perf_counter() - started) * 1000)
            cls._log_path(
                event="ui_ingress_forward_failed",
                envelope=envelope,
                target=target_url,
                detail=str(exc),
                status_code=502,
                latency_ms=latency_ms,
                stored=stored,
            )
            raise RuntimeError(f"Failed to forward query message: {exc}") from exc
        latency_ms = int((perf_counter() - started) * 1000)
        downstream = downstream_result.get("payload", {})
        status_code = int(downstream_result.get("status_code", 0))

        # Persist downstream answer envelope to keep Gradio chat history complete.
        cls._persist_downstream_answer(downstream_payload=downstream, cfg=cfg)

        cls._log_path(
            event="ui_ingress_forward_done",
            envelope=envelope,
            target=target_url,
            detail="forwarded=true",
            status_code=status_code,
            latency_ms=latency_ms,
            stored=stored,
        )
        return MessageIngressResult(
            message="Query forwarded to downstream service.",
            message_id=envelope.message_id,
            session_id=envelope.session_id,
            turn_id=envelope.turn_id,
            type=envelope.type,
            forwarded=True,
            stored=stored,
            downstream=downstream,
        )

