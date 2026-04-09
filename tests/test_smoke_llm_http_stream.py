#!/usr/bin/env python3
"""LLM HTTP stream smoke utilities and gated live test.

This module serves two purposes:
1) Keep a CLI entry for manual smoke checks against `/v1/chat/stream`.
2) Provide a unittest case that is skipped by default to avoid accidental
   live network calls during `python -m unittest discover`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import unittest
from typing import Dict
from urllib.parse import urlparse

LOGGER = logging.getLogger(__name__)


class SmokeLlmHttpStreamService:
    """Public service interface for LLM SSE smoke checks."""

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        """Build and return a CLI argument parser."""
        parser = argparse.ArgumentParser(
            description="Stream one user line against LLM service SSE."
        )
        parser.add_argument(
            "--url",
            default=(os.getenv("LLM_SERVICE_URL", "http://127.0.0.1:8001").rstrip("/")),
            help="Worker base URL (no path)",
        )
        parser.add_argument(
            "--message",
            default="Say the word ok in one word.",
            help="User message content",
        )
        parser.add_argument(
            "--timeout-seconds",
            type=float,
            default=120.0,
            help="HTTP client timeout in seconds",
        )
        return parser

    @classmethod
    def build_headers(cls) -> Dict[str, str]:
        """Build request headers, including optional bearer token."""
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        api_key = (os.getenv("LLM_SERVICE_API_KEY") or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    @classmethod
    def is_live_smoke_enabled(cls) -> bool:
        """Return True only when explicit env gate is enabled."""
        return (os.getenv("ICAI_RUN_LLM_SMOKE") or "").strip() == "1"

    @classmethod
    def is_url_reachable(cls, base_url: str, timeout_seconds: float = 1.0) -> bool:
        """Check whether host/port from URL accepts TCP connections."""
        parsed = urlparse(base_url)
        host = parsed.hostname
        if not host:
            return False
        # Use URL port when provided, otherwise derive from scheme.
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            with socket.create_connection((host, port), timeout=timeout_seconds):
                return True
        except OSError:
            return False

    @classmethod
    def run_stream(cls, url: str, message: str, timeout_seconds: float = 120.0) -> int:
        """Send one message to `/v1/chat/stream` and print received deltas."""
        endpoint = f"{url.rstrip('/')}/v1/chat/stream"
        payload = {"messages": [{"role": "user", "content": message}]}
        headers = cls.build_headers()
        LOGGER.info("Starting smoke stream request: endpoint=%s", endpoint)
        try:
            import httpx
        except ImportError as exc:
            LOGGER.exception("Missing required dependency: httpx")
            print(f"httpx required: {exc}", file=sys.stderr)
            return 1

        try:
            with httpx.Client(timeout=timeout_seconds) as client:
                with client.stream("POST", endpoint, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        raw_payload = line[6:].strip()
                        try:
                            event = json.loads(raw_payload)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(event, dict) and "delta" in event:
                            sys.stdout.write(str(event.get("delta", "")))
                            sys.stdout.flush()
                        if isinstance(event, dict) and event.get("error"):
                            LOGGER.error("Smoke stream returned error: %s", event.get("error"))
                            print(f"\n[error] {event.get('error')}", file=sys.stderr)
                            return 2
            print()
            return 0
        except Exception as exc:  # Keep full guard for CLI stability.
            LOGGER.exception("Smoke stream request failed")
            print(f"HTTP error: {exc}", file=sys.stderr)
            return 3

    @classmethod
    def main(cls) -> int:
        """CLI entrypoint."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        parser = cls.build_parser()
        args = parser.parse_args()
        return cls.run_stream(args.url, args.message, args.timeout_seconds)


@unittest.skipUnless(
    SmokeLlmHttpStreamService.is_live_smoke_enabled(),
    "Live smoke disabled. Set ICAI_RUN_LLM_SMOKE=1 to enable.",
)
class SmokeLlmHttpStreamLiveTests(unittest.TestCase):
    """Live smoke tests for `/v1/chat/stream` guarded by environment variable."""

    @classmethod
    def setUpClass(cls) -> None:
        """Skip the live test when URL is not reachable."""
        super().setUpClass()
        cls.base_url = (os.getenv("LLM_SERVICE_URL") or "http://127.0.0.1:8001").rstrip("/")
        if not SmokeLlmHttpStreamService.is_url_reachable(cls.base_url):
            raise unittest.SkipTest(f"Smoke target is unreachable: {cls.base_url}")

    def test_live_stream_smoke(self) -> None:
        """Verify live SSE endpoint responds without transport failure."""
        status = SmokeLlmHttpStreamService.run_stream(
            url=self.base_url,
            message="Say the word ok in one word.",
            timeout_seconds=120.0,
        )
        self.assertEqual(status, 0)


if __name__ == "__main__":
    raise SystemExit(SmokeLlmHttpStreamService.main())
