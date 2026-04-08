#!/usr/bin/env python3
"""
Smoke test: POST JSON to an LLM worker ``/v1/chat/stream`` and print SSE deltas.

Requires a running worker (``uvicorn app.llm_service.main:app``) and reachable URL.
Does not start servers; uses httpx only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream one user line against LLM service SSE.")
    parser.add_argument(
        "--url",
        default=os.getenv("LLM_SERVICE_URL", "http://127.0.0.1:8001").rstrip("/"),
        help="Worker base URL (no path)",
    )
    parser.add_argument(
        "--message",
        default="Say the word ok in one word.",
        help="User message content",
    )
    args = parser.parse_args()
    endpoint = f"{args.url}/v1/chat/stream"
    try:
        import httpx
    except ImportError as exc:
        print("httpx required:", exc, file=sys.stderr)
        return 1

    payload = {"messages": [{"role": "user", "content": args.message}]}
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    key = (os.getenv("LLM_SERVICE_API_KEY") or "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"

    try:
        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", endpoint, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    raw = line[6:].strip()
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict) and "delta" in obj:
                        sys.stdout.write(str(obj.get("delta", "")))
                        sys.stdout.flush()
                    if isinstance(obj, dict) and obj.get("error"):
                        print("\n[error]", obj.get("error"), file=sys.stderr)
                        return 2
        print()
        return 0
    except httpx.HTTPError as exc:
        print("HTTP error:", exc, file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
