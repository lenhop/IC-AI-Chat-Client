"""
Development entry for FastAPI.

Run:
    /opt/miniconda3/bin/python scripts/run.py
"""

from __future__ import annotations

import os

import uvicorn


if __name__ == "__main__":
    host = (os.getenv("UVICORN_HOST") or "0.0.0.0").strip()
    port_raw = (os.getenv("UVICORN_PORT") or "8000").strip()
    try:
        port = int(port_raw)
    except ValueError:
        port = 8000

    # Key point: import string enables reload mode to restart child processes correctly.
    uvicorn.run("app.main:app", host=host, port=port, reload=True)
