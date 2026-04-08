"""FastAPI session append route (fakeredis + patched get_config)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import fakeredis
from fastapi import FastAPI
from starlette.testclient import TestClient

from app.config import RedisSettings
from app.memory.session_store import SessionStore
from app.routes.sessions import router


class TestAppendSessionMessage(unittest.TestCase):
    """POST /api/sessions/{id}/messages"""

    def setUp(self) -> None:
        self._client = fakeredis.FakeRedis(decode_responses=True)
        self._rs = RedisSettings(
            enabled=True,
            url="redis://localhost:6379/0",
            key_prefix="api:",
            session_ttl_seconds=3600,
        )
        self._app = FastAPI()
        self._app.state.redis = self._client
        self._app.state.redis_settings = self._rs
        self._app.include_router(router)
        self._tc = TestClient(self._app)

    @patch("app.routes.sessions.get_config")
    def test_append_ok(self, mock_gc) -> None:
        mock_gc.return_value = SimpleNamespace(user_id="u1")
        store = SessionStore(self._client, self._rs)
        sid = store.create_session("u1", "deepseek")
        r = self._tc.post(
            f"/api/sessions/{sid}/messages",
            json={"type": "plan", "content": "step", "turn_id": "tid-uuid-1"},
        )
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json().get("ok"))
        msgs = store.get_messages(sid, "u1")
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].get("type"), "plan")

    @patch("app.routes.sessions.get_config")
    def test_append_forbidden_user(self, mock_gc) -> None:
        mock_gc.return_value = SimpleNamespace(user_id="other")
        store = SessionStore(self._client, self._rs)
        sid = store.create_session("u1", "deepseek")
        r = self._tc.post(
            f"/api/sessions/{sid}/messages",
            json={"type": "plan", "content": "step", "turn_id": "tid-uuid-2"},
        )
        self.assertEqual(r.status_code, 403)


if __name__ == "__main__":
    unittest.main()
