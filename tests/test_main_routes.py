"""Route-level contract tests for main app after legacy decommission."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.main import app


class MainRoutesTests(unittest.TestCase):
    """Verify public route behavior keeps only the Gradio main path."""

    def setUp(self) -> None:
        """Create a lightweight test client for route assertions."""
        self.client = TestClient(app)

    def test_root_redirects_to_gradio(self) -> None:
        """Root path must keep redirecting clients to the primary Gradio UI."""
        response = self.client.get("/", follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers.get("location"), "/gradio")

    def test_gradio_ui_route_is_reachable(self) -> None:
        """Primary UI endpoint stays available after legacy path cleanup."""
        response = self.client.get("/gradio")
        self.assertEqual(response.status_code, 200)

    def test_legacy_routes_are_unavailable(self) -> None:
        """Removed endpoints should consistently return HTTP 404."""
        legacy_paths = (
            ("get", "/legacy", {}),
            ("post", "/api/chat/stream", {"json": {"messages": [{"role": "user", "content": "hi"}]}}),
            ("post", "/api/sessions", {}),
            ("get", "/api/sessions/sid/messages", {}),
            ("post", "/api/sessions/sid/messages", {"json": {"type": "plan", "content": "x", "turn_id": "t"}}),
        )

        for method, path, kwargs in legacy_paths:
            request_fn = getattr(self.client, method)
            response = request_fn(path, **kwargs)
            self.assertEqual(response.status_code, 404, msg=f"{method.upper()} {path} should be 404")


if __name__ == "__main__":
    unittest.main()
