"""Capability-auth tests for the local SDK/bridge HTTP servers.

Covers security test plan CLIENT rows:
  TC-47  Every non-health endpoint requires the per-launch capability (401).
  TC-48  /health is open (Q2 decision); other endpoints still 401.
  TC-49  A non-ASCII Authorization byte yields 401, not a 500/TypeError.
  TC-50  The local capability is never forwarded to the remote backend;
         the backend credential travels only in X-Loreguard-Backend-Authorization.

The embedded SDK server (src/http_server.py) builds its FastAPI app inside a
background-thread bootstrap (`_run_server`). We drive the *real* app + middleware
through httpx's ASGITransport by short-circuiting uvicorn (fake Config/Server that
capture the app instead of serving), so the middleware runs exactly as in prod.

The bridge worker server (src/main.py) exposes a module-level `app` with the same
capability middleware, tested directly for a second, independent confirmation.
"""

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import uvicorn

from src.http_server import EmbeddedHTTPServer


class FakeTunnel:
    """Minimal stand-in for BackendTunnel used by the SDK server handlers."""

    def __init__(self):
        self.connected = True
        self.capabilities = ["chat"]
        self.backend_url = "wss://console.loreguard.com/workers"
        self.chunk_detector = None
        self.received = None  # kwargs captured from send_chat_request

    async def send_chat_request(self, **kwargs):
        self.received = kwargs
        queue = asyncio.Queue()
        await queue.put(
            {"type": "done", "data": {"speech": "hi", "verified": True, "citations": []}}
        )
        return queue

    def cancel_chat_request(self, request_id):
        pass


def build_sdk_app(tunnel):
    """Instantiate the real EmbeddedHTTPServer FastAPI app without serving.

    Returns (app, server). `server.api_token` is the per-launch capability.
    """
    holder = {}

    class _FakeConfig:
        def __init__(self, app=None, **kwargs):
            self.app = app
            self.kwargs = kwargs

    class _FakeServer:
        def __init__(self, config):
            self.config = config
            holder["app"] = config.app

        def install_signal_handlers(self):
            pass

        async def serve(self, sockets=None):
            return

    tmp = tempfile.mkdtemp(prefix="lg-sdk-app-")

    def _fake_data_dir():
        path = Path(tmp)
        path.mkdir(parents=True, exist_ok=True)
        return path

    server = EmbeddedHTTPServer(tunnel=tunnel, main_loop=None)
    server.actual_port = 12345  # arbitrary; the fake server never binds
    with patch("src.runtime.get_data_dir", side_effect=_fake_data_dir), patch.object(
        uvicorn, "Config", _FakeConfig
    ), patch.object(uvicorn, "Server", _FakeServer):
        server._run_server()

    # _run_server installs a fresh event loop on this thread; close it so later
    # asyncio.run() calls are clean and no ResourceWarning is emitted.
    if server._loop is not None:
        server._loop.close()

    return holder["app"], server


async def _asgi_status(app, path, headers):
    """Call the ASGI app directly with raw header bytes and return the status code.

    Used to feed a raw non-ASCII header byte that httpx's str-header validation
    would otherwise reject.
    """
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": headers + [(b"host", b"local")],
        "server": ("127.0.0.1", 12345),
        "client": ("127.0.0.1", 5555),
    }
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    await app(scope, receive, send)
    start = next(m for m in messages if m["type"] == "http.response.start")
    return start["status"]


async def _chunked_oversize_status(app, path, headers):
    """Send an oversized streaming body without a Content-Length header."""

    async def chunks():
        yield b"x" * (600 * 1024)
        yield b"y" * (600 * 1024)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://local") as client:
        response = await client.post(path, headers=headers, content=chunks())
        return response.status_code


class SDKServerCapabilityAuthTest(unittest.TestCase):
    def setUp(self):
        self.tunnel = FakeTunnel()
        self.app, self.server = build_sdk_app(self.tunnel)
        self.capability = self.server.api_token

    # --- request drivers -----------------------------------------------------

    def _request(self, method, path, *, headers=None, json=None):
        async def _inner():
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://local") as c:
                return await c.request(method, path, headers=headers, json=json)

        return asyncio.run(_inner())

    def _bearer(self, token):
        return {"Authorization": f"Bearer {token}"}

    # --- TC-48: /health open --------------------------------------------------

    def test_health_open_without_token(self):
        resp = self._request("GET", "/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json().get("status"), "ok")

    def test_health_open_even_with_wrong_token(self):
        # /health short-circuits before the capability check.
        resp = self._request("GET", "/health", headers=self._bearer("totally-wrong"))
        self.assertEqual(resp.status_code, 200)

    # --- TC-47: every non-health endpoint is gated ---------------------------

    NON_HEALTH_ENDPOINTS = [
        ("GET", "/api/capabilities"),
        ("GET", "/api/characters"),
        ("GET", "/api/models"),
        ("POST", "/api/chat"),
        ("POST", "/api/admin/reload-model"),
        ("GET", "/openapi.json"),
        ("GET", "/docs"),
        ("OPTIONS", "/api/chat"),
        ("GET", "/definitely-not-a-route"),
    ]

    def test_non_health_endpoints_require_token(self):
        for method, path in self.NON_HEALTH_ENDPOINTS:
            with self.subTest(endpoint=f"{method} {path}", token="missing"):
                resp = self._request(method, path)
                self.assertEqual(
                    resp.status_code, 401, f"{method} {path} should be 401 without a token"
                )
            with self.subTest(endpoint=f"{method} {path}", token="wrong"):
                resp = self._request(method, path, headers=self._bearer("nope-" + "x" * 40))
                self.assertEqual(
                    resp.status_code, 401, f"{method} {path} should be 401 with a wrong token"
                )

    def test_valid_capability_passes_middleware(self):
        # The gate must accept the correct per-launch capability (not deny-all).
        resp = self._request(
            "GET", "/api/capabilities", headers=self._bearer(self.capability)
        )
        self.assertEqual(resp.status_code, 200)

    def test_chunked_body_is_capped_before_handler_buffering(self):
        status = asyncio.run(
            _chunked_oversize_status(
                self.app,
                "/api/chat",
                {
                    "Authorization": f"Bearer {self.capability}",
                    "Content-Type": "application/json",
                },
            )
        )
        self.assertEqual(status, 413)
        self.assertIsNone(self.tunnel.received)

    def test_gate_runs_before_routing(self):
        # A wrong token on a nonexistent path returns 401 (auth), never 404.
        resp = self._request("GET", "/definitely-not-a-route", headers=self._bearer("bad"))
        self.assertEqual(resp.status_code, 401)

    # --- TC-49: non-ASCII Authorization byte ---------------------------------

    def test_non_ascii_authorization_byte_yields_401_not_500(self):
        # Raw 0xFF byte in the Authorization header. Without the byte-compare fix
        # secrets.compare_digest would raise TypeError -> 500; with it, a clean 401.
        async def _inner():
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://local") as c:
                return await c.get(
                    "/api/capabilities", headers=[(b"authorization", b"Bearer \xff")]
                )

        resp = asyncio.run(_inner())
        self.assertEqual(resp.status_code, 401)
        self.assertNotEqual(resp.status_code, 500)

        # Also confirm via a direct ASGI call (guarantees the raw byte reaches
        # the middleware regardless of any client-side header validation).
        status = asyncio.run(
            _asgi_status(self.app, "/api/capabilities", [(b"authorization", b"Bearer \x80\xfe")])
        )
        self.assertEqual(status, 401)

    # --- TC-50: local capability never leaves the machine --------------------

    def test_chat_forwards_backend_credential_only(self):
        resp = self._request(
            "POST",
            "/api/chat",
            headers={
                "Authorization": f"Bearer {self.capability}",
                "X-Loreguard-Backend-Authorization": "Bearer backend-secret",
            },
            json={"character_id": "npc", "message": "hi"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIsNotNone(self.tunnel.received)
        forwarded = self.tunnel.received.get("api_token")
        # The outbound credential is the backend header value...
        self.assertEqual(forwarded, "backend-secret")
        # ...and is NEVER the local per-launch capability.
        self.assertNotEqual(forwarded, self.capability)

    def test_chat_omits_credential_when_backend_header_absent(self):
        # Local capability present, no backend header -> nothing forwarded.
        resp = self._request(
            "POST",
            "/api/chat",
            headers={"Authorization": f"Bearer {self.capability}"},
            json={"character_id": "npc", "message": "hi"},
        )
        self.assertEqual(resp.status_code, 200)
        forwarded = self.tunnel.received.get("api_token")
        self.assertEqual(forwarded, "")  # not substituted with the local capability
        self.assertNotEqual(forwarded, self.capability)

    def test_chat_handler_source_uses_backend_header(self):
        # Static guard: the outbound credential is read from the backend header,
        # and no handler builds it from the local Authorization header.
        src = Path("src/http_server.py").read_text()
        self.assertIn(
            'auth_header = request.headers.get("x-loreguard-backend-authorization", "")', src
        )
        self.assertIn(
            'api_token = auth_header.replace("Bearer ", "") '
            'if auth_header.startswith("Bearer ") else ""',
            src,
        )
        # The local capability must not be repurposed as the outbound api_token.
        self.assertNotIn('api_token = request.headers.get("authorization"', src)


class BridgeServerCapabilityAuthTest(unittest.TestCase):
    """Same capability gate on the bridge worker server (src/main.py app)."""

    @classmethod
    def setUpClass(cls):
        import src.main as bridge

        cls.app = bridge.app
        cls.capability = bridge.local_api_token

    def _request(self, method, path, *, headers=None, json=None):
        async def _inner():
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://local") as c:
                return await c.request(method, path, headers=headers, json=json)

        return asyncio.run(_inner())

    def test_health_open_without_token(self):  # TC-48
        resp = self._request("GET", "/health")
        self.assertEqual(resp.status_code, 200)

    def test_non_health_endpoints_require_token(self):  # TC-47
        for method, path in [
            ("GET", "/models"),
            ("POST", "/api/chat"),
            ("GET", "/openapi.json"),
            ("GET", "/nope"),
        ]:
            with self.subTest(endpoint=f"{method} {path}"):
                self.assertEqual(self._request(method, path).status_code, 401)
                self.assertEqual(
                    self._request(
                        method, path, headers={"Authorization": "Bearer wrong-" + "x" * 40}
                    ).status_code,
                    401,
                )

    def test_valid_capability_passes(self):  # TC-47
        resp = self._request(
            "GET", "/models", headers={"Authorization": f"Bearer {self.capability}"}
        )
        # 401 would mean the gate rejected a valid capability. /models may return
        # 200 or a non-auth error depending on LLM state; it must not be 401.
        self.assertNotEqual(resp.status_code, 401)

    def test_chunked_body_is_capped_before_pydantic_buffering(self):
        status = asyncio.run(
            _chunked_oversize_status(
                self.app,
                "/api/chat",
                {
                    "Authorization": f"Bearer {self.capability}",
                    "Content-Type": "application/json",
                },
            )
        )
        self.assertEqual(status, 413)

    def test_non_ascii_authorization_byte_yields_401_not_500(self):  # TC-49
        status = asyncio.run(
            _asgi_status(self.app, "/models", [(b"authorization", b"Bearer \xff")])
        )
        self.assertEqual(status, 401)


if __name__ == "__main__":
    unittest.main()
