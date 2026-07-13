"""SDK capability-header tests.

Covers security test plan CLIENT rows:
  TC-52  All four SDKs (Python, JavaScript, C#, GDScript) attach the per-launch
         capability on every request path, and fail closed with a clear error
         when runtime.json lacks api_token.
  TC-51  (static) The TUI local-inference path authenticates the local proxy with
         the local capability and forwards the cloud token only as
         X-Loreguard-Backend-Authorization.

Per the task, the live-SDK parts of TC-52 (which need a running SDK server +
model) are not exercised. The Python SDK is importable, so its fail-closed and
header-attachment behaviour is asserted directly; the JS/C#/GDScript SDKs are not
runnable here and are asserted by source inspection / light parsing.
"""

import ast
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
SDK = REPO / "sdk"

MISSING_MSG = "runtime credential is missing"


def _load_python_sdk():
    path = SDK / "python" / "loreguard_sdk.py"
    spec = importlib.util.spec_from_file_location("loreguard_sdk_undertest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _section(text: str, start: str, end: str | None = None) -> str:
    """Return the slice of `text` from `start` up to `end` (or EOF).

    Raises ValueError (failing the test clearly) if a marker is absent.
    """
    i = text.index(start)
    if end is None:
        return text[i:]
    j = text.index(end, i + len(start))
    return text[i:j]


class PythonSdkCapabilityTest(unittest.TestCase):
    def setUp(self):
        self.sdk = _load_python_sdk()

    def test_get_auth_headers_returns_capability(self):
        with patch.object(self.sdk, "get_runtime_info", return_value={"api_token": "cap-123"}):
            self.assertEqual(
                self.sdk.get_auth_headers(), {"Authorization": "Bearer cap-123"}
            )

    def test_get_auth_headers_fails_closed_without_token(self):
        # Old runtime.json without api_token, and no runtime.json at all.
        for stub in ({}, {"port": 5}, None):
            with self.subTest(runtime=stub):
                with patch.object(self.sdk, "get_runtime_info", return_value=stub):
                    with self.assertRaises(RuntimeError) as ctx:
                        self.sdk.get_auth_headers()
                    self.assertIn(MISSING_MSG, str(ctx.exception))

    def test_all_request_paths_attach_capability(self):
        source = (SDK / "python" / "loreguard_sdk.py").read_text()
        tree = ast.parse(source)
        request_funcs = {"chat", "chat_sync", "get_capabilities", "health_check"}
        found = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in request_funcs:
                calls = {
                    n.func.id
                    for n in ast.walk(node)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                }
                found[node.name] = "get_auth_headers" in calls
        self.assertEqual(set(found), request_funcs, "missing request functions in Python SDK")
        for name, ok in found.items():
            self.assertTrue(ok, f"Python SDK '{name}' does not call get_auth_headers()")


class JavaScriptSdkCapabilityTest(unittest.TestCase):
    def setUp(self):
        self.src = (SDK / "javascript" / "loreguard-sdk.js").read_text()

    def test_get_auth_headers_fails_closed(self):
        helper = _section(self.src, "function getAuthHeaders(", "function isRunning(")
        self.assertIn("!info.api_token", helper)
        self.assertIn("throw new Error(", helper)
        self.assertIn(MISSING_MSG, helper)
        self.assertIn("Bearer ${info.api_token}", helper)

    def test_request_paths_attach_capability(self):
        sections = {
            "chat": _section(self.src, "async function* chat(", "async function chatSimple("),
            "chatSimple": _section(
                self.src, "async function chatSimple(", "async function healthCheck("
            ),
            "healthCheck": _section(self.src, "async function healthCheck(", "// CommonJS exports"),
        }
        for name, body in sections.items():
            with self.subTest(func=name):
                self.assertIn("getAuthHeaders(", body, f"JS '{name}' omits getAuthHeaders()")


class CSharpSdkCapabilityTest(unittest.TestCase):
    def setUp(self):
        self.src = (SDK / "csharp" / "LoreguardSDK.cs").read_text()

    def test_request_paths_attach_capability_and_fail_closed(self):
        sections = {
            "Chat": _section(
                self.src,
                "public static IEnumerator Chat(",
                "public static IEnumerator ChatSimple(",
            ),
            "ChatSimple": _section(
                self.src, "public static IEnumerator ChatSimple(", "// Data classes"
            ),
        }
        for name, body in sections.items():
            with self.subTest(func=name):
                self.assertIn(
                    'SetRequestHeader("Authorization", $"Bearer {runtimeInfo.api_token}")',
                    body,
                    f"C# '{name}' does not attach the capability header",
                )
                self.assertIn(
                    "IsNullOrEmpty(runtimeInfo.api_token)",
                    body,
                    f"C# '{name}' has no fail-closed guard",
                )
                self.assertIn(MISSING_MSG, body)


class GdScriptSdkCapabilityTest(unittest.TestCase):
    def setUp(self):
        self.src = (SDK / "gdscript" / "LoreguardSDK.gd").read_text()

    def test_get_auth_header_fails_closed(self):
        helper = _section(self.src, "static func get_auth_header(", "static func is_running(")
        # No api_token -> empty sentinel, which callers treat as fail-closed.
        self.assertIn('return ""', helper)
        self.assertIn("api_token", helper)
        self.assertIn('Bearer %s" % str(info["api_token"])', helper)

    def test_request_paths_attach_capability_and_fail_closed(self):
        sections = {
            "chat": _section(self.src, "func chat(", "func _on_request_completed("),
            "chat_streaming": _section(self.src, "func _streaming_request(", None),
        }
        for name, body in sections.items():
            with self.subTest(func=name):
                self.assertIn("get_auth_header()", body, f"GDScript '{name}' omits capability")
                self.assertIn(
                    "headers[-1].is_empty()", body, f"GDScript '{name}' has no fail-closed guard"
                )
                self.assertIn(MISSING_MSG, body)


class TuiLocalInferenceHeaderTest(unittest.TestCase):
    """TC-51 (static): local proxy authenticated by capability; cloud token separate."""

    def setUp(self):
        self.src = (REPO / "src" / "tui" / "widgets" / "npc_chat.py").read_text()

    def test_local_request_uses_capability_and_forwards_cloud_token_separately(self):
        # Local proxy authenticates with the per-launch capability...
        self.assertIn('"Authorization": f"Bearer {local_capability}"', self.src)
        # ...and the cloud/worker token is forwarded only as the backend header.
        self.assertIn(
            'local_headers["X-Loreguard-Backend-Authorization"] = f"Bearer {self._api_token}"',
            self.src,
        )
        # The capability must come from runtime.json, not the cloud token.
        self.assertIn("local_capability = get_local_capability()", self.src)

    def test_get_local_capability_reads_runtime_token(self):
        cap = _section(self.src, "def get_local_capability(", "def ")
        self.assertIn("RuntimeInfo.load()", cap)
        self.assertIn("info.api_token", cap)


if __name__ == "__main__":
    unittest.main()
