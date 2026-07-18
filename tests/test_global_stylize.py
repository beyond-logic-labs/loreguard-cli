"""Tests for the tier-3 global stylize handler (engine ADR-0037/0038).

The volunteer lane's client half: global_stylize_request -> containment-safe
prompt -> LLMProxy.generate -> global_stylize_response. Server-side
verification stays the enforcement point; these tests pin the client contract:
response envelope shape, fact-only prompting, output sanitation, and the
volunteer-mode capability pin.
"""

import asyncio
import json
import os
from unittest.mock import AsyncMock, patch

from src.tunnel import STYLIZE_DEFAULT_MAX_CHARS, BackendTunnel


def _make_tunnel(generate_result=None, generate_exc=None) -> BackendTunnel:
    tunnel = BackendTunnel(
        backend_url="wss://example.invalid/workers",
        llm_proxy=AsyncMock(),
        worker_id="worker-test",
        worker_token="lgv_test-token",
    )
    tunnel.llm_proxy.endpoint = "http://localhost:8080"
    if generate_exc is not None:
        tunnel.llm_proxy.generate = AsyncMock(side_effect=generate_exc)
    else:
        tunnel.llm_proxy.generate = AsyncMock(return_value=generate_result or {})
    tunnel._send = AsyncMock()
    return tunnel


def _request(payload: dict) -> dict:
    return {
        "id": "msg-123",
        "type": "global_stylize_request",
        "traceId": "trace-abc",
        "payload": payload,
    }


def _sent_message(tunnel: BackendTunnel) -> dict:
    assert tunnel._send.await_count == 1
    return tunnel._send.await_args.args[0]


# ---------------------------------------------------------------------------
# Handler: response envelope
# ---------------------------------------------------------------------------


def test_handler_success_envelope():
    tunnel = _make_tunnel({"content": "A quiet update from the station."})
    data = _request({
        "job_id": "job-1",
        "content_kind": "forum_post",
        "facts": [{"key": "topic", "value": "the station is quiet"}],
    })

    asyncio.run(tunnel._handle_global_stylize_request(data))

    msg = _sent_message(tunnel)
    assert msg["type"] == "global_stylize_response"
    assert msg["senderId"] == "worker-test"
    assert msg["traceId"] == "trace-abc"
    payload = msg["payload"]
    assert payload["job_id"] == "job-1"
    assert payload["request_id"] == "msg-123"
    assert payload["text"] == "A quiet update from the station."
    assert payload["worker_id"] == "worker-test"
    assert "error" not in payload


def test_handler_llm_error_sets_error_field():
    tunnel = _make_tunnel({"error": "Could not connect to LLM server", "content": ""})
    asyncio.run(tunnel._handle_global_stylize_request(_request({"job_id": "job-2"})))

    payload = _sent_message(tunnel)["payload"]
    assert payload["error"] == "Could not connect to LLM server"
    assert payload["text"] == ""


def test_handler_exception_sets_error_field():
    tunnel = _make_tunnel(generate_exc=RuntimeError("kaput"))
    asyncio.run(tunnel._handle_global_stylize_request(_request({"job_id": "job-3"})))

    payload = _sent_message(tunnel)["payload"]
    assert "kaput" in payload["error"]
    assert payload["text"] == ""


def test_handler_empty_output_is_an_error():
    tunnel = _make_tunnel({"content": "   \n  "})
    asyncio.run(tunnel._handle_global_stylize_request(_request({"job_id": "job-4"})))

    payload = _sent_message(tunnel)["payload"]
    assert payload["error"] == "empty stylize output"
    assert payload["text"] == ""


def test_handler_missing_job_id_sends_nothing():
    tunnel = _make_tunnel({"content": "text"})
    asyncio.run(tunnel._handle_global_stylize_request(_request({})))

    assert tunnel._send.await_count == 0
    assert tunnel.llm_proxy.generate.await_count == 0


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_stylize_prompt_carries_facts_and_constraints():
    tunnel = _make_tunnel()
    request = tunnel._stylize_to_llm_request({
        "job_id": "job-5",
        "content_kind": "forum_post",
        "facts": [
            {"key": "author", "value": "moss_witch"},
            {"key": "topic", "value": "signal tower went dark at 2300"},
        ],
        "allowed_entities": ["Meridian Station"],
        "style_hints": "casual, lowercase",
        "max_output_chars": 400,
        "timeout_ms": 5000,
    })

    system = request["messages"][0]["content"]
    user = request["messages"][1]["content"]
    assert "ONLY the facts" in system
    assert "moss_witch" in user
    assert "signal tower went dark at 2300" in user
    assert "Meridian Station" in user
    assert "casual, lowercase" in user
    assert "forum post" in user
    assert "400 characters" in user
    assert request["max_tokens"] == 200
    assert request["timeout"] == 5.0
    assert request["disable_thinking"] is True
    assert request["require_content"] is True


def test_stylize_prompt_defaults_mirror_engine_cap():
    tunnel = _make_tunnel()
    request = tunnel._stylize_to_llm_request({"job_id": "job-6"})

    user = request["messages"][1]["content"]
    assert f"{STYLIZE_DEFAULT_MAX_CHARS} characters" in user
    assert request["max_tokens"] == 1024
    assert "timeout" not in request


# ---------------------------------------------------------------------------
# Output sanitation
# ---------------------------------------------------------------------------


def test_sanitize_normalizes_newlines_and_control_chars():
    tunnel = _make_tunnel()
    assert tunnel._sanitize_stylize_output("a\r\nb\rc", 0) == "a\nb\nc"
    assert tunnel._sanitize_stylize_output("a\x00b\x7fc", 0) == "abc"
    assert tunnel._sanitize_stylize_output("keep\ttabs\nand lines", 0) == "keep\ttabs\nand lines"


def test_sanitize_unwraps_fences_and_quotes():
    tunnel = _make_tunnel()
    assert tunnel._sanitize_stylize_output("```\nplain text\n```", 0) == "plain text"
    assert tunnel._sanitize_stylize_output("```text\nfenced body\n```", 0) == "fenced body"
    assert tunnel._sanitize_stylize_output('"quoted body"', 0) == "quoted body"
    assert tunnel._sanitize_stylize_output("“curly quoted”", 0) == "curly quoted"


def test_sanitize_trims_to_cap_at_a_boundary():
    tunnel = _make_tunnel()
    text = "First sentence here. Second sentence that runs long past the cap."
    result = tunnel._sanitize_stylize_output(text, 40)
    assert len(result) <= 40
    assert result == "First sentence here. Second sentence"

    short = tunnel._sanitize_stylize_output("no trimming needed", 40)
    assert short == "no trimming needed"


def test_sanitize_empty_stays_empty():
    tunnel = _make_tunnel()
    assert tunnel._sanitize_stylize_output("", 0) == ""
    assert tunnel._sanitize_stylize_output("  \n ", 0) == ""


# ---------------------------------------------------------------------------
# Volunteer mode capability pin
# ---------------------------------------------------------------------------


class _FakeWS:
    def __init__(self, ack: dict):
        self.sent = []
        self._ack = json.dumps(ack)

    async def send(self, message: str):
        self.sent.append(message)

    async def recv(self):
        return self._ack


def _register(tunnel: BackendTunnel) -> dict:
    tunnel.ws = _FakeWS({"type": "worker_ack", "payload": {"accepted": True}})
    ok, reason = asyncio.run(tunnel._register_worker())
    assert ok, reason
    return json.loads(tunnel.ws.sent[0])


def test_volunteer_mode_pins_capabilities_to_completion():
    tunnel = _make_tunnel()
    with patch.dict(os.environ, {"LOREGUARD_VOLUNTEER": "1"}):
        registration = _register(tunnel)
    assert registration["payload"]["worker"]["capabilities"] == ["completion"]


def test_default_mode_keeps_chat_and_completion():
    tunnel = _make_tunnel()
    with patch.dict(os.environ, {"LOREGUARD_VOLUNTEER": "0"}):
        registration = _register(tunnel)
    assert registration["payload"]["worker"]["capabilities"] == ["chat", "completion"]
