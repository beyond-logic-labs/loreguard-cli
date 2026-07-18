"""Tests for Ed25519 dispatch verification (loreguard-engine#87).

The canonical byte vector is shared verbatim with the engine test
(loreguard-engine internal/cloud/workers/dispatch_signing_test.go); if it
changes there it MUST change here, plus a context-version bump.
"""

import base64
import os
from unittest.mock import AsyncMock, patch

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from src.dispatch_verify import (
    PUBKEY_ENV,
    canonical_stylize_signing_bytes,
    iso_to_unix,
    load_pinned_pubkey,
    verify_stylize_dispatch,
)
from src.tunnel import BackendTunnel

VECTOR_PAYLOAD = {
    "job_id": "job-1",
    "content_kind": "forum_post",
    "facts": [{"key": "topic", "value": "café is open"}],
    "allowed_entities": ["Meridian Station"],
    "style_hints": "casual",
    "max_output_chars": 400,
}

# Shared cross-language vector (UTF-8 byte lengths: "café is open" is 13).
CANONICAL_VECTOR = (
    b"lgv-dispatch-v1\x00"
    b"5:msg-1\x00"
    b"1700000000\x00"
    b"11:server-node\x00"
    b"5:job-1\x00"
    b"10:forum_post\x00"
    b"1\x00"
    b"5:topic\x00"
    b"13:caf\xc3\xa9 is open\x00"
    b"1\x00"
    b"16:Meridian Station\x00"
    b"6:casual\x00"
    b"400\x00"
    b"0\x00"
)


def test_canonical_bytes_match_engine_vector():
    got = canonical_stylize_signing_bytes("msg-1", 1700000000, "server-node", VECTOR_PAYLOAD)
    assert got == CANONICAL_VECTOR


def test_iso_to_unix():
    # 2023-11-14T22:13:20Z == 1700000000
    assert iso_to_unix("2023-11-14T22:13:20Z") == 1700000000
    # Go RFC3339Nano fractional seconds (9 digits) and zone offsets parse.
    assert iso_to_unix("2023-11-14T22:13:20.123456789Z") == 1700000000
    assert iso_to_unix("2023-11-15T00:13:20.5+02:00") == 1700000000
    assert iso_to_unix("garbage") is None
    assert iso_to_unix("") is None


def _signed_envelope(priv: Ed25519PrivateKey, payload: dict) -> dict:
    ts = "2023-11-14T22:13:20Z"
    message = canonical_stylize_signing_bytes("msg-1", 1700000000, "server-node", payload)
    sig = priv.sign(message)
    return {
        "id": "msg-1",
        "type": "global_stylize_request",
        "timestamp": ts,
        "senderId": "server-node",
        "signature": "ed25519:" + base64.urlsafe_b64encode(sig).rstrip(b"=").decode(),
        "payload": payload,
    }


def test_verify_round_trip_and_tampering():
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()
    data = _signed_envelope(priv, dict(VECTOR_PAYLOAD))

    ok, reason = verify_stylize_dispatch(pub, data)
    assert ok, reason

    # Tampered fact value.
    tampered = _signed_envelope(priv, dict(VECTOR_PAYLOAD))
    tampered["payload"] = {**VECTOR_PAYLOAD, "facts": [{"key": "topic", "value": "café is closed"}]}
    ok, reason = verify_stylize_dispatch(pub, tampered)
    assert not ok and reason == "signature mismatch"

    # Tampered sender.
    spoofed = _signed_envelope(priv, dict(VECTOR_PAYLOAD))
    spoofed["senderId"] = "attacker"
    assert not verify_stylize_dispatch(pub, spoofed)[0]

    # Missing / malformed signatures.
    unsigned = _signed_envelope(priv, dict(VECTOR_PAYLOAD))
    unsigned["signature"] = ""
    assert verify_stylize_dispatch(pub, unsigned) == (False, "missing or non-ed25519 signature")
    hmac_legacy = _signed_envelope(priv, dict(VECTOR_PAYLOAD))
    hmac_legacy["signature"] = "deadbeef" * 8
    assert not verify_stylize_dispatch(pub, hmac_legacy)[0]

    # Wrong key.
    other_pub = Ed25519PrivateKey.generate().public_key()
    assert not verify_stylize_dispatch(other_pub, _signed_envelope(priv, dict(VECTOR_PAYLOAD)))[0]


def test_load_pinned_pubkey():
    priv = Ed25519PrivateKey.generate()
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    raw = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)

    with patch.dict(os.environ, {PUBKEY_ENV: ""}):
        assert load_pinned_pubkey() is None
    with patch.dict(os.environ, {PUBKEY_ENV: base64.b64encode(raw).decode()}):
        assert load_pinned_pubkey() is not None
    with patch.dict(os.environ, {PUBKEY_ENV: base64.urlsafe_b64encode(raw).rstrip(b"=").decode()}):
        assert load_pinned_pubkey() is not None
    with patch.dict(os.environ, {PUBKEY_ENV: "not-base64!!"}):
        with pytest.raises(ValueError):
            load_pinned_pubkey()
    with patch.dict(os.environ, {PUBKEY_ENV: base64.b64encode(b"short").decode()}):
        with pytest.raises(ValueError):
            load_pinned_pubkey()


# ---------------------------------------------------------------------------
# Tunnel integration
# ---------------------------------------------------------------------------


def _make_tunnel(**kwargs) -> BackendTunnel:
    tunnel = BackendTunnel(
        backend_url=kwargs.pop("backend_url", "wss://example.invalid/workers"),
        llm_proxy=AsyncMock(),
        worker_id="worker-test",
        worker_token="lgv_test-token",
        **kwargs,
    )
    tunnel.llm_proxy.endpoint = "http://localhost:8080"
    tunnel.llm_proxy.generate = AsyncMock(return_value={"content": "styled text"})
    tunnel._send = AsyncMock()
    return tunnel


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def test_handler_rejects_unsigned_when_pinned():
    priv = Ed25519PrivateKey.generate()
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    pin = base64.b64encode(priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()

    tunnel = _make_tunnel()
    unsigned = {
        "id": "msg-1",
        "timestamp": "2023-11-14T22:13:20Z",
        "senderId": "server-node",
        "traceId": "t",
        "payload": dict(VECTOR_PAYLOAD),
    }
    with patch.dict(os.environ, {PUBKEY_ENV: pin}):
        _run(tunnel._handle_global_stylize_request(unsigned))

    assert tunnel._send.await_count == 0
    assert tunnel.llm_proxy.generate.await_count == 0


def test_handler_accepts_signed_when_pinned():
    priv = Ed25519PrivateKey.generate()
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    pin = base64.b64encode(priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()

    tunnel = _make_tunnel()
    signed = _signed_envelope(priv, dict(VECTOR_PAYLOAD))
    signed["traceId"] = "t"
    with patch.dict(os.environ, {PUBKEY_ENV: pin}):
        _run(tunnel._handle_global_stylize_request(signed))

    assert tunnel._send.await_count == 1
    payload = tunnel._send.await_args.args[0]["payload"]
    assert payload["text"] == "styled text"
    assert "error" not in payload


def test_handler_accepts_unsigned_without_pin():
    tunnel = _make_tunnel()
    unsigned = {
        "id": "msg-1",
        "timestamp": "2023-11-14T22:13:20Z",
        "senderId": "server-node",
        "traceId": "t",
        "payload": dict(VECTOR_PAYLOAD),
    }
    with patch.dict(os.environ, {PUBKEY_ENV: ""}):
        _run(tunnel._handle_global_stylize_request(unsigned))
    assert tunnel._send.await_count == 1


def test_volunteer_mode_refuses_plaintext_to_non_loopback():
    with patch.dict(os.environ, {"LOREGUARD_VOLUNTEER": "1"}):
        with pytest.raises(ValueError):
            _make_tunnel(backend_url="ws://server.example.com/workers")
        # Loopback plaintext stays allowed for the dev round trip.
        _make_tunnel(backend_url="ws://localhost:8081/workers")
        _make_tunnel(backend_url="ws://127.0.0.1:8081/workers")
        # TLS is always fine.
        _make_tunnel(backend_url="wss://server.example.com/workers")
