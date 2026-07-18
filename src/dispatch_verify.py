"""Verify Ed25519-signed volunteer dispatches (loreguard-engine#87).

Mirrors the engine's canonical signing bytes
(loreguard-engine internal/cloud/workers/dispatch_signing.go). The two
implementations MUST stay byte-identical: both repos pin the same test
vector, and string lengths are UTF-8 BYTE lengths, not code points. The
signature deliberately covers a length-prefixed field encoding instead of
JSON bytes, so neither side ever depends on JSON canonicalization.
"""

import base64
import logging
import os
import re
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

CONTEXT = b"lgv-dispatch-v1"
SIGNATURE_PREFIX = "ed25519:"
PUBKEY_ENV = "LOREGUARD_DISPATCH_PUBKEY"


def canonical_stylize_signing_bytes(msg_id: str, ts_unix: int, sender_id: str, payload: dict) -> bytes:
    """Build the byte string the engine signed for a stylize dispatch."""
    out = bytearray()

    def s(value: str) -> None:
        raw = value.encode("utf-8")
        out.extend(str(len(raw)).encode("ascii"))
        out.extend(b":")
        out.extend(raw)
        out.append(0)

    def n(value: int) -> None:
        out.extend(str(int(value)).encode("ascii"))
        out.append(0)

    out.extend(CONTEXT)
    out.append(0)
    s(msg_id)
    n(ts_unix)
    s(sender_id)
    s(str(payload.get("job_id") or ""))
    s(str(payload.get("content_kind") or ""))
    facts = payload.get("facts") or []
    n(len(facts))
    for fact in facts:
        s(str(fact.get("key") or ""))
        s(str(fact.get("value") or ""))
    entities = payload.get("allowed_entities") or []
    n(len(entities))
    for entity in entities:
        s(str(entity))
    s(str(payload.get("style_hints") or ""))
    n(payload.get("max_output_chars") or 0)
    n(payload.get("timeout_ms") or 0)
    return bytes(out)


def iso_to_unix(timestamp: str) -> Optional[int]:
    """Parse an RFC3339 envelope timestamp to unix seconds.

    Go emits RFC3339Nano (up to 9 fractional digits, any zone offset);
    fromisoformat wants at most 6 fractional digits, so trim first. The
    signature binds whole seconds (Go's Time.Unix()), matching truncation.
    """
    if not timestamp:
        return None
    try:
        cleaned = timestamp.replace("Z", "+00:00")
        cleaned = re.sub(
            r"\.(\d{1,9})", lambda m: "." + m.group(1)[:6].ljust(6, "0"), cleaned, count=1
        )
        return int(datetime.fromisoformat(cleaned).timestamp())
    except ValueError:
        return None


def load_pinned_pubkey():
    """Load the pinned dispatch public key from LOREGUARD_DISPATCH_PUBKEY.

    Returns None when no pin is configured. Raises ValueError on a malformed
    pin: callers must treat that as fail-closed (reject jobs), never as
    "no pin".
    """
    raw = os.environ.get(PUBKEY_ENV, "").strip()
    if not raw:
        return None

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    padded = raw + "=" * (-len(raw) % 4)
    key_bytes = None
    # Strict decoding (validate=True): the lax default silently drops
    # non-alphabet characters and would turn a urlsafe pin into garbage.
    for altchars in (None, b"-_"):
        try:
            candidate = base64.b64decode(padded, altchars=altchars, validate=True)
        except Exception:
            continue
        if len(candidate) == 32:
            key_bytes = candidate
            break
    if key_bytes is None:
        raise ValueError(f"{PUBKEY_ENV} must be a base64 32-byte Ed25519 public key")
    return Ed25519PublicKey.from_public_bytes(key_bytes)


def verify_stylize_dispatch(pubkey, data: dict) -> tuple[bool, str]:
    """Verify a global_stylize_request envelope against the pinned key.

    Returns (ok, reason). Any missing, malformed, or mismatching signature
    fails: with a pin configured there is no unsigned fallback.
    """
    signature = str(data.get("signature") or "")
    if not signature.startswith(SIGNATURE_PREFIX):
        return False, "missing or non-ed25519 signature"
    sig_b64 = signature[len(SIGNATURE_PREFIX):]
    try:
        sig_raw = base64.urlsafe_b64decode(sig_b64 + "=" * (-len(sig_b64) % 4))
    except Exception:
        return False, "undecodable signature"
    if len(sig_raw) != 64:
        return False, "wrong signature length"

    ts_unix = iso_to_unix(str(data.get("timestamp") or ""))
    if ts_unix is None:
        return False, "missing or unparseable timestamp"

    message = canonical_stylize_signing_bytes(
        str(data.get("id") or ""),
        ts_unix,
        str(data.get("senderId") or ""),
        data.get("payload") or {},
    )

    from cryptography.exceptions import InvalidSignature

    try:
        pubkey.verify(sig_raw, message)
    except InvalidSignature:
        return False, "signature mismatch"
    return True, ""
