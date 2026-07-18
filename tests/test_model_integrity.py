"""Tests for GGUF download integrity pinning (loreguard-engine#90)."""

import hashlib
import json
from unittest.mock import MagicMock, patch

import pytest

from src.hf_discovery import fetch_hf_repo_files
from src.model_integrity import (
    IntegrityError,
    fetch_expected_sha256,
    parse_hf_resolve_url,
    sha256_file,
    verify_digest,
)

GOOD_SHA = "a" * 64


def _urlopen_returning(payload) -> MagicMock:
    """Build a urlopen mock usable as a context manager."""
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode()
    cm = MagicMock()
    cm.__enter__.return_value = response
    cm.__exit__.return_value = False
    return MagicMock(return_value=cm)


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------


def test_parse_hf_resolve_url():
    repo, rev, filename = parse_hf_resolve_url(
        "https://huggingface.co/beyond-logic-labs/loreguard-x-gguf/resolve/abc123/model-Q6_K.gguf"
    )
    assert repo == "beyond-logic-labs/loreguard-x-gguf"
    assert rev == "abc123"
    assert filename == "model-Q6_K.gguf"

    assert parse_hf_resolve_url("https://example.com/model.gguf") is None
    assert parse_hf_resolve_url("") is None
    # Query strings are stripped from the filename.
    _, _, name = parse_hf_resolve_url(
        "https://huggingface.co/org/repo/resolve/main/m.gguf?download=true"
    )
    assert name == "m.gguf"


# ---------------------------------------------------------------------------
# Digest verification
# ---------------------------------------------------------------------------


def test_verify_digest_match_and_case_insensitive():
    verify_digest(GOOD_SHA, GOOD_SHA, "m.gguf")
    verify_digest(GOOD_SHA.upper(), GOOD_SHA, "m.gguf")


def test_verify_digest_mismatch_raises():
    with pytest.raises(IntegrityError):
        verify_digest("b" * 64, GOOD_SHA, "m.gguf")


def test_verify_digest_without_pin_accepts():
    # Defense-in-depth: no pin available -> accept with a warning, never raise.
    verify_digest("b" * 64, None, "m.gguf")
    verify_digest("b" * 64, "", "m.gguf")


def test_sha256_file(tmp_path):
    path = tmp_path / "m.gguf"
    path.write_bytes(b"gguf bytes")
    assert sha256_file(path) == hashlib.sha256(b"gguf bytes").hexdigest()


# ---------------------------------------------------------------------------
# HF tree API lookup
# ---------------------------------------------------------------------------


def test_fetch_expected_sha256_reads_lfs_oid():
    tree = [
        {"path": "README.md", "size": 10},
        {"path": "m.gguf", "lfs": {"oid": GOOD_SHA, "size": 123}},
    ]
    with patch("src.model_integrity.urlopen", _urlopen_returning(tree)):
        assert fetch_expected_sha256("org/repo", "main", "m.gguf") == GOOD_SHA


def test_fetch_expected_sha256_strips_prefix_and_rejects_junk():
    tree = [{"path": "m.gguf", "lfs": {"oid": "sha256:" + GOOD_SHA}}]
    with patch("src.model_integrity.urlopen", _urlopen_returning(tree)):
        assert fetch_expected_sha256("org/repo", "main", "m.gguf") == GOOD_SHA

    junk = [{"path": "m.gguf", "lfs": {"oid": "not-a-digest"}}]
    with patch("src.model_integrity.urlopen", _urlopen_returning(junk)):
        assert fetch_expected_sha256("org/repo", "main", "m.gguf") is None


def test_fetch_expected_sha256_missing_file_or_api_error():
    with patch("src.model_integrity.urlopen", _urlopen_returning([])):
        assert fetch_expected_sha256("org/repo", "main", "m.gguf") is None

    with patch("src.model_integrity.urlopen", MagicMock(side_effect=OSError("down"))):
        assert fetch_expected_sha256("org/repo", "main", "m.gguf") is None


# ---------------------------------------------------------------------------
# Discovery pinning
# ---------------------------------------------------------------------------


def test_discovery_pins_revision_and_digest():
    repo = {
        "sha": "commit123",
        "lastModified": "2026-07-01T00:00:00.000Z",
        "siblings": [
            {"rfilename": "m-Q6_K.gguf", "size": 42, "lfs": {"oid": "sha256:" + GOOD_SHA}},
            {"rfilename": "README.md", "size": 5},
        ],
    }
    with patch("src.hf_discovery.urlopen", _urlopen_returning(repo)):
        files = fetch_hf_repo_files("org/repo")

    assert len(files) == 1
    f = files[0]
    assert f.url == "https://huggingface.co/org/repo/resolve/commit123/m-Q6_K.gguf"
    assert f.sha256 == GOOD_SHA
    assert f.revision == "commit123"
