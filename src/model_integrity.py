"""GGUF download integrity pinning (loreguard-engine#90).

Defense-in-depth for model downloads: discovery pins each file's HuggingFace
LFS sha256 digest and the repo commit revision, and the downloader hashes
the byte stream and refuses a file whose digest does not match. When no
expected digest can be obtained (HF API unreachable, non-HF mirror), the
download is accepted with a logged warning: the primary control remains
HTTPS to the pinned organization, this layer only hardens it.
"""

import hashlib
import json
import logging
import re
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# https://huggingface.co/{org}/{repo}/resolve/{revision}/{filename}
_HF_RESOLVE_RE = re.compile(
    r"^https://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/(.+?)(?:\?.*)?$"
)


def parse_hf_resolve_url(url: str) -> Optional[tuple[str, str, str]]:
    """Split an HF resolve URL into (repo_id, revision, filename)."""
    m = _HF_RESOLVE_RE.match(url or "")
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def fetch_expected_sha256(repo_id: str, revision: str, filename: str) -> Optional[str]:
    """Fetch the LFS sha256 for one file from the HF tree API.

    Returns the lowercase hex digest, or None when the API is unreachable or
    the file has no LFS digest (small non-LFS files).
    """
    api_url = f"https://huggingface.co/api/models/{repo_id}/tree/{revision}?recursive=true"
    try:
        req = Request(api_url, headers={"User-Agent": "loreguard-client/1.0"})
        with urlopen(req, timeout=10) as response:
            entries = json.loads(response.read().decode())
    except (URLError, json.JSONDecodeError, OSError) as e:
        logger.warning("Integrity: failed to fetch tree for %s@%s: %s", repo_id, revision, e)
        return None

    for entry in entries:
        if entry.get("path") != filename:
            continue
        oid = (entry.get("lfs") or {}).get("oid") or ""
        oid = oid.removeprefix("sha256:").strip().lower()
        if re.fullmatch(r"[0-9a-f]{64}", oid):
            return oid
        return None
    return None


def expected_sha256_for_url(url: str) -> Optional[str]:
    """Resolve the expected sha256 for an HF resolve URL, if obtainable."""
    parsed = parse_hf_resolve_url(url)
    if not parsed:
        return None
    return fetch_expected_sha256(*parsed)


class IntegrityError(Exception):
    """Raised when a downloaded file's digest does not match the pin."""


def verify_digest(computed_hex: str, expected_hex: Optional[str], label: str) -> None:
    """Compare a computed digest against the expected pin.

    No-op (with a warning) when there is no pin; raises IntegrityError on a
    mismatch. Callers must treat the error as fatal for the downloaded file
    (delete it, never load it).
    """
    if not expected_hex:
        logger.warning("Integrity: no sha256 pin available for %s; accepting unverified download", label)
        return
    if computed_hex.lower() != expected_hex.lower():
        raise IntegrityError(
            f"sha256 mismatch for {label}: expected {expected_hex}, got {computed_hex}"
        )
    logger.info("Integrity: sha256 verified for %s", label)


def sha256_file(path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash an existing file (for verifying files not hashed in-stream)."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()
