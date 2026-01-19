"""Dynamic model discovery from HuggingFace.

Fetches loreguard-* models from the beyond-logic-labs organization
and generates ModelInfo objects dynamically.
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

from .models_registry import ModelInfo, HF_ORG

logger = logging.getLogger(__name__)


def _parse_days_ago(iso_date: Optional[str]) -> Optional[int]:
    """Parse ISO date string and return days ago."""
    if not iso_date:
        return None
    try:
        # Parse ISO format: "2026-01-12T23:29:38.000Z"
        dt = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = now - dt
        return max(0, delta.days)
    except (ValueError, TypeError):
        return None


# Cache settings
CACHE_FILE = Path.home() / ".cache" / "loreguard" / "hf_models_cache.json"
CACHE_TTL_HOURS = 24  # Refresh cache every 24 hours


@dataclass
class GGUFFile:
    """Info about a GGUF file in a HuggingFace repo."""
    filename: str
    size_bytes: int
    url: str
    last_modified: Optional[str] = None  # ISO date string from HF API


def _parse_quant_type(filename: str) -> tuple[str, bool]:
    """Parse quantization type and UD flag from filename.

    Returns: (quant_type, is_ud)
    Examples:
        "loreguard-vanilla-UD-Q6_K.gguf" -> ("Q6_K", True)
        "loreguard-pipeline-v3-Q6_K.gguf" -> ("Q6_K", False)
        "loreguard-vanilla-F16.gguf" -> ("F16", False)
    """
    # Remove .gguf extension
    name = filename.replace(".gguf", "")

    # Check for UD (Unsloth Dynamic)
    is_ud = "-UD-" in name

    # Extract quant type (last part after dash)
    parts = name.split("-")
    quant = parts[-1] if parts else "unknown"

    return quant, is_ud


def _estimate_size_gb(quant: str, is_ud: bool = False) -> float:
    """Estimate GGUF file size in GB based on quantization for 8B models.

    These are approximate sizes for Llama 3.1 8B models.
    """
    # Base sizes for standard quantizations (8B models)
    size_map = {
        "Q4_K": 4.9,
        "Q4_K_M": 4.9,
        "Q4K": 4.9,
        "Q4KM": 4.9,
        "Q5_K": 5.5,
        "Q5_K_M": 5.7,
        "Q5K": 5.5,
        "Q5KM": 5.7,
        "Q6_K": 6.6,
        "Q6K": 6.6,
        "Q8_0": 8.5,
        "Q80": 8.5,
        "Q8": 8.5,
        "F16": 16.0,
        "BF16": 16.0,
    }

    # Normalize quant string
    quant_upper = quant.upper().replace("_", "")

    for key, size in size_map.items():
        if key.replace("_", "") == quant_upper:
            return size

    # Default estimate based on quant level
    if "Q4" in quant_upper:
        return 4.9
    elif "Q5" in quant_upper:
        return 5.7
    elif "Q6" in quant_upper:
        return 6.6
    elif "Q8" in quant_upper:
        return 8.5
    elif "F16" in quant_upper or "BF16" in quant_upper:
        return 16.0

    return 5.0  # Default fallback


def _estimate_hardware(size_gb: float, quant: str) -> str:
    """Estimate hardware requirements based on size and quantization."""
    if size_gb <= 5:
        return "8GB RAM • 6GB VRAM"
    elif size_gb <= 6:
        return "10GB RAM • 6-8GB VRAM"
    elif size_gb <= 8:
        return "12GB RAM • 8GB VRAM"
    elif size_gb <= 10:
        return "16GB RAM • 10GB VRAM"
    else:
        return "16GB+ RAM • 12GB+ VRAM"


def _generate_model_id(repo_name: str, filename: str) -> str:
    """Generate a unique model ID from repo and filename.

    Example:
        repo="loreguard-pipeline-v3-gguf", file="loreguard-pipeline-v3-Q6_K.gguf"
        -> "loreguard-pipeline-v3-q6k"
    """
    # Remove .gguf and lowercase
    base = filename.replace(".gguf", "").lower()
    # Replace underscores with nothing for cleaner IDs
    base = base.replace("_", "")
    return base


def _generate_display_name(filename: str) -> str:
    """Generate display name from filename.

    Example: "loreguard-pipeline-v3-Q6_K.gguf" -> "Loreguard Pipeline v3 Q6_K"
    """
    name = filename.replace(".gguf", "")
    # Replace dashes and underscores with spaces
    name = name.replace("-", " ").replace("_", " ")
    # Title case
    return name.title()


def _generate_description(quant: str, is_ud: bool, size_gb: float) -> str:
    """Generate description based on quantization."""
    ud_note = " (Unsloth Dynamic)" if is_ud else ""

    if quant in ("Q4_K_M", "Q4KM"):
        return f"Smallest size, good for 6GB VRAM{ud_note}"
    elif quant in ("Q5_K_M", "Q5KM"):
        return f"Good quality/size balance{ud_note}"
    elif quant in ("Q6_K", "Q6K"):
        return f"Recommended. Best quality/size balance{ud_note}"
    elif quant in ("Q8_0", "Q80"):
        return f"Maximum quantized quality{ud_note}"
    elif quant in ("F16", "BF16"):
        return f"Full precision reference ({size_gb:.1f}GB)"
    else:
        return f"{quant} quantization{ud_note}"


def fetch_hf_repo_files(repo_id: str) -> list[GGUFFile]:
    """Fetch list of GGUF files from a HuggingFace repo."""
    api_url = f"https://huggingface.co/api/models/{repo_id}"

    try:
        req = Request(api_url, headers={"User-Agent": "loreguard-client/1.0"})
        with urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
    except (URLError, json.JSONDecodeError) as e:
        logger.warning("HF Discovery: Failed to fetch %s: %s", repo_id, e)
        return []

    # Get repo-level last modified date
    repo_last_modified = data.get("lastModified")

    gguf_files = []
    siblings = data.get("siblings", [])

    for file_info in siblings:
        filename = file_info.get("rfilename", "")
        if filename.endswith(".gguf"):
            size = file_info.get("size", 0)
            url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
            gguf_files.append(GGUFFile(
                filename=filename,
                size_bytes=size,
                url=url,
                last_modified=repo_last_modified,
            ))

    return gguf_files


def fetch_org_repos(org: str = HF_ORG) -> list[str]:
    """Fetch list of model repos from a HuggingFace organization."""
    api_url = f"https://huggingface.co/api/models?author={org}"

    try:
        req = Request(api_url, headers={"User-Agent": "loreguard-client/1.0"})
        with urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
    except (URLError, json.JSONDecodeError) as e:
        logger.warning("HF Discovery: Failed to fetch org repos: %s", e)
        return []

    # Filter for loreguard-* repos (will check for GGUF files later)
    repos = []
    for repo in data:
        repo_id = repo.get("id", "")
        # Include any loreguard-* repo - we'll filter by GGUF files in fetch_hf_repo_files
        if repo_id.startswith(f"{org}/loreguard-"):
            repos.append(repo_id)

    return repos


def discover_models(org: str = HF_ORG, use_cache: bool = True) -> list[ModelInfo]:
    """Discover all loreguard models from HuggingFace.

    Args:
        org: HuggingFace organization name
        use_cache: Whether to use cached results

    Returns:
        List of ModelInfo objects for discovered models
    """
    # Check cache first
    if use_cache:
        cached = _load_cache()
        if cached:
            return cached

    logger.info("HF Discovery: Fetching models from %s...", org)

    models = []
    repos = fetch_org_repos(org)

    for repo_id in repos:
        files = fetch_hf_repo_files(repo_id)

        for gguf_file in files:
            quant, is_ud = _parse_quant_type(gguf_file.filename)

            # Use API size if available, otherwise estimate based on quantization
            if gguf_file.size_bytes and gguf_file.size_bytes > 0:
                size_gb = gguf_file.size_bytes / (1024**3)
                size_bytes = gguf_file.size_bytes
            else:
                size_gb = _estimate_size_gb(quant, is_ud)
                size_bytes = int(size_gb * (1024**3))

            # Skip F16/BF16 entirely (too large, not practical for inference)
            if quant in ("F16", "BF16"):
                continue

            model_id = _generate_model_id(repo_id.split("/")[1], gguf_file.filename)
            days_ago = _parse_days_ago(gguf_file.last_modified)

            model = ModelInfo(
                id=model_id,
                name=_generate_display_name(gguf_file.filename),
                filename=gguf_file.filename,
                size_gb=round(size_gb, 1),
                size_bytes=size_bytes,
                context_length=8192,  # Default for Llama 3.1
                url=gguf_file.url,
                description=_generate_description(quant, is_ud, size_gb),
                hardware=_estimate_hardware(size_gb, quant),
                recommended=(quant in ("Q6_K", "Q6K") and is_ud),  # Recommend UD Q6_K
                days_ago=days_ago,
            )
            models.append(model)

    # Sort by recommendation, then size
    models.sort(key=lambda m: (not m.recommended, m.size_gb))

    # Save to cache
    _save_cache(models)

    logger.info("HF Discovery: Found %d models", len(models))
    return models


def _load_cache() -> Optional[list[ModelInfo]]:
    """Load models from cache if valid."""
    if not CACHE_FILE.exists():
        return None

    try:
        with open(CACHE_FILE) as f:
            data = json.load(f)

        # Check TTL
        cached_time = data.get("timestamp", 0)
        if time.time() - cached_time > CACHE_TTL_HOURS * 3600:
            return None

        # Reconstruct ModelInfo objects
        models = []
        for m in data.get("models", []):
            models.append(ModelInfo(**m))
        return models

    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def _save_cache(models: list[ModelInfo]) -> None:
    """Save models to cache."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": time.time(),
        "models": [
            {
                "id": m.id,
                "name": m.name,
                "filename": m.filename,
                "size_gb": m.size_gb,
                "size_bytes": m.size_bytes,
                "context_length": m.context_length,
                "url": m.url,
                "description": m.description,
                "hardware": m.hardware,
                "recommended": m.recommended,
                "experimental": m.experimental,
                "is_mlx": m.is_mlx,
                "requires_apple_silicon": m.requires_apple_silicon,
                "days_ago": m.days_ago,
            }
            for m in models
        ],
    }

    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def clear_cache() -> None:
    """Clear the model discovery cache."""
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
        print("[HF Discovery] Cache cleared")


if __name__ == "__main__":
    # Test discovery
    models = discover_models(use_cache=False)
    for m in models:
        print(f"  - {m.id}: {m.name} ({m.size_gb}GB) {'⭐' if m.recommended else ''}")
