"""Configuration for Loreguard Client.

Simple configuration loader from environment variables.
Also supports persistent file-based configuration for the TUI.
"""

import json
import os
import platform
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


# =============================================================================
# Persistent Configuration (File-based)
# =============================================================================

def get_data_dir() -> Path:
    """Get the data directory for loreguard."""
    if platform.system() == "Windows":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif platform.system() == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))

    data_dir = base / "loreguard"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_config_path() -> Path:
    """Get the path to the config file."""
    return get_data_dir() / "config.json"


@dataclass
class LoreguardConfig:
    """Loreguard client persistent configuration."""
    api_token: str = ""
    model_path: str = ""  # Store as string for JSON serialization
    adapter_path: str = ""  # Optional LoRA adapter path
    dev_mode: bool = False
    context_size: int = 16384  # llama-server context window size (configurable per game)
    max_speech_tokens: int = 50  # Max tokens for NPC speech output (Pass 4). Default: 50 (~40 words)

    def save(self) -> None:
        """Save configuration to disk."""
        config_path = get_config_path()
        with open(config_path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls) -> "LoreguardConfig":
        """Load configuration from disk."""
        config_path = get_config_path()
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    data = json.load(f)
                return cls(
                    api_token=data.get("api_token", ""),
                    model_path=data.get("model_path", ""),
                    adapter_path=data.get("adapter_path", ""),
                    dev_mode=data.get("dev_mode", False),
                    context_size=data.get("context_size", 16384),
                    max_speech_tokens=data.get("max_speech_tokens", 50),
                )
            except (json.JSONDecodeError, KeyError):
                pass
        return cls()

    def get_model_path_obj(self) -> Optional[Path]:
        """Get the model path as a Path object."""
        if self.model_path:
            path = Path(self.model_path)
            if path.exists():
                return path
        return None

    def set_model_path(self, path: Optional[Path]) -> None:
        """Set the model path."""
        self.model_path = str(path) if path else ""

    def get_adapter_path_obj(self) -> Optional[Path]:
        """Get the adapter path as a Path object."""
        if self.adapter_path:
            path = Path(self.adapter_path)
            if path.exists():
                return path
        return None

    def set_adapter_path(self, path: Optional[Path]) -> None:
        """Set the adapter path."""
        self.adapter_path = str(path) if path else ""

    def has_saved_config(self) -> bool:
        """Check if we have a saved configuration with token and model."""
        return bool(self.api_token and self.model_path and Path(self.model_path).exists())

    def clear_token(self) -> None:
        """Clear the saved token."""
        self.api_token = ""
        self.save()

    def clear_model(self) -> None:
        """Clear the saved model."""
        self.model_path = ""
        self.save()


# =============================================================================
# Environment Variable Configuration
# =============================================================================


@lru_cache(maxsize=1)
def load_config() -> dict:
    """
    Load application configuration from environment variables.

    Returns:
        dict with configuration values
    """
    return {
        # Server settings
        "LLM_ENDPOINT": os.getenv("LLM_ENDPOINT", "http://localhost:8080"),
        "BACKEND_URL": os.getenv("LOREGUARD_BACKEND", "wss://api.loreguard.com/workers"),
        "HOST": os.getenv("HOST", "127.0.0.1"),
        "PORT": os.getenv("PORT", "8081"),

        # Worker authentication (required for backend connection)
        # Get API token from loreguard.com dashboard
        "WORKER_ID": os.getenv("LOREGUARD_WORKER_ID", os.getenv("WORKER_ID", "")),
        # LOREGUARD_TOKEN is preferred, WORKER_TOKEN kept for backwards compatibility
        "LOREGUARD_TOKEN": os.getenv("LOREGUARD_TOKEN", os.getenv("WORKER_TOKEN", "")),
        "MODEL_ID": os.getenv("LOREGUARD_MODEL_ID", os.getenv("MODEL_ID", "default")),

        # Context limits (in characters, ~4 chars per token)
        # MAX_MESSAGE_LENGTH: Max size of a single message (default 100KB ~25K tokens)
        # MAX_TOTAL_CONTEXT: Max total context size (default 500KB ~125K tokens)
        # Set based on your model's context window (e.g., 32K model = ~128KB)
        "MAX_MESSAGE_LENGTH": int(os.getenv("MAX_MESSAGE_LENGTH", "100000")),
        "MAX_TOTAL_CONTEXT": int(os.getenv("MAX_TOTAL_CONTEXT", "500000")),
        "MAX_TIMEOUT": float(os.getenv("MAX_TIMEOUT", "300.0")),

        # Context compaction: if True, truncate old messages instead of erroring
        "CONTEXT_COMPACTION": os.getenv("CONTEXT_COMPACTION", "true").lower() == "true",

        # ADR-0027: Pre-shipped models directory for enterprise bundles.
        # When set, model loaders check this directory first before downloading from HF.
        # Expected subdirectories: hhem/, deberta/, distilbert/, llm/
        "MODELS_DIR": os.getenv("LOREGUARD_MODELS_DIR", ""),

        # Pre-shipped llama-server binary path (enterprise bundles).
        # When set, skips auto-download and uses this binary directly.
        "LLAMA_SERVER_PATH": os.getenv("LOREGUARD_LLAMA_SERVER_PATH", ""),

        # Bundle directory (set by game launchers that ship a loreguard bundle).
        # When set, the client auto-discovers models from manifest.txt inside the bundle.
        # This is the single env var a game needs to set — no per-model configuration required.
        "BUNDLE_DIR": os.getenv("LOREGUARD_BUNDLE_DIR", ""),
    }


def get_bundle_dir() -> Optional[Path]:
    """Get the loreguard bundle directory, if configured via LOREGUARD_BUNDLE_DIR.

    Game launchers set this to the bundle root so the client can auto-discover
    models from manifest.txt without any per-model configuration.
    """
    bundle_dir = get_config_value("BUNDLE_DIR")
    if bundle_dir:
        path = Path(bundle_dir)
        if path.exists() and path.is_dir():
            return path
    return None


def get_bundle_manifest() -> dict:
    """Parse the bundle's manifest.txt into a logical-name → dir-name mapping.

    Returns an empty dict if no bundle dir is configured or manifest is missing.

    Manifest format:
        nli=vectara--hallucination_evaluation_model
        embedding=BAAI--bge-small-en-v1.5
        reranker=cross-encoder--ms-marco-MiniLM-L-6-v2
    """
    bundle_dir = get_bundle_dir()
    if not bundle_dir:
        return {}
    manifest_path = bundle_dir / "models" / "manifest.txt"
    if not manifest_path.exists():
        return {}
    result = {}
    for line in manifest_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip()
    return result


def get_models_dir() -> Optional[Path]:
    """Get the pre-shipped models directory, if configured (ADR-0027).

    Checks LOREGUARD_MODELS_DIR first, then falls back to the bundle's models dir.
    Returns None if neither is set, meaning models should be auto-downloaded from HF.
    """
    models_dir = get_config_value("MODELS_DIR")
    if models_dir:
        path = Path(models_dir)
        if path.exists() and path.is_dir():
            return path
    bundle_dir = get_bundle_dir()
    if bundle_dir:
        path = bundle_dir / "models"
        if path.exists() and path.is_dir():
            return path
    return None


def resolve_model_path(model_name: str, subdir: str = "") -> str:
    """Resolve a model path, preferring pre-shipped models over HF downloads.

    Resolution order:
    1. LOREGUARD_MODELS_DIR/<subdir>  (explicit override)
    2. Bundle models dir using manifest.txt  (HF name → manifest key → local dir)
    3. Bundle models dir using HF name → org--model convention  (fallback)
    4. Original HF model name  (download from HuggingFace)

    Args:
        model_name: HuggingFace model name (e.g., 'vectara/hallucination_evaluation_model')
        subdir: Subdirectory within MODELS_DIR to check (e.g., 'hhem', 'deberta')

    Returns:
        Local path if pre-shipped model found, otherwise the original HF model name.
    """
    # 1. Explicit LOREGUARD_MODELS_DIR/<subdir>
    explicit_dir = get_config_value("MODELS_DIR")
    if explicit_dir and subdir:
        local_path = Path(explicit_dir) / subdir
        if local_path.exists() and any(local_path.iterdir()):
            return str(local_path)

    # 2 & 3. Bundle directory resolution
    bundle_dir = get_bundle_dir()
    if bundle_dir:
        bundle_models = bundle_dir / "models"

        # Try manifest.txt: find the dir name for this HF model name
        manifest = get_bundle_manifest()
        # The manifest uses org--model naming (/ replaced by --)
        hf_as_dir = model_name.replace("/", "--")
        for _key, dir_name in manifest.items():
            if dir_name == hf_as_dir:
                local_path = bundle_models / dir_name
                if local_path.exists() and any(local_path.iterdir()):
                    return str(local_path)
                break

        # Fallback: check bundle models dir using org--model convention directly
        local_path = bundle_models / hf_as_dir
        if local_path.exists() and any(local_path.iterdir()):
            return str(local_path)

    return model_name


def get_config_value(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get a single configuration value."""
    config = load_config()
    return config.get(key, default)


def require_config_value(key: str) -> str:
    """Get a required configuration value, raising if not found."""
    value = get_config_value(key)
    if not value:
        raise RuntimeError(f"Required configuration '{key}' is not set")
    return value
