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
    }


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
