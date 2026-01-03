"""Configuration for Loreguard Client.

Simple configuration loader from environment variables.
"""

import os
from functools import lru_cache
from typing import Optional

from rich.console import Console

console = Console()


@lru_cache(maxsize=1)
def load_config() -> dict:
    """
    Load application configuration from environment variables.

    Returns:
        dict with configuration values
    """
    console.print("[yellow]Loading config from environment variables[/yellow]")
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
