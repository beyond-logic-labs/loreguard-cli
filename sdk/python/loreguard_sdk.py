"""Loreguard SDK for Python game engines.

This SDK helps Python-based games discover and connect to loreguard-client.

Usage:
    from loreguard_sdk import get_base_url, chat

    # Get the URL for loreguard-client
    url = get_base_url()  # e.g., "http://127.0.0.1:52341"

    # Chat with an NPC (streaming)
    async for event in chat("merchant-npc", "What do you have for sale?"):
        if "t" in event:
            print(event["t"], end="", flush=True)
        elif "speech" in event:
            print()  # Done

Requirements:
    pip install httpx
"""

import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, AsyncIterator, Optional

try:
    import httpx
except ImportError:
    httpx = None  # Optional dependency


def get_data_dir() -> Path:
    """Get loreguard data directory (matches loreguard-client).

    Returns:
        Path to the loreguard data directory:
        - macOS: ~/Library/Application Support/loreguard
        - Linux: ~/.local/share/loreguard (or $XDG_DATA_HOME/loreguard)
        - Windows: %APPDATA%/loreguard
    """
    if platform.system() == "Windows":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif platform.system() == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "loreguard"


def get_runtime_info() -> Optional[dict]:
    """Load runtime info from loreguard-client.

    Returns:
        Runtime info dict if loreguard-client is running, None otherwise.
        Contains: port, pid, url, started_at, version, backend_connected
    """
    runtime_path = get_data_dir() / "runtime.json"
    if not runtime_path.exists():
        return None
    try:
        with open(runtime_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def get_local_port() -> int:
    """Get the port loreguard-client is running on.

    Returns:
        The port number

    Raises:
        RuntimeError: If loreguard-client is not running
    """
    info = get_runtime_info()
    if info and "port" in info:
        return info["port"]
    raise RuntimeError(
        "loreguard-client not running. "
        "Start it with: loreguard"
    )


def get_base_url() -> str:
    """Get the base URL for loreguard-client API.

    Returns:
        URL like "http://127.0.0.1:52341"

    Raises:
        RuntimeError: If loreguard-client is not running
    """
    port = get_local_port()
    return f"http://127.0.0.1:{port}"


def is_running() -> bool:
    """Check if loreguard-client is running.

    Returns:
        True if running, False otherwise
    """
    info = get_runtime_info()
    if not info:
        return False

    # Verify process is actually running
    pid = info.get("pid")
    if pid:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    return False


def get_status_via_cli() -> Optional[dict]:
    """Get status by calling loreguard CLI (fallback method).

    Useful when file access isn't available but subprocess is.

    Returns:
        Status dict if successful, None otherwise
    """
    try:
        result = subprocess.run(
            ["loreguard", "status"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError):
        pass
    return None


async def chat(
    character_id: str,
    message: str,
    player_handle: str = "",
    current_context: str = "",
    stream: bool = True,
) -> AsyncIterator[dict[str, Any]]:
    """Chat with an NPC via loreguard-client.

    Args:
        character_id: The NPC's ID
        message: Player's message to the NPC
        player_handle: Player's display name (optional)
        current_context: Game context like "in a dark cave" (optional)
        stream: If True, yields tokens as they arrive. If False, yields final response.

    Yields:
        For streaming: {"t": "token"} for each token, then {"speech": "...", "verified": True, ...}
        For non-streaming: Single dict with complete response

    Raises:
        RuntimeError: If loreguard-client is not running
        ImportError: If httpx is not installed

    Example:
        async for event in chat("merchant", "Hello!"):
            if "t" in event:
                print(event["t"], end="")
            elif "speech" in event:
                print(f"\\nVerified: {event['verified']}")
    """
    if httpx is None:
        raise ImportError("httpx is required for chat. Install with: pip install httpx")

    url = f"{get_base_url()}/api/chat"
    headers = {"Content-Type": "application/json"}

    if stream:
        headers["Accept"] = "text/event-stream"

    body = {
        "character_id": character_id,
        "message": message,
        "player_handle": player_handle,
        "current_context": current_context,
    }

    async with httpx.AsyncClient() as client:
        if stream:
            async with client.stream(
                "POST",
                url,
                headers=headers,
                json=body,
                timeout=120.0,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            yield data
                            # Stop after 'done' or 'error' event
                            if "speech" in data or "error" in data:
                                break
                        except json.JSONDecodeError:
                            continue
        else:
            response = await client.post(url, headers=headers, json=body, timeout=120.0)
            response.raise_for_status()
            yield response.json()


async def health_check() -> dict[str, Any]:
    """Check loreguard-client health.

    Returns:
        Health status dict with llm_available, backend_connected, etc.

    Raises:
        RuntimeError: If loreguard-client is not running
        ImportError: If httpx is not installed
    """
    if httpx is None:
        raise ImportError("httpx is required. Install with: pip install httpx")

    url = f"{get_base_url()}/health"
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=5.0)
        response.raise_for_status()
        return response.json()


# Synchronous versions for simpler use cases
def chat_sync(
    character_id: str,
    message: str,
    player_handle: str = "",
    current_context: str = "",
) -> dict[str, Any]:
    """Synchronous chat (non-streaming).

    For async streaming, use the async `chat()` function instead.

    Returns:
        Complete response dict with speech, verified, citations, etc.
    """
    if httpx is None:
        raise ImportError("httpx is required. Install with: pip install httpx")

    url = f"{get_base_url()}/api/chat"
    body = {
        "character_id": character_id,
        "message": message,
        "player_handle": player_handle,
        "current_context": current_context,
    }

    with httpx.Client() as client:
        response = client.post(url, json=body, timeout=120.0)
        response.raise_for_status()
        return response.json()
