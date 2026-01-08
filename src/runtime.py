"""Runtime info for service discovery.

This module manages the runtime.json file that SDKs use to discover
the port loreguard-client is running on.

Location (platform-specific):
  - macOS: ~/Library/Application Support/loreguard/runtime.json
  - Linux: ~/.local/share/loreguard/runtime.json
  - Windows: %APPDATA%/loreguard/runtime.json
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .config import get_data_dir


@lru_cache(maxsize=1)
def get_version() -> str:
    """Get the package version from pyproject.toml."""
    try:
        # Try importlib.metadata first (Python 3.8+)
        from importlib.metadata import version
        return version("loreguard-cli")
    except Exception:
        pass

    # Fallback: read from pyproject.toml
    try:
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            content = pyproject_path.read_text()
            for line in content.split("\n"):
                if line.startswith("version"):
                    # Parse: version = "0.6.0"
                    return line.split("=")[1].strip().strip('"')
    except Exception:
        pass

    return "0.6.0"  # Fallback


def get_runtime_path() -> Path:
    """Get path to runtime.json."""
    return get_data_dir() / "runtime.json"


@dataclass
class RuntimeInfo:
    """Runtime info written on startup for SDK discovery.

    Attributes:
        port: The port the FastAPI server is listening on
        pid: Process ID of loreguard-client
        started_at: ISO timestamp when the server started
        version: Version of loreguard-client
        llm_port: Port of the local llama.cpp server (if running)
        backend_connected: Whether connected to Loreguard backend
    """

    port: int
    pid: int
    started_at: str
    version: str
    llm_port: Optional[int] = None
    backend_connected: bool = False

    def save(self) -> None:
        """Write runtime info to disk."""
        path = get_runtime_path()
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def update_backend_status(self, connected: bool) -> None:
        """Update backend connection status and save."""
        self.backend_connected = connected
        self.save()

    @classmethod
    def load(cls) -> Optional["RuntimeInfo"]:
        """Load runtime info from disk.

        Returns:
            RuntimeInfo if file exists and is valid, None otherwise.
        """
        path = get_runtime_path()
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            return cls(
                port=data["port"],
                pid=data["pid"],
                started_at=data["started_at"],
                version=data["version"],
                llm_port=data.get("llm_port"),
                backend_connected=data.get("backend_connected", False),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    @classmethod
    def clear(cls) -> None:
        """Remove runtime file on shutdown."""
        path = get_runtime_path()
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass  # Ignore errors during cleanup

    def to_status_dict(self) -> dict:
        """Convert to status dict for CLI output."""
        return {
            "running": True,
            "port": self.port,
            "pid": self.pid,
            "url": f"http://127.0.0.1:{self.port}",
            "started_at": self.started_at,
            "version": self.version,
            "llm_port": self.llm_port,
            "backend_connected": self.backend_connected,
        }


def write_runtime_info(
    port: int,
    version: Optional[str] = None,
    llm_port: Optional[int] = None,
) -> RuntimeInfo:
    """Create and save runtime info.

    Args:
        port: The port the server is listening on
        version: Version string (defaults to package version)
        llm_port: Port of local llama.cpp server (optional)

    Returns:
        The created RuntimeInfo instance
    """
    info = RuntimeInfo(
        port=port,
        pid=os.getpid(),
        started_at=datetime.utcnow().isoformat() + "Z",
        version=version or get_version(),
        llm_port=llm_port,
        backend_connected=False,
    )
    info.save()
    return info


def get_status(verify_health: bool = True) -> dict:
    """Get current status for CLI output.

    Args:
        verify_health: If True, also check if the server responds to /health

    Returns:
        Status dict with running=True and info, or running=False
    """
    info = RuntimeInfo.load()
    if not info:
        return {"running": False}

    # Check if the process is actually running
    try:
        os.kill(info.pid, 0)  # Signal 0 just checks if process exists
    except OSError:
        # Process not running, clean up stale file
        RuntimeInfo.clear()
        return {"running": False}

    # Optionally verify the server is responding
    if verify_health:
        try:
            import urllib.request
            url = f"http://127.0.0.1:{info.port}/health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    # Update backend_connected from health response
                    import json
                    health = json.loads(response.read().decode())
                    info.backend_connected = health.get("backend_connected", False)
        except Exception:
            # Server not responding, but process exists - might be starting up
            pass

    return info.to_status_dict()
