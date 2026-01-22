"""llama-server download and management.

Downloads pre-built llama.cpp server binaries from GitHub releases:
- Windows: CUDA 12.4
- Linux: CUDA 12.4
- macOS: Metal (Apple Silicon)
"""

import asyncio
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, Callable, Optional

import httpx

LLAMA_VERSION = "b7789"  # Must match loreguard-engine bundle version

# Download URLs for each platform
BINARIES = {
    "windows": {
        "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_VERSION}/llama-{LLAMA_VERSION}-bin-win-cuda-12.4-x64.zip",
        "archive_type": "zip",
        "binary_name": "llama-server.exe",
    },
    "linux": {
        "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_VERSION}/llama-{LLAMA_VERSION}-bin-ubuntu-x64.tar.gz",
        "archive_type": "tar.gz",
        "binary_name": "llama-server",
    },
    "macos": {
        "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_VERSION}/llama-{LLAMA_VERSION}-bin-macos-arm64.tar.gz",
        "archive_type": "tar.gz",
        "binary_name": "llama-server",
    },
}


@dataclass
class DownloadProgress:
    """Progress update during download."""
    downloaded_bytes: int
    total_bytes: int
    percent: float
    speed_mbps: float


def get_platform() -> str:
    """Get the current platform name."""
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    elif system == "windows":
        return "windows"
    elif system == "linux":
        return "linux"
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


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


def get_bin_dir() -> Path:
    """Get the bin directory for llama-server."""
    bin_dir = get_data_dir() / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    return bin_dir


def get_models_dir() -> Path:
    """Get the models directory."""
    models_dir = get_data_dir() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_slot_cache_dir() -> Path:
    """Get the directory for slot KV cache persistence.

    ADR-0014: This directory stores KV cache snapshots for warmup context,
    enabling cross-message caching without requiring multiple slots.
    """
    cache_dir = get_data_dir() / "kv_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_llama_server_path() -> Path:
    """Get the path to llama-server binary."""
    plat = get_platform()
    binary_name = BINARIES[plat]["binary_name"]
    return get_bin_dir() / binary_name


def get_version_file_path() -> Path:
    """Get the path to the version marker file."""
    return get_bin_dir() / ".llama_version"


def get_installed_version() -> Optional[str]:
    """Get the currently installed llama-server version, or None if not installed."""
    version_file = get_version_file_path()
    if version_file.exists():
        try:
            return version_file.read_text().strip()
        except OSError:
            return None
    return None


def is_llama_server_installed() -> bool:
    """Check if llama-server is installed with the correct version."""
    server_path = get_llama_server_path()
    if not (server_path.exists() and server_path.is_file()):
        return False

    # Check version matches
    installed_version = get_installed_version()
    if installed_version != LLAMA_VERSION:
        # Version mismatch - need to re-download
        return False

    return True


def cleanup_old_llama_server() -> Optional[str]:
    """Remove old llama-server installation if version mismatch detected.

    Returns the old version string if cleanup was performed, None otherwise.
    """
    installed_version = get_installed_version()
    if installed_version is None:
        return None

    if installed_version == LLAMA_VERSION:
        return None

    # Version mismatch - clean up old installation
    bin_dir = get_bin_dir()
    if bin_dir.exists():
        shutil.rmtree(bin_dir)
        bin_dir.mkdir(parents=True, exist_ok=True)

    return installed_version


async def download_file(
    url: str,
    dest: Path,
    progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
) -> None:
    """Download a file with progress tracking."""
    import time

    async with httpx.AsyncClient(follow_redirects=True, timeout=600.0) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()

            total = int(response.headers.get("content-length", 0))
            downloaded = 0
            start_time = time.time()

            with open(dest, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)

                    if progress_callback and total > 0:
                        elapsed = time.time() - start_time
                        speed = (downloaded / elapsed / 1024 / 1024) if elapsed > 0 else 0
                        progress_callback(DownloadProgress(
                            downloaded_bytes=downloaded,
                            total_bytes=total,
                            percent=(downloaded / total) * 100,
                            speed_mbps=speed,
                        ))


def extract_archive(archive_path: Path, dest_dir: Path, archive_type: str) -> None:
    """Extract an archive to a directory."""
    if archive_type == "zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
    elif archive_type == "tar.gz":
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(dest_dir)
    else:
        raise ValueError(f"Unknown archive type: {archive_type}")


def find_binary_in_extracted(extract_dir: Path, binary_name: str) -> Optional[Path]:
    """Find the llama-server binary in extracted files."""
    # Check common locations
    possible_paths = [
        extract_dir / binary_name,
        extract_dir / "build" / "bin" / binary_name,
        extract_dir / "bin" / binary_name,
    ]

    for path in possible_paths:
        if path.exists():
            return path

    # Search recursively
    for path in extract_dir.rglob(binary_name):
        return path

    return None


def make_executable(path: Path) -> None:
    """Make a file executable on Unix systems."""
    if platform.system() != "Windows":
        current = path.stat().st_mode
        path.chmod(current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


async def download_llama_server(
    progress_callback: Optional[Callable[[str, DownloadProgress | None], None]] = None,
) -> Path:
    """Download and install llama-server for the current platform.

    Args:
        progress_callback: Called with (status_message, progress_or_none)

    Returns:
        Path to the installed llama-server binary
    """
    plat = get_platform()
    config = BINARIES[plat]
    bin_dir = get_bin_dir()

    def notify(msg: str, progress: DownloadProgress | None = None):
        if progress_callback:
            progress_callback(msg, progress)

    # Clean up old version if upgrading
    old_version = cleanup_old_llama_server()
    if old_version:
        notify(f"Upgrading llama-server from {old_version} to {LLAMA_VERSION}...")
    else:
        notify(f"Downloading llama-server {LLAMA_VERSION} for {plat}...")

    # Download to temp file
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        archive_name = f"llama-server.{config['archive_type'].replace('.', '_')}"
        archive_path = tmp_path / archive_name

        # Download
        await download_file(
            config["url"],
            archive_path,
            lambda p: notify("Downloading...", p),
        )

        notify("Extracting...")

        # Extract
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        extract_archive(archive_path, extract_dir, config["archive_type"])

        # Find binary
        binary_path = find_binary_in_extracted(extract_dir, config["binary_name"])
        if not binary_path:
            raise RuntimeError(f"Could not find {config['binary_name']} in archive")

        # Copy to bin directory
        dest_path = bin_dir / config["binary_name"]
        shutil.copy2(binary_path, dest_path)

        # Also copy any .dylib/.so files (required on macOS/Linux)
        # Some archives store symlinks as text files - we need to handle those
        def copy_lib_file(lib_file: Path, dest_dir: Path) -> None:
            """Copy a library file, handling symlinks stored as text."""
            dest = dest_dir / lib_file.name
            # Check if it's a small file that might be a symlink stored as text
            if lib_file.stat().st_size < 100:
                try:
                    content = lib_file.read_text().strip()
                    # If it looks like a filename (no slashes, ends with .dylib or .so)
                    if ('/' not in content and
                        (content.endswith('.dylib') or '.so' in content)):
                        # It's a symlink stored as text - create actual symlink
                        if dest.exists() or dest.is_symlink():
                            dest.unlink()
                        dest.symlink_to(content)
                        return
                except (UnicodeDecodeError, OSError):
                    pass
            # Normal file - copy it
            shutil.copy2(lib_file, dest)

        for lib_file in binary_path.parent.glob("*.dylib"):
            copy_lib_file(lib_file, bin_dir)
        for lib_file in binary_path.parent.glob("*.so*"):
            copy_lib_file(lib_file, bin_dir)

        # Make executable
        make_executable(dest_path)
        for lib in bin_dir.glob("*.dylib"):
            if lib.is_file() and not lib.is_symlink():
                make_executable(lib)
        for lib in bin_dir.glob("*.so*"):
            if lib.is_file() and not lib.is_symlink():
                make_executable(lib)

        # Write version marker file for future version checks
        version_file = get_version_file_path()
        version_file.write_text(LLAMA_VERSION)

        notify(f"llama-server {LLAMA_VERSION} installed successfully!")

    return get_llama_server_path()


class LlamaServerProcess:
    """Manages a llama-server subprocess."""

    def __init__(
        self,
        model_path: Path,
        port: int = 8080,
        lora_path: Optional[Path] = None,
        context_size: int = 16384,
    ):
        self.model_path = model_path
        self.port = port
        self.lora_path = lora_path
        self.context_size = context_size
        self.process: Optional[subprocess.Popen] = None
        self._output_lines: list[str] = []

    def start(self) -> None:
        """Start the llama-server process."""
        if self.process is not None:
            raise RuntimeError("Server already running")

        server_path = get_llama_server_path()
        if not server_path.exists():
            raise RuntimeError("llama-server not installed")

        # Build command
        cmd = [
            str(server_path),
            "-m", str(self.model_path),
            "--port", str(self.port),
            "--host", "127.0.0.1",
            "-c", str(self.context_size),  # Context length (configurable per game)
            "-ngl", "99",   # GPU layers (use all)
            # ADR-0014: Enable slot save/restore for disk-based KV cache persistence
            # This allows saving warmup context to disk and restoring across messages
            "--slots",  # Enable /slots API endpoint (required for save/restore)
            "--slot-save-path", str(get_slot_cache_dir()),
            # ADR-0014: Force single slot when slot caching is enabled
            # This minimizes VRAM usage and ensures slot 0 pinning works correctly.
            # Without this, llama-server may allocate multiple slots, each consuming
            # KV cache memory proportional to context_size * model_hidden_dim.
            "-np", "1",
            # Use custom Jinja template without tool-calling logic.
            # Llama 3.1's built-in template forces tool-calling format even without tools,
            # so we use a stripped-down template that only handles chat messages.
            "--jinja",
            "--chat-template-file", str(Path(__file__).parent.parent / "templates" / "llama31-no-tools.jinja"),
        ]

        # Add LoRA adapter if specified
        if self.lora_path and self.lora_path.exists():
            cmd.extend(["--lora", str(self.lora_path)])

        # Start process
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

    def stop(self) -> None:
        """Stop the llama-server process."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    def is_running(self) -> bool:
        """Check if the server is running."""
        if self.process is None:
            return False
        return self.process.poll() is None

    async def wait_for_ready(self, timeout: float = 120.0) -> bool:
        """Wait for the server to be ready.

        Returns True when server responds with 200 on /health.
        Returns False if process dies or timeout is reached.
        """
        import time

        start = time.time()
        async with httpx.AsyncClient() as client:
            while time.time() - start < timeout:
                # Check if process died
                if self.process and self.process.poll() is not None:
                    # Process exited - read any error output
                    return False

                try:
                    response = await client.get(
                        f"http://127.0.0.1:{self.port}/health",
                        timeout=5.0
                    )
                    if response.status_code == 200:
                        return True
                    # 503 means "loading" - server is up but model not ready yet
                    # Keep waiting
                except httpx.RequestError:
                    pass

                await asyncio.sleep(1.0)

        return False

    def read_output(self) -> list[str]:
        """Read available output lines from the process."""
        if self.process and self.process.stdout:
            import select

            # Non-blocking read
            while True:
                # Check if there's data to read (Unix only)
                if hasattr(select, 'select'):
                    readable, _, _ = select.select([self.process.stdout], [], [], 0)
                    if not readable:
                        break

                line = self.process.stdout.readline()
                if not line:
                    break
                self._output_lines.append(line.strip())

        lines = self._output_lines
        self._output_lines = []
        return lines
