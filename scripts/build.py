#!/usr/bin/env python3
"""Build standalone Loreguard binary for distribution.

Usage:
    python scripts/build.py

Creates:
    dist/loreguard (or loreguard.exe on Windows)
"""

import os
import platform
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).parent.parent
    os.chdir(root)

    # Determine output name based on platform
    system = platform.system().lower()
    if system == "windows":
        name = "loreguard.exe"
    else:
        name = "loreguard"

    print(f"Building Loreguard for {system}...")

    # PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name", "loreguard",
        "--clean",
        # Add src to path so imports work
        "--paths", ".",
        # Collect all src submodules
        "--collect-submodules", "src",
        # Add hidden imports that PyInstaller might miss
        "--hidden-import", "src",
        "--hidden-import", "src.config",
        "--hidden-import", "src.llm",
        "--hidden-import", "src.llama_server",
        "--hidden-import", "src.tunnel",
        "--hidden-import", "src.term_ui",
        "--hidden-import", "src.models_registry",
        "--hidden-import", "src.cli",
        "--hidden-import", "src.npc_chat",
        "--hidden-import", "rich",
        "--hidden-import", "rich.console",
        "--hidden-import", "httpx",
        "--hidden-import", "websockets",
        "--hidden-import", "aiofiles",
        "--hidden-import", "pydantic",
        # Entry point - the interactive wizard
        "src/wizard.py",
    ]

    # Add icon on Windows/macOS
    icon_path = root / "assets" / "icon.ico"
    if icon_path.exists():
        cmd.extend(["--icon", str(icon_path)])

    # Run PyInstaller
    result = subprocess.run(cmd, cwd=root)

    if result.returncode == 0:
        dist_path = root / "dist" / name
        print(f"\n✅ Build successful!")
        print(f"   Binary: {dist_path}")
        print(f"   Size: {dist_path.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        print("\n❌ Build failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
