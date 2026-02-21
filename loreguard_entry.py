#!/usr/bin/env python3
"""Top-level PyInstaller entry point.

Imports src as a proper package so relative imports inside src/__main__.py work.
"""
import multiprocessing
multiprocessing.freeze_support()  # Required for PyInstaller on macOS/Windows

from src.__main__ import main

main()
