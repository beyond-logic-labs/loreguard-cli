#!/usr/bin/env python3
"""Loreguard - Unified entry point.

Automatically detects mode:
- No arguments → Interactive wizard
- With arguments → Headless CLI mode
"""

import sys


def main():
    """Main entry point."""
    # Check if any meaningful arguments were provided
    args = sys.argv[1:]

    # Filter out help flags - these should show CLI help
    if any(a in ('-h', '--help') for a in args):
        from .cli import main as cli_main
        cli_main()
        return

    # Check if we have real arguments (not just the script name)
    has_args = bool(args)

    if has_args:
        # Headless CLI mode
        from .cli import main as cli_main
        cli_main()
    else:
        # Interactive wizard mode
        from .wizard import main as wizard_main
        wizard_main()


if __name__ == "__main__":
    main()
