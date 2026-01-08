#!/usr/bin/env python3
"""Loreguard - Unified entry point.

Automatically detects mode:
- No arguments → Interactive TUI wizard
- With arguments → Headless CLI mode
- status → Show runtime status (for SDK discovery)
"""

import sys
import json


def main():
    """Main entry point."""
    # Check if any meaningful arguments were provided
    args = sys.argv[1:]

    # Handle 'status' command - for SDK discovery
    if args and args[0] == "status":
        from .runtime import get_status
        status = get_status()
        print(json.dumps(status, indent=2))
        sys.exit(0 if status.get("running") else 1)

    # Filter out help flags - these should show CLI help
    if any(a in ('-h', '--help') for a in args):
        from .cli import main as cli_main
        cli_main()
        return

    # Check for mode flags
    dev_mode = '--dev' in args
    verbose = '-v' in args or '--verbose' in args

    # Flags that should still launch TUI mode (not CLI mode)
    tui_only_flags = {'--dev', '-v', '--verbose'}

    # Check if we have real arguments that require CLI mode
    has_cli_args = bool([a for a in args if a not in tui_only_flags])

    if has_cli_args:
        # Headless CLI mode
        from .cli import main as cli_main
        cli_main()
    else:
        # Interactive TUI wizard mode
        from .tui import LoreguardApp
        app = LoreguardApp(dev_mode=dev_mode, verbose=verbose)
        app.run()


if __name__ == "__main__":
    main()
