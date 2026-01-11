"""Main Loreguard TUI application using Textual framework."""

import atexit
import os
import sys
from pathlib import Path
from typing import Optional, Any

from textual.app import App
from textual.binding import Binding
from textual import events
from textual.driver import Driver

from .styles import LOREGUARD_CSS
from .widgets.hardware_info import HardwareData


def _restore_terminal():
    """Restore terminal state on exit."""
    if sys.stdout.isatty():
        # Show cursor
        sys.stdout.write("\033[?25h")
        # Reset terminal attributes
        sys.stdout.write("\033[0m")
        sys.stdout.flush()
        # Try stty sane as last resort
        try:
            os.system("stty sane 2>/dev/null")
        except Exception:
            pass


# Register terminal restore on exit
atexit.register(_restore_terminal)


class LoreguardApp(App):
    """Main Loreguard TUI application."""

    CSS = LOREGUARD_CSS

    # Disable Textual's default command palette (we use our own)
    ENABLE_COMMAND_PALETTE = False

    # Use dark mode (Dracula) by default
    DARK = True

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True, priority=True),
        Binding("ctrl+t", "toggle_dark", "Theme", show=False),
    ]

    # Shared application state
    api_token: str = ""
    worker_id: str = ""
    model_path: Optional[Path] = None
    adapter_path: Optional[Path] = None  # Optional LoRA adapter
    hardware: Optional[HardwareData] = None
    dev_mode: bool = False
    verbose: bool = False

    # Process references (managed by MainScreen)
    _llama_process: Any = None
    _tunnel: Any = None

    def get_driver_class(self) -> type[Driver]:
        """Return a driver class that ignores mouse events."""
        base_driver = super().get_driver_class()
        if getattr(base_driver, "_LOREGUARD_NO_MOUSE", False):
            return base_driver

        class NoMouseDriver(base_driver):
            _LOREGUARD_NO_MOUSE = True

            def start_application_mode(self) -> None:
                super().start_application_mode()
                # Note: Mouse support stays enabled for scroll events
                # Click events are filtered in process_event()
                # Text selection: use Shift+click in most terminals

            def stop_application_mode(self) -> None:
                super().stop_application_mode()

            def process_event(self, event: events.Event) -> None:
                if isinstance(event, events.MouseEvent):
                    # Allow scroll events for scrolling, block others for text selection
                    if not isinstance(event, (events.MouseScrollDown, events.MouseScrollUp)):
                        return
                super().process_event(event)

        return NoMouseDriver

    def __init__(self, dev_mode: bool = False, verbose: bool = False) -> None:
        super().__init__()
        self.dev_mode = dev_mode
        self.verbose = verbose
        self.theme = "dracula"  # Default to Dracula theme

        # Configure debug logging when verbose mode is enabled
        if verbose:
            self._setup_debug_logging()

    def _setup_debug_logging(self) -> None:
        """Setup debug logging to file when verbose mode is enabled."""
        import logging
        from pathlib import Path

        log_file = Path.cwd() / "loreguard-debug.log"

        # Get root logger
        root_logger = logging.getLogger()

        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # File handler - captures all debug output
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))

        # Console handler - only warnings (don't mess up TUI)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(logging.Formatter("%(message)s"))

        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        root_logger.setLevel(logging.DEBUG)

        # Store log file path for reference
        self._log_file = log_file

    def on_mount(self) -> None:
        """Handle mount event - push the main screen."""
        from .screens.main import MainScreen
        self.push_screen(MainScreen())

    def action_quit(self) -> None:
        """Quit immediately - no waiting for cleanup."""
        import os
        import signal

        # Force kill llama-server process immediately (don't wait)
        if self._llama_process and self._llama_process.process:
            try:
                self._llama_process.process.kill()
            except Exception:
                pass
            self._llama_process = None

        # Stop SDK server (quick - just set flag and close socket)
        try:
            from ..http_server import force_stop_sdk_server
            force_stop_sdk_server()
        except Exception:
            pass

        # Close tunnel WebSocket synchronously if possible
        if self._tunnel and hasattr(self._tunnel, 'ws') and self._tunnel.ws:
            try:
                # Just close the underlying socket, don't await graceful close
                if hasattr(self._tunnel.ws, 'transport') and self._tunnel.ws.transport:
                    self._tunnel.ws.transport.close()
            except Exception:
                pass
            self._tunnel = None

        # Exit immediately
        self.exit()
