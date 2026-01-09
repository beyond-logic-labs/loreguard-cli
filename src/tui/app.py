"""Main Loreguard TUI application using Textual framework."""

from pathlib import Path
from typing import Optional, Any

from textual.app import App
from textual.binding import Binding
from textual import events
from textual.driver import Driver

from .styles import LOREGUARD_CSS
from .widgets.hardware_info import HardwareData


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

    def on_mount(self) -> None:
        """Handle mount event - push the main screen."""
        from .screens.main import MainScreen
        self.push_screen(MainScreen())

    def action_quit(self) -> None:
        """Quit with cleanup."""
        # Stop SDK server
        from ..http_server import stop_sdk_server
        stop_sdk_server()

        if self._llama_process:
            self._llama_process.stop()
        self.exit()
