"""Server monitor widget - compact horizontal status line."""

from typing import Optional

import httpx
from textual.widgets import Static
from textual.timer import Timer
from rich.text import Text
from rich.style import Style

from ..styles import FG, FG_DIM, GREEN, YELLOW, RED, CYAN


class ServerMonitor(Static):
    """Compact horizontal server status line for bottom right."""

    DEFAULT_CSS = f"""
    ServerMonitor {{
        width: auto;
        height: 1;
        padding: 0 1;
        background: transparent;
        text-align: right;
        content-align: right middle;
    }}
    """

    def __init__(self) -> None:
        super().__init__()
        self._server_status: str = "stopped"  # stopped, loading, running
        self._requests: int = 0
        self._tokens: int = 0
        self._active_slots: int = 0
        self._total_slots: int = 0
        self._last_tps: float = 0.0
        self._port: int = 8080
        self._poll_timer: Optional[Timer] = None

    def on_mount(self) -> None:
        """Start polling when mounted."""
        self._poll_timer = self.set_interval(3.0, self._poll_server)
        self.refresh()

    def on_unmount(self) -> None:
        """Stop polling when unmounted."""
        if self._poll_timer:
            self._poll_timer.stop()

    def set_port(self, port: int) -> None:
        """Set the server port to poll."""
        self._port = port

    def set_server_status(self, status: str) -> None:
        """Manually set server status."""
        self._server_status = status
        self.refresh()

    def record_request(self, tokens: int, tps: float) -> None:
        """Record a completed request."""
        self._requests += 1
        self._tokens += tokens
        self._last_tps = tps
        self.refresh()

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self._requests = 0
        self._tokens = 0
        self._last_tps = 0.0
        self._active_slots = 0
        self.refresh()

    async def _poll_server(self) -> None:
        """Poll the server for status."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                # Check health
                try:
                    response = await client.get(f"http://127.0.0.1:{self._port}/health")
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("status") == "ok":
                            self._server_status = "running"
                        else:
                            self._server_status = "loading"
                    elif response.status_code == 503:
                        self._server_status = "loading"
                    else:
                        self._server_status = "error"
                except httpx.ConnectError:
                    self._server_status = "stopped"
                except Exception:
                    self._server_status = "error"

                # Check slots if server is running
                if self._server_status == "running":
                    try:
                        response = await client.get(f"http://127.0.0.1:{self._port}/slots")
                        if response.status_code == 200:
                            slots = response.json()
                            self._total_slots = len(slots)
                            self._active_slots = sum(1 for s in slots if s.get("is_processing", False))
                    except Exception:
                        pass

                self.refresh()

        except Exception:
            self._server_status = "stopped"
            self.refresh()

    def render(self) -> Text:
        """Render compact horizontal status line."""
        text = Text()

        # Server status with icon
        text.append("Server: ", style=Style(color=FG_DIM))
        if self._server_status == "running":
            text.append("●", style=Style(color=GREEN, bold=True))
        elif self._server_status == "loading":
            text.append("○", style=Style(color=YELLOW, bold=True))
        else:
            text.append("●", style=Style(color=RED, bold=True))

        # Separator
        text.append("  │  ", style=Style(color=FG_DIM))

        # Requests
        text.append("Reqs: ", style=Style(color=FG_DIM))
        text.append(f"{self._requests}", style=Style(color=CYAN if self._requests > 0 else FG))

        text.append("  │  ", style=Style(color=FG_DIM))

        # Tokens
        text.append("Tokens: ", style=Style(color=FG_DIM))
        text.append(f"{self._tokens:,}", style=Style(color=CYAN if self._tokens > 0 else FG))

        # Speed (only if we have data)
        if self._last_tps > 0:
            text.append("  │  ", style=Style(color=FG_DIM))
            text.append(f"{self._last_tps:.1f} tk/s", style=Style(color=GREEN))

        return text

    @property
    def requests(self) -> int:
        """Get total requests processed."""
        return self._requests

    @property
    def tokens(self) -> int:
        """Get total tokens generated."""
        return self._tokens
