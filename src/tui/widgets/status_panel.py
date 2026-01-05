"""Status panel widget displaying status lines and activity log."""

from textual.containers import Vertical
from textual.widgets import Static, RichLog
from rich.text import Text
from rich.style import Style

from ..styles import CYAN, GREEN, YELLOW, RED, FG, FG_DIM


class StatusLine(Static):
    """A single status line with label and value."""

    def __init__(self, label: str, value: str = "", status: str = "normal") -> None:
        super().__init__()
        self._label = label
        self._value = value
        self._status = status

    def render(self) -> Text:
        """Render the status line."""
        text = Text()
        text.append(f"{self._label}: ", style=Style(color=FG_DIM))

        # Color based on status
        color = FG
        if self._status == "success":
            color = GREEN
        elif self._status == "warning":
            color = YELLOW
        elif self._status == "error":
            color = RED
        elif self._status == "info":
            color = CYAN

        text.append(self._value, style=Style(color=color))
        return text

    def set_value(self, value: str, status: str = "normal") -> None:
        """Update the value and status."""
        self._value = value
        self._status = status
        self.refresh()


class StatusPanel(Vertical):
    """Panel displaying multiple status lines and an activity log."""

    def __init__(self) -> None:
        super().__init__()
        self._status_lines: dict[str, StatusLine] = {}

    def compose(self):
        """Compose the status panel."""
        # Status lines
        self._status_lines["server"] = StatusLine("Server", "Not started")
        self._status_lines["model"] = StatusLine("Model", "None loaded")
        self._status_lines["backend"] = StatusLine("Backend", "Disconnected")
        self._status_lines["nli"] = StatusLine("NLI", "Not initialized")

        yield self._status_lines["server"]
        yield self._status_lines["model"]
        yield self._status_lines["backend"]
        yield self._status_lines["nli"]
        yield Static("")  # Spacer
        yield RichLog(id="activity-log", highlight=True, markup=True)

    def set_status(self, line_id: str, value: str, status: str = "normal") -> None:
        """Update a status line."""
        if line_id in self._status_lines:
            self._status_lines[line_id].set_value(value, status)

    def log(self, message: str, level: str = "info") -> None:
        """Add a message to the activity log."""
        log_widget = self.query_one("#activity-log", RichLog)

        # Format based on level
        prefix = ""
        if level == "error":
            prefix = f"[{RED}]ERROR:[/] "
        elif level == "warning":
            prefix = f"[{YELLOW}]WARN:[/] "
        elif level == "success":
            prefix = f"[{GREEN}]OK:[/] "
        elif level == "info":
            prefix = f"[{CYAN}]INFO:[/] "

        log_widget.write(f"{prefix}{message}")

    def clear_log(self) -> None:
        """Clear the activity log."""
        log_widget = self.query_one("#activity-log", RichLog)
        log_widget.clear()
