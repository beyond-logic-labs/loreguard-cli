"""Terminal UI utilities using Rich.

This module provides a clean, Rich-based TUI that replaces custom ANSI handling.
For interactive prompts (menus, inputs), use InquirerPy in the wizard.
"""

import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TaskProgressColumn
from rich.table import Table
from rich.text import Text


# Global console instance
console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# Print helpers
# ═══════════════════════════════════════════════════════════════════════════════

def print_success(message: str):
    """Print a success message."""
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str):
    """Print an error message."""
    console.print(f"[red]✗[/red] {message}")


def print_info(message: str):
    """Print an info message."""
    console.print(f"[cyan]→[/cyan] {message}")


def print_warning(message: str):
    """Print a warning message."""
    console.print(f"[yellow]![/yellow] {message}")


def print_header(title: str):
    """Print a styled header."""
    console.print()
    console.rule(f"[bold magenta]{title}[/bold magenta]", style="cyan")
    console.print()


def show_cursor():
    """Show the terminal cursor."""
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()


def hide_cursor():
    """Hide the terminal cursor."""
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════════════
# Progress Display - uses Rich Progress
# ═══════════════════════════════════════════════════════════════════════════════

class ProgressDisplay:
    """Progress bar display using Rich Progress.

    Example:
        progress = ProgressDisplay(
            title="Downloading Model",
            total=1000000,
            subtitle="qwen3-4b.gguf",
        )
        progress.update(50000, "50 KB / 1 MB")
        progress.clear()
    """

    def __init__(
        self,
        title: str = "Progress",
        total: int = 100,
        subtitle: str = "",
        footer: str = "",
    ):
        self.title = title
        self.total = total
        self.subtitle = subtitle
        self.footer = footer
        self._progress: Optional[Progress] = None
        self._task_id = None
        self._live: Optional[Live] = None
        self._started = False

    def _start(self):
        """Start the progress display."""
        if self._started:
            return

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TextColumn("[dim]{task.fields[status]}"),
            console=console,
            transient=True,
        )
        self._task_id = self._progress.add_task(
            self.title,
            total=self.total,
            status=self.subtitle,
        )
        self._progress.start()
        self._started = True

    def update(self, current: int, status: str = ""):
        """Update progress."""
        if not self._started:
            self._start()

        # Update total if it changed
        if self._progress and self._task_id is not None:
            self._progress.update(
                self._task_id,
                completed=current,
                total=self.total,
                status=status or self.subtitle,
            )

    def clear(self):
        """Stop and clear the progress display."""
        if self._progress:
            self._progress.stop()
            self._progress = None
        self._started = False


# ═══════════════════════════════════════════════════════════════════════════════
# Live Status Display - uses Rich Live
# ═══════════════════════════════════════════════════════════════════════════════

class LiveStatusDisplay:
    """Rich Live-based status display with integrated message log.

    Uses Rich's Live display which properly handles concurrent output,
    avoiding cursor positioning issues.

    Example:
        status = LiveStatusDisplay(title="Loreguard Running")
        status.set_line("server", "llama-server", "✓ Running")
        status.set_line("model", "Model", "qwen3-4b.gguf")
        status.log("Request completed", "success")
        status.start()
        # ... do work ...
        status.stop()
    """

    def __init__(self, title: str = ""):
        self.title = title
        self._status_lines: dict[str, tuple[str, str]] = {}
        self._line_order: list[str] = []
        self.log_messages: deque[tuple[datetime, str, str]] = deque(maxlen=5)
        self.footer = ""
        self._live: Optional[Live] = None

    def set_line(self, key: str, label: str, value: str):
        """Set a status line by key. Lines are displayed in insertion order."""
        if key not in self._status_lines:
            self._line_order.append(key)
        self._status_lines[key] = (label, value)
        self._update()

    def set_title(self, title: str):
        """Update the panel title."""
        self.title = title
        self._update()

    def set_footer(self, footer: str):
        """Update the panel footer."""
        self.footer = footer
        self._update()

    def log(self, message: str, level: str = "info"):
        """Add a log message to the activity section."""
        self.log_messages.append((datetime.now(), level, message))
        self._update()

    def start(self):
        """Start the live display."""
        self._live = Live(
            self._render(),
            console=console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.start()

    def stop(self):
        """Stop the live display."""
        if self._live:
            self._live.stop()
            self._live = None

    def _update(self):
        """Update the display if running."""
        if self._live:
            self._live.update(self._render())

    def _render(self):
        """Build the Rich renderable panel."""
        # Status table with aligned columns
        table = Table.grid(padding=(0, 2))
        table.add_column(style="cyan", justify="right", min_width=12)
        table.add_column()

        for key in self._line_order:
            if key not in self._status_lines:
                continue
            label, value = self._status_lines[key]
            if not label:  # Empty line / spacer
                table.add_row("", "")
                continue

            # Color based on status indicators
            if "✓" in value:
                style = "green"
            elif "✗" in value:
                style = "red"
            elif "..." in value or "ing" in value.lower():
                style = "yellow"
            else:
                style = ""
            table.add_row(f"{label}:", Text(value, style=style))

        # Build log section
        log_lines = []
        for ts, level, msg in self.log_messages:
            time_str = ts.strftime("%H:%M:%S")
            color_map = {
                "info": "cyan",
                "success": "green",
                "error": "red",
                "warn": "yellow",
            }
            color = color_map.get(level, "white")
            log_lines.append(f"[dim]{time_str}[/dim] [{color}]{msg}[/{color}]")

        if log_lines:
            log_text = "\n".join(log_lines)
        else:
            log_text = "[dim]Waiting for requests...[/dim]"

        # Combine into group
        separator = Text("─── Recent Activity ───", style="dim")
        log_content = Text.from_markup(log_text)
        content = Group(table, Text(""), separator, log_content)

        # Build panel with title and footer
        return Panel(
            content,
            title=f"[bold cyan]{self.title}[/bold cyan]",
            subtitle=f"[dim]{self.footer}[/dim]" if self.footer else None,
            border_style="cyan",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Menu Item
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MenuItem:
    """A menu item."""
    label: str
    value: str
    description: str = ""
    disabled: bool = False
    tag: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# Menu - Simple Rich-based menu using numbered choices
# ═══════════════════════════════════════════════════════════════════════════════

class Menu:
    """Simple menu using Rich console prompts.

    Since we're running inside Textual, this provides a fallback for legacy code.
    """

    def __init__(
        self,
        items: list,
        title: str = "",
        prompt: str = "",
        allow_cancel: bool = True,
    ):
        self.items = items
        self.title = title
        self.prompt = prompt
        self.allow_cancel = allow_cancel

    def run(self) -> Optional[MenuItem]:
        """Run the menu and return selected item or None if cancelled."""
        if not self.items:
            return None

        console.print()
        if self.title:
            console.print(f"[bold cyan]{self.title}[/bold cyan]")
        if self.prompt:
            console.print(f"[dim]{self.prompt}[/dim]")
        console.print()

        for i, item in enumerate(self.items, 1):
            tag_str = f" [dim][{item.tag}][/dim]" if item.tag else ""
            console.print(f"  [cyan]{i}.[/cyan] {item.label}{tag_str}")
            if item.description:
                console.print(f"     [dim]{item.description}[/dim]")

        console.print()
        try:
            choice = console.input("[cyan]Choice (number or q to cancel):[/cyan] ")
            if choice.lower() in ('q', 'quit', 'cancel', ''):
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(self.items):
                return self.items[idx]
            return None
        except (ValueError, EOFError, KeyboardInterrupt):
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# Input Field - Simple Rich-based input
# ═══════════════════════════════════════════════════════════════════════════════

class InputField:
    """Simple input field using Rich console prompts."""

    def __init__(
        self,
        prompt: str = "Enter value:",
        default: str = "",
        password: bool = False,
        validator=None,
    ):
        self.prompt = prompt
        self.default = default
        self.password = password
        self.validator = validator

    def run(self, title: str = "") -> Optional[str]:
        """Run the input and return value or None if cancelled."""
        console.print()
        if title:
            console.print(f"[bold cyan]{title}[/bold cyan]")

        try:
            if self.password:
                import getpass
                value = getpass.getpass(f"{self.prompt} ")
            else:
                value = console.input(f"[cyan]{self.prompt}[/cyan] ")

            if self.validator:
                error = self.validator(value)
                if error:
                    console.print(f"[red]✗ {error}[/red]")
                    return self.run(title)

            return value if value else self.default

        except (EOFError, KeyboardInterrupt):
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# Legacy exports for compatibility
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    """ANSI color codes - kept for compatibility, prefer Rich markup instead."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    MUTED = "\033[90m"


def supports_color() -> bool:
    """Check if terminal supports colors."""
    return console.is_terminal


def check_for_cancel() -> bool:
    """Check if user pressed Escape. Returns False (not implemented with Rich)."""
    return False
