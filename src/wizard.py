#!/usr/bin/env python3
"""Loreguard Wizard - Interactive terminal setup wizard.

Full-screen TUI with Rich, featuring alternate buffer mode.
"""

import asyncio
import logging
import os
import platform
import signal
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable, List


@contextmanager
def suppress_external_output():
    """Suppress stdout/stderr from external libraries during TUI mode."""
    # Save original
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    # Redirect to devnull
    devnull = open(os.devnull, 'w')
    sys.stdout = devnull
    sys.stderr = devnull

    # Also suppress transformers/pytorch logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    try:
        yield
    finally:
        # Restore
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TaskProgressColumn
from rich.table import Table
from rich.text import Text
from rich.style import Style
from rich.box import ROUNDED
from rich.align import Align
from rich.layout import Layout

# Logger instance
log = logging.getLogger("loreguard")

# Module-level verbose flag
_verbose = False

# ═══════════════════════════════════════════════════════════════════════════════
# Theme - Loreguard color scheme
# ═══════════════════════════════════════════════════════════════════════════════

class Theme:
    """Loreguard color palette."""
    # Primary colors (from Crush)
    PINK = "#FF79C6"      # Crush's pink/magenta
    PURPLE = "#BD93F9"    # Crush's purple
    CYAN = "#8BE9FD"
    GREEN = "#50FA7B"
    YELLOW = "#F1FA8C"
    RED = "#FF5555"
    ORANGE = "#FFB86C"

    # Selection highlight (Crush uses this blue-purple)
    SELECTED_BG = "#44475A"

    # Neutrals
    FG = "#F8F8F2"
    FG_DIM = "#6272A4"
    BG = "#282A36"
    BG_DARKER = "#1E1F29"

    # Styles
    ACCENT = Style(color=PINK, bold=True)
    TITLE = Style(color=PINK, bold=True)  # Pink like Crush
    SUCCESS = Style(color=GREEN)
    ERROR = Style(color=RED)
    WARNING = Style(color=YELLOW)
    INFO = Style(color=CYAN)
    DIM = Style(color=FG_DIM)
    SELECTED = Style(color=FG, bold=True)
    UNSELECTED = Style(color=FG_DIM)


# Simple banner with LORE (cyan) GUARD (pink) in a # rectangle
def get_banner(width: int) -> str:
    """Generate simple banner with LOREGUARD logo."""
    # Calculate inner width for the box (min 30, max width-4)
    inner_width = min(max(30, width - 4), 50)

    # Build the box
    border_char = "#"
    top_border = border_char * (inner_width + 2)
    empty_line = f"{border_char}{' ' * inner_width}{border_char}"

    # LORE (cyan) + GUARD (pink) centered
    lore_guard = "[bold #8BE9FD]LORE[/][bold #FF79C6]GUARD[/]"
    # For centering calculation, actual text is 9 chars (LOREGUARD)
    text_len = 9
    padding = (inner_width - text_len) // 2
    logo_line = f"{border_char}{' ' * padding}{lore_guard}{' ' * (inner_width - padding - text_len)}{border_char}"

    # Subtitle centered
    subtitle = "Local inference for your game NPCs"
    sub_padding = (inner_width - len(subtitle)) // 2
    subtitle_line = f"[#6272A4]{border_char}{' ' * sub_padding}{subtitle}{' ' * (inner_width - sub_padding - len(subtitle))}{border_char}[/]"

    lines = [
        f"[#6272A4]{top_border}[/]",
        f"[#6272A4]{empty_line}[/]",
        f"[#6272A4]{border_char}[/]{' ' * padding}{lore_guard}{' ' * (inner_width - padding - text_len)}[#6272A4]{border_char}[/]",
        subtitle_line,
        f"[#6272A4]{empty_line}[/]",
        f"[#6272A4]{top_border}[/]",
    ]

    return "\n".join(lines)


def get_hardware_panel(hardware_info: str) -> Text:
    """Create a detailed hardware info display."""
    if not hardware_info:
        return Text("")

    # Parse hardware info (format: "CPU • RAM • GPU")
    parts = hardware_info.split(" • ") if " • " in hardware_info else [hardware_info]

    text = Text()
    text.append("\n")

    # Display each hardware component
    labels = ["CPU", "RAM", "GPU"]

    for i, part in enumerate(parts):
        label = labels[i] if i < len(labels) else ""

        text.append("  ", style="")
        if label and "GB" not in part:  # CPU doesn't have "GB"
            text.append(f"{label}: ", style=Theme.FG_DIM)
        elif "GB RAM" in part:
            text.append("RAM: ", style=Theme.FG_DIM)
            part = part.replace(" RAM", "")
        elif i == 2 or "GPU" in part.upper() or "M1" in part or "M2" in part or "M3" in part:
            text.append("GPU: ", style=Theme.FG_DIM)

        text.append(part, style=Theme.CYAN)
        text.append("\n")

    return text


# Global console with force_terminal for alt screen support
console = Console(force_terminal=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Full-screen App - Persistent shell with Live display
# ═══════════════════════════════════════════════════════════════════════════════

class TUIApp:
    """Persistent TUI shell - everything renders inside this container."""

    def __init__(self):
        self.hardware_info: str = ""
        self._content: RenderableType = Text("")
        self._title: str = ""
        self._footer: str = ""
        self._live: Optional[Live] = None
        self._in_alt_screen = False

    def _render(self) -> RenderableType:
        """Build the full layout: banner + hardware + content area."""
        parts = []

        # Banner
        width = console.size.width
        banner = get_banner(width)
        parts.append(Text.from_markup(banner))

        # Hardware info
        if self.hardware_info:
            parts.append(get_hardware_panel(self.hardware_info))

        parts.append(Text(""))

        # Content area (modal with title)
        if self._content:
            term_width = console.size.width
            modal_width = min(70, term_width - 4)

            # Build title text
            title_text = None
            if self._title:
                title_text = Text(f" {self._title} ", style=f"bold {Theme.PINK}")

            # Build footer text
            footer_text = None
            if self._footer:
                footer_text = Text(f" {self._footer} ", style=Theme.FG_DIM)

            modal = Panel(
                self._content,
                title=title_text,
                subtitle=footer_text,
                border_style=Theme.PURPLE,
                box=ROUNDED,
                width=modal_width,
                padding=(0, 1),
            )
            parts.append(Align.center(modal, width=term_width))

        return Group(*parts)

    def start(self):
        """Enter alternate screen and start Live display."""
        console.set_alt_screen(True)
        self._in_alt_screen = True
        hide_cursor()
        self._live = Live(
            self._render(),
            console=console,
            refresh_per_second=4,
            screen=True,
            vertical_overflow="visible",
        )
        self._live.start()

    def stop(self):
        """Stop Live display and exit alternate screen."""
        if self._live:
            self._live.stop()
            self._live = None
        if self._in_alt_screen:
            show_cursor()
            console.set_alt_screen(False)
            self._in_alt_screen = False

    def set_content(self, content: RenderableType, title: str = "", footer: str = ""):
        """Update the content area. This is the main way to render things."""
        self._content = content
        self._title = title
        self._footer = footer
        if self._live:
            self._live.update(self._render())

    def draw(self, content: RenderableType, title: str = ""):
        """Legacy draw method - now uses set_content."""
        self.set_content(content, title=title)


# Global app instance
_app: Optional[TUIApp] = None


def get_app() -> TUIApp:
    """Get or create the global app instance."""
    global _app
    if _app is None:
        _app = TUIApp()
    return _app


def _configure_logging(verbose: bool = False) -> Optional[Path]:
    """Configure logging level based on verbose flag."""
    if verbose:
        log_file = Path.cwd() / "loreguard-debug.log"
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(logging.Formatter("%(message)s"))

        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        root_logger.setLevel(logging.DEBUG)
        log.setLevel(logging.DEBUG)
        return log_file
    else:
        logging.basicConfig(level=logging.WARNING, format="%(message)s")
        log.setLevel(logging.WARNING)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Terminal Input (cross-platform)
# ═══════════════════════════════════════════════════════════════════════════════

def getch() -> str:
    """Get a single keypress."""
    if sys.platform == 'win32':
        import msvcrt
        ch = msvcrt.getwch()
        if ch in ('\x00', '\xe0'):
            ch2 = msvcrt.getwch()
            if ch2 == 'H': return 'UP'
            if ch2 == 'P': return 'DOWN'
            if ch2 == 'K': return 'LEFT'
            if ch2 == 'M': return 'RIGHT'
            return ch2
        return ch
    else:
        import tty
        import termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                ch2 = sys.stdin.read(1)
                if ch2 == '[':
                    ch3 = sys.stdin.read(1)
                    if ch3 == 'A': return 'UP'
                    if ch3 == 'B': return 'DOWN'
                    if ch3 == 'C': return 'RIGHT'
                    if ch3 == 'D': return 'LEFT'
                return 'ESC'
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def hide_cursor():
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()


def show_cursor():
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════════════
# TUI Components
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MenuItem:
    """A menu item."""
    label: str
    value: str
    description: str = ""
    tag: str = ""
    disabled: bool = False


class SelectMenu:
    """Crush-style menu with searchable filter and arrow key navigation."""

    # Selection highlight background color (Crush's exact color)
    SELECTED_BG = Theme.SELECTED_BG

    def __init__(
        self,
        items: List[MenuItem],
        title: str = "",
        subtitle: str = "",
        width: int = 60,
        filterable: bool = True,
    ):
        self.items = items
        self.title = title
        self.subtitle = subtitle
        self.selected = 0
        self.width = width
        self.filterable = filterable
        self.filter_text = ""

    def _get_filtered_items(self) -> List[MenuItem]:
        """Get items matching the current filter."""
        if not self.filter_text:
            return self.items
        query = self.filter_text.lower()
        return [item for item in self.items if query in item.label.lower()]

    def run(self, app: Optional["TUIApp"] = None) -> Optional[MenuItem]:
        """Run the menu and return selected item."""
        if not self.items:
            return None

        show_cursor()  # Show cursor for filter input
        try:
            self._draw(app)

            while True:
                key = getch()
                filtered = self._get_filtered_items()

                if key == 'UP' or key == 'k':
                    if filtered:
                        self.selected = max(0, self.selected - 1)
                elif key == 'DOWN' or key == 'j':
                    if filtered:
                        self.selected = min(len(filtered) - 1, self.selected + 1)
                elif key in ('\r', '\n'):
                    if filtered:
                        return filtered[self.selected]
                elif key == 'ESC':
                    return None
                elif key == '\x03':  # Ctrl+C
                    raise KeyboardInterrupt()
                elif key in ('\x7f', '\x08'):  # Backspace
                    if self.filterable:
                        self.filter_text = self.filter_text[:-1]
                        self.selected = 0  # Reset selection on filter change
                elif self.filterable and len(key) == 1 and ord(key) >= 32:
                    self.filter_text += key
                    self.selected = 0  # Reset selection on filter change

                self._draw(app)
        finally:
            hide_cursor()

    def _draw(self, app: Optional["TUIApp"] = None):
        """Draw the menu."""
        content = self._render_content()
        if app:
            app.draw(content, title=self.title)
        else:
            console.clear()
            console.print(Panel(content, border_style=Theme.PURPLE, box=ROUNDED, padding=(0, 1)))

    def _render_content(self) -> Group:
        """Render Crush-style menu content with filter."""
        lines = []
        inner_width = self.width - 4  # Account for panel borders and padding
        filtered = self._get_filtered_items()

        # Filter input (like Crush's "> Type to filter")
        if self.filterable:
            filter_line = Text()
            filter_line.append(" > ", style=f"bold {Theme.PINK}")
            if self.filter_text:
                filter_line.append(self.filter_text, style=Theme.FG)
            else:
                filter_line.append("Type to filter", style=Theme.FG_DIM)
            filter_line.append("▋", style=Theme.PINK)  # Cursor
            lines.append(filter_line)
        elif self.subtitle:
            # Show subtitle only when filter is disabled
            lines.append(Text(f" {self.subtitle}", style=Theme.FG_DIM))
            lines.append(Text(""))

        # Menu items with full-width highlight for selected
        for i, item in enumerate(filtered):
            is_selected = i == self.selected

            # Build the line content
            label = item.label
            tag = f"  {item.tag}" if item.tag else ""

            if is_selected:
                # Full-width highlighted row with cyan background (like Crush)
                line = Text()
                line.append(" " + label, style=f"bold white on {Theme.CYAN}")
                # Pad to fill width, then add tag
                padding = inner_width - len(label) - len(tag) - 1
                line.append(" " * max(1, padding), style=f"on {Theme.CYAN}")
                if tag:
                    line.append(tag, style=f"bold white on {Theme.CYAN}")
                lines.append(line)
            else:
                # Non-selected items in WHITE (not dim)
                line = Text()
                line.append(" " + label, style="white")
                if tag:
                    padding = inner_width - len(label) - len(tag) - 1
                    line.append(" " * max(1, padding))
                    tag_style = Theme.GREEN if "✓" in tag else Theme.FG_DIM
                    line.append(tag, style=tag_style)
                lines.append(line)

        # Show "No matches" if filter has no results
        if not filtered and self.filter_text:
            lines.append(Text(" No matches", style=Theme.FG_DIM))

        # Footer with key hints (Crush style)
        lines.append(Text(""))
        footer = Text()
        footer.append(" ↑↓ ", style=f"bold {Theme.FG}")
        footer.append("choose  ", style=Theme.FG_DIM)
        footer.append("enter ", style=f"bold {Theme.FG}")
        footer.append("confirm  ", style=Theme.FG_DIM)
        footer.append("esc ", style=f"bold {Theme.FG}")
        footer.append("cancel", style=Theme.FG_DIM)
        lines.append(footer)

        return Group(*lines)


class TextInput:
    """Crush-styled text input with cursor."""

    def __init__(
        self,
        prompt: str = "Enter value:",
        password: bool = False,
        validator: Optional[Callable[[str], Optional[str]]] = None,
        title: str = "",
    ):
        self.prompt = prompt
        self.password = password
        self.validator = validator
        self.title = title
        self.value = ""
        self.error: Optional[str] = None

    def run(self, title: str = "", app: Optional["TUIApp"] = None) -> Optional[str]:
        """Run the input and return value."""
        show_cursor()
        try:
            self._draw(app)

            while True:
                key = getch()

                if key in ('\r', '\n'):
                    if self.validator:
                        self.error = self.validator(self.value)
                        if self.error:
                            self._draw(app)
                            continue
                    return self.value
                elif key == 'ESC':
                    return None
                elif key in ('\x7f', '\x08'):  # Backspace
                    self.value = self.value[:-1]
                    self.error = None
                elif key == '\x03':  # Ctrl+C
                    raise KeyboardInterrupt()
                elif len(key) == 1 and ord(key) >= 32:
                    self.value += key
                    self.error = None

                self._draw(app)
        finally:
            hide_cursor()

    def _draw(self, app: Optional["TUIApp"] = None):
        """Draw the input."""
        content = self._render_content()
        title = self.title
        if app:
            app.draw(content, title=title)
        else:
            console.clear()
            console.print(Panel(content, border_style=Theme.PURPLE, box=ROUNDED, padding=(0, 1)))

    def _render_content(self) -> Group:
        """Render Crush-style input content."""
        lines = []

        # Prompt (like "> Type to filter" in Crush)
        prompt_line = Text()
        prompt_line.append(" > ", style=f"bold {Theme.CYAN}")
        prompt_line.append(self.prompt, style=Theme.FG_DIM)
        lines.append(prompt_line)

        # Input field with cursor
        display = "•" * len(self.value) if self.password else self.value
        input_line = Text()
        input_line.append("   " + display, style=Theme.FG)
        input_line.append("▋", style=Theme.PINK)
        lines.append(input_line)

        # Error
        if self.error:
            lines.append(Text(""))
            lines.append(Text(f"   ✗ {self.error}", style=Theme.ERROR))

        # Footer with key hints
        lines.append(Text(""))
        footer = Text()
        footer.append(" enter ", style=f"bold {Theme.FG}")
        footer.append("confirm  ", style=Theme.FG_DIM)
        footer.append("esc ", style=f"bold {Theme.FG}")
        footer.append("cancel", style=Theme.FG_DIM)
        lines.append(footer)

        return Group(*lines)


class ProgressDisplay:
    """Styled progress display."""

    def __init__(self, title: str = "", total: int = 100):
        self.title = title
        self.total = total
        self._progress: Optional[Progress] = None
        self._task_id = None
        self._started = False

    def _start(self):
        if self._started:
            return

        self._progress = Progress(
            SpinnerColumn(style=Theme.PINK),
            TextColumn("[bold]{task.description}"),
            BarColumn(complete_style=Theme.PINK, finished_style=Theme.GREEN),
            TaskProgressColumn(),
            TextColumn("[dim]{task.fields[status]}"),
            console=console,
            transient=True,
        )
        self._task_id = self._progress.add_task(self.title, total=self.total, status="")
        self._progress.start()
        self._started = True

    def update(self, current: int, status: str = ""):
        if not self._started:
            self._start()
        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, completed=current, total=self.total, status=status)

    def clear(self):
        if self._progress:
            self._progress.stop()
            self._progress = None
        self._started = False


class StatusPanel:
    """Styled live status display - renders inside TUIApp."""

    def __init__(self, title: str = "", app: Optional['TUIApp'] = None):
        self.title = title
        self._app = app
        self._lines: dict[str, tuple[str, str, str]] = {}  # key -> (label, value, style)
        self._order: list[str] = []
        self._logs: list[tuple[str, str, str]] = []  # (time, level, msg)
        self.footer = ""

    def set_line(self, key: str, label: str, value: str, style: str = ""):
        if key not in self._lines:
            self._order.append(key)
        self._lines[key] = (label, value, style)
        self._update()

    def log(self, message: str, level: str = "info"):
        from datetime import datetime
        self._logs.append((datetime.now().strftime("%H:%M:%S"), level, message))
        if len(self._logs) > 5:
            self._logs.pop(0)
        self._update()

    def _render_content(self) -> RenderableType:
        """Render just the content (for embedding in TUIApp)."""
        lines = []

        # Status lines
        for key in self._order:
            if key not in self._lines:
                continue
            label, value, style = self._lines[key]
            if not label:
                lines.append(Text(""))
                continue

            line = Text()
            line.append(f"  {label}: ", style=Theme.DIM)

            value_style = Theme.SUCCESS if "✓" in value else Theme.WARNING if "..." in value else style or Theme.FG
            line.append(value, style=value_style)
            lines.append(line)

        # Logs section
        if self._logs:
            lines.append(Text(""))
            lines.append(Text("  ─── Recent Activity ───", style=Theme.DIM))
            for ts, level, msg in self._logs:
                log_style = {"info": Theme.INFO, "success": Theme.SUCCESS, "error": Theme.ERROR, "warn": Theme.WARNING}.get(level, Theme.DIM)
                log_line = Text(f"  {ts} ", style=Theme.DIM)
                log_line.append(msg, style=log_style)
                lines.append(log_line)

        return Group(*lines) if lines else Text("")

    def start(self):
        """Start rendering - uses TUIApp if available."""
        self._update()

    def stop(self):
        """Stop rendering - no-op when using TUIApp."""
        pass

    def _update(self):
        """Update the display through TUIApp."""
        if self._app:
            self._app.set_content(self._render_content(), title=self.title, footer=self.footer)


class CommandPalette:
    """Styled command palette (/ menu)."""

    def __init__(self, commands: List[tuple[str, str, Callable]]):
        """commands: list of (key, description, callback)"""
        self.commands = commands
        self.selected = 0
        self.filter = ""

    def _get_filtered(self) -> List[tuple[str, str, Callable]]:
        if not self.filter:
            return self.commands
        return [c for c in self.commands if self.filter.lower() in c[1].lower()]

    def _render(self) -> Panel:
        filtered = self._get_filtered()
        lines = []

        # Search input
        search_line = Text("  /", style=Theme.PINK)
        search_line.append(self.filter, style=Theme.FG)
        search_line.append("▋", style=Theme.PINK)
        lines.append(search_line)
        lines.append(Text(""))

        # Commands
        for i, (key, desc, _) in enumerate(filtered):
            is_selected = i == self.selected

            if is_selected:
                line = Text("  ❯ ", style=Theme.SELECTED)
                line.append(key, style=Theme.SELECTED)
                line.append(f"  {desc}", style=Theme.DIM)
            else:
                line = Text("    ", style=Theme.UNSELECTED)
                line.append(key, style=Theme.UNSELECTED)
                line.append(f"  {desc}", style=Theme.DIM)
            lines.append(line)

        if not filtered:
            lines.append(Text("    No matching commands", style=Theme.DIM))

        return Panel(
            Group(*lines),
            title=Text("Commands", style=Theme.TITLE),
            border_style=Theme.PURPLE,
            box=ROUNDED,
            padding=(1, 2),
        )

    def run(self) -> Optional[Callable]:
        """Run palette and return selected callback."""
        hide_cursor()
        try:
            console.clear()
            console.print(self._render())

            while True:
                key = getch()
                filtered = self._get_filtered()

                if key == 'UP' or key == 'k':
                    self.selected = max(0, self.selected - 1)
                elif key == 'DOWN' or key == 'j':
                    self.selected = min(len(filtered) - 1, self.selected + 1) if filtered else 0
                elif key in ('\r', '\n'):
                    if filtered and 0 <= self.selected < len(filtered):
                        return filtered[self.selected][2]
                    return None
                elif key == 'ESC' or key == '\x03':
                    return None
                elif key in ('\x7f', '\x08'):  # Backspace
                    self.filter = self.filter[:-1]
                    self.selected = 0
                elif len(key) == 1 and ord(key) >= 32:
                    self.filter += key
                    self.selected = 0

                console.clear()
                console.print(self._render())
        finally:
            show_cursor()


# ═══════════════════════════════════════════════════════════════════════════════
# Print helpers
# ═══════════════════════════════════════════════════════════════════════════════

def print_success(message: str):
    console.print(Text(f"  ✓ {message}", style=Theme.SUCCESS))


def print_error(message: str):
    console.print(Text(f"  ✗ {message}", style=Theme.ERROR))


def print_info(message: str):
    console.print(Text(f"  → {message}", style=Theme.INFO))


def print_warning(message: str):
    console.print(Text(f"  ! {message}", style=Theme.WARNING))


def print_banner():
    """Print the styled banner."""
    width = console.size.width
    banner = get_banner(width)
    console.print(Text.from_markup(banner))


# ═══════════════════════════════════════════════════════════════════════════════
# Hardware Detection
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HardwareInfo:
    cpu: str
    ram_gb: Optional[float]
    gpu: str
    gpu_vram_gb: Optional[float]
    gpu_mem_type: Optional[str]


def _run_cmd(args: list[str]) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=False, timeout=3)
        return result.stdout.strip()
    except Exception:
        return ""


def _get_cpu_name() -> str:
    cpu = platform.processor().strip()
    if cpu:
        return cpu
    if sys.platform == "darwin":
        cpu = _run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except:
            pass
    return cpu or "Unknown"


def _get_ram_gb() -> Optional[float]:
    if sys.platform == "darwin":
        mem = _run_cmd(["sysctl", "-n", "hw.memsize"])
        if mem.isdigit():
            return round(int(mem) / (1024 ** 3), 1)
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        if len(parts) >= 2 and parts[1].isdigit():
                            return round(int(parts[1]) * 1024 / (1024 ** 3), 1)
        except:
            pass
    return None


def _get_gpu_info() -> tuple[str, Optional[float], Optional[str]]:
    if sys.platform == "darwin":
        output = _run_cmd(["system_profiler", "SPDisplaysDataType"])
        names = []
        vram = None
        mem_type = None
        for line in output.splitlines():
            if "Chipset Model:" in line:
                names.append(line.split(":", 1)[1].strip())
            if "VRAM" in line and ":" in line:
                try:
                    val = line.split(":", 1)[1].strip().split()[0]
                    vram = float(val) if val.replace(".", "").isdigit() else None
                except:
                    pass
            if "Memory Type:" in line:
                mem_type = line.split(":", 1)[1].strip()
        return ", ".join(names) if names else "Unknown", vram, mem_type
    return "Unknown", None, None


def detect_hardware() -> HardwareInfo:
    cpu = _get_cpu_name()
    ram_gb = _get_ram_gb()
    gpu, vram, mem_type = _get_gpu_info()
    return HardwareInfo(cpu=cpu, ram_gb=ram_gb, gpu=gpu, gpu_vram_gb=vram, gpu_mem_type=mem_type)


def _format_hardware(hw: HardwareInfo) -> str:
    parts = [hw.cpu]
    if hw.ram_gb:
        parts.append(f"{hw.ram_gb:.0f}GB RAM")
    if hw.gpu and hw.gpu != "Unknown":
        parts.append(hw.gpu)
    return " • ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Model helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _classify_model_fit(model_size_gb: float, hardware: Optional[HardwareInfo]) -> str:
    if not hardware or not hardware.ram_gb:
        return "unknown"
    usable = hardware.ram_gb - 2.0
    if model_size_gb <= usable:
        return "fits"
    return "too_big"


def _suggest_model_id(models, hardware: Optional[HardwareInfo]) -> Optional[str]:
    if not hardware or not hardware.ram_gb:
        return None
    ram = hardware.ram_gb
    if ram >= 16:
        preferred = ["qwen3-4b-instruct", "qwen3-8b"]
    elif ram >= 8:
        preferred = ["qwen3-4b-instruct", "llama-3.2-3b-instruct"]
    else:
        preferred = ["qwen3-1.7b"]

    for model_id in preferred:
        if any(m.id == model_id for m in models):
            return model_id
    return None


def _resolve_backend_model_id(filename_stem: str) -> str:
    mappings = {"qwen3-4b": "qwen3-4b", "qwen3-8b": "qwen3-8b", "llama-3": "llama-3.1-8b"}
    for pattern, backend_id in mappings.items():
        if pattern in filename_stem.lower():
            return backend_id
    return "external"


# ═══════════════════════════════════════════════════════════════════════════════
# Wizard Steps
# ═══════════════════════════════════════════════════════════════════════════════

async def step_authentication(app: Optional[TUIApp] = None) -> tuple[Optional[str], Optional[str], bool]:
    """Step 1: Authentication."""
    menu = SelectMenu(
        items=[
            MenuItem("Paste token", "token", "Manually enter your API token"),
            MenuItem("Dev mode", "dev", "Test locally without backend"),
        ],
        title="Step 1/4: Authentication",
        subtitle="Choose how to connect",
        filterable=False,  # Only 2 options, no need for filter
    )

    choice = menu.run(app)
    if not choice:
        return None, None, False

    if choice.value == "dev":
        if app:
            app.draw(Text(" ✓ Dev mode enabled", style=Theme.SUCCESS), title="Step 1/4: Authentication")
        return "dev_mock_token", "dev-worker", True

    # Token input
    def validate(v):
        return "Token required" if not v.strip() else None

    input_field = TextInput(
        prompt="Paste your API token",
        password=True,
        validator=validate,
        title="Step 1/4: Authentication",
    )
    token = input_field.run(app=app)

    if not token:
        return await step_authentication(app)

    # Validate token
    import httpx
    import socket

    if app:
        app.draw(Text(" → Validating token...", style=Theme.INFO), title="Step 1/4: Authentication")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.loreguard.com/api/auth/me",
                headers={"Authorization": f"Bearer {token}"},
            )
            if response.status_code == 200:
                data = response.json()
                name = data.get("studio", {}).get("name") or data.get("email", "user")
                hostname = socket.gethostname().split(".")[0]
                if app:
                    app.draw(Text(f" ✓ Authenticated as {name}", style=Theme.SUCCESS), title="Step 1/4: Authentication")
                return token, hostname, False
            else:
                if app:
                    app.draw(Text(" ✗ Authentication failed", style=Theme.ERROR), title="Step 1/4: Authentication")
                await asyncio.sleep(1)
                return await step_authentication(app)
    except Exception as e:
        if app:
            app.draw(Text(f" ✗ Connection error: {e}", style=Theme.ERROR), title="Step 1/4: Authentication")
        await asyncio.sleep(1)
        return await step_authentication(app)


async def step_model_selection(hardware: Optional[HardwareInfo], app: Optional[TUIApp] = None) -> Optional[Path]:
    """Step 2: Model selection."""
    from .models_registry import SUPPORTED_MODELS
    from .llama_server import get_models_dir

    models_dir = get_models_dir()
    suggested_id = _suggest_model_id(SUPPORTED_MODELS, hardware)

    installed = {m.id for m in SUPPORTED_MODELS if (models_dir / m.filename).exists()}

    items = []
    for model in SUPPORTED_MODELS:
        tags = []
        if model.id in installed:
            tags.append("✓ installed")
        else:
            tags.append(f"{model.size_gb:.1f}GB")
        if model.id == suggested_id:
            tags.append("suggested")
        if model.recommended:
            tags.append("recommended")

        items.append(MenuItem(
            model.name,
            model.id,
            model.description,
            tag=" • ".join(tags),
        ))

    items.append(MenuItem("Custom path...", "__custom__", "Use your own .gguf file"))

    menu = SelectMenu(
        items=items,
        title="Step 2/4: Model Selection",
        subtitle="Choose a model to run",
    )

    choice = menu.run(app)
    if not choice:
        return None

    if choice.value == "__custom__":
        input_field = TextInput(prompt="Enter path to .gguf file:")
        path_str = input_field.run(app=app)
        if not path_str:
            return await step_model_selection(hardware, app)

        model_path = Path(path_str.replace("~", str(Path.home())))
        if not model_path.exists():
            if app:
                app.draw(Text(f"✗ File not found: {model_path}", style=Theme.ERROR), title="Step 2/4: Model Selection")
            await asyncio.sleep(1)
            return await step_model_selection(hardware, app)

        if app:
            app.draw(Text(f"✓ Using: {model_path.name}", style=Theme.SUCCESS), title="Step 2/4: Model Selection")
        return model_path

    # Find model
    model = next((m for m in SUPPORTED_MODELS if m.id == choice.value), None)
    if not model:
        return None

    model_path = models_dir / model.filename

    if model_path.exists():
        if app:
            app.draw(Text(f"✓ Model ready: {model.name}", style=Theme.SUCCESS), title="Step 2/4: Model Selection")
        return model_path

    # Download
    if app:
        app.draw(Text(f"→ Downloading {model.name}...", style=Theme.INFO), title="Step 2/4: Model Selection")

    import httpx
    model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:
            async with client.stream("GET", model.url) as response:
                total = model.size_bytes or int(response.headers.get("content-length", 0))
                downloaded = 0

                with open(model_path, "wb") as f:
                    async for chunk in response.aiter_bytes(1024 * 1024):
                        f.write(chunk)
                        downloaded += len(chunk)
                        pct = int(downloaded / total * 100) if total > 0 else 0
                        mb_done = downloaded // 1024 // 1024
                        mb_total = total // 1024 // 1024
                        if app:
                            app.draw(Text(f"→ Downloading {model.name}... {pct}% ({mb_done}MB / {mb_total}MB)", style=Theme.INFO), title="Step 2/4: Model Selection")

        if app:
            app.draw(Text(f"✓ Downloaded: {model.name}", style=Theme.SUCCESS), title="Step 2/4: Model Selection")
        return model_path

    except Exception as e:
        if app:
            app.draw(Text(f"✗ Download failed: {e}", style=Theme.ERROR), title="Step 2/4: Model Selection")
        if model_path.exists():
            model_path.unlink()
        await asyncio.sleep(1)
        return await step_model_selection(hardware, app)


async def step_nli_setup(app: Optional[TUIApp] = None) -> bool:
    """Step 3: NLI setup."""
    from .nli import is_nli_model_available, download_nli_model
    import concurrent.futures

    if app:
        app.draw(Text("→ Checking NLI model...", style=Theme.INFO), title="Step 3/4: NLI Model Setup")

    if is_nli_model_available():
        if app:
            app.draw(Text("✓ NLI model ready", style=Theme.SUCCESS), title="Step 3/4: NLI Model Setup")
        return True

    if app:
        app.draw(Text("→ Downloading NLI model (~1.4GB)...", style=Theme.INFO), title="Step 3/4: NLI Model Setup")

    try:
        # Run sync download in thread to not block
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            success = await loop.run_in_executor(pool, download_nli_model)

        if success:
            if app:
                app.draw(Text("✓ NLI model ready", style=Theme.SUCCESS), title="Step 3/4: NLI Model Setup")
        else:
            if app:
                app.draw(Text("! NLI download failed - continuing without", style=Theme.WARNING), title="Step 3/4: NLI Model Setup")

        return success

    except Exception as e:
        if app:
            app.draw(Text(f"! NLI setup failed: {e}", style=Theme.WARNING), title="Step 3/4: NLI Model Setup")
        return False


async def step_start(
    model_path: Path,
    token: str,
    worker_id: str,
    dev_mode: bool,
    nli_enabled: bool = True,
    app: Optional[TUIApp] = None,
) -> int:
    """Step 4: Start services."""
    from .llama_server import LlamaServerProcess, is_llama_server_installed, download_llama_server, DownloadProgress

    # Create status panel that renders inside the TUIApp
    status = StatusPanel(title="Loreguard Running", app=app)
    status.footer = "/ commands • ctrl+c quit"
    status.start()

    # Download llama-server if needed
    if not is_llama_server_installed():
        status.set_line("server", "llama-server", "Downloading...")
        try:
            def on_progress(msg: str, prog: DownloadProgress | None):
                if prog:
                    status.set_line("server", "llama-server", f"Downloading... {int(prog.percent)}%")
            await download_llama_server(on_progress)
            status.set_line("server", "llama-server", "✓ Downloaded")
        except Exception as e:
            status.stop()
            print_error(f"Failed: {e}")
            return 1

    # Start llama-server
    status.set_line("server", "llama-server", "Starting...")
    status.set_line("model", "Model", model_path.name)

    llama = LlamaServerProcess(model_path, port=8080)
    llama.start()

    status.set_line("server", "llama-server", "Loading model...")
    ready = await llama.wait_for_ready(timeout=120.0)

    if not ready:
        status.stop()
        llama.stop()
        print_error("llama-server failed to start")
        return 1

    status.set_line("server", "llama-server", "✓ Running on :8080")

    # Connect backend
    tunnel = None
    if not dev_mode:
        status.set_line("backend", "Backend", "Connecting...")
        try:
            from .tunnel import BackendTunnel
            from .llm import LLMProxy

            llm_proxy = LLMProxy("http://127.0.0.1:8080")

            nli_service = None
            if nli_enabled:
                status.set_line("nli", "NLI", "Loading...")
                from .nli import NLIService
                nli_service = NLIService()
                # Suppress transformers/pytorch warnings during model loading
                with suppress_external_output():
                    model_loaded = nli_service.load_model()
                if model_loaded:
                    status.set_line("nli", "NLI", f"✓ Ready ({nli_service.device})")
                else:
                    status.set_line("nli", "NLI", "✗ Failed")
                    nli_service = None

            model_id = _resolve_backend_model_id(model_path.stem)
            tunnel = BackendTunnel(
                backend_url="wss://api.loreguard.com/workers",
                llm_proxy=llm_proxy,
                worker_id=worker_id,
                worker_token=token,
                model_id=model_id,
                nli_service=nli_service,
                log_callback=status.log,
            )
            asyncio.create_task(tunnel.connect())
            await asyncio.sleep(2)
            status.set_line("backend", "Backend", "✓ Connected")
        except Exception as e:
            status.set_line("backend", "Backend", f"✗ {e}")
    else:
        status.set_line("mode", "Mode", "Dev (local only)")

    # Stats
    status.set_line("spacer", "", "")
    status.set_line("requests", "Requests", "0")
    status.set_line("tokens", "Tokens", "0")

    request_count = [0]
    total_tokens = [0]

    def on_request(npc: str, tokens: int, ttft_ms: float, total_ms: float):
        request_count[0] += 1
        total_tokens[0] += tokens
        tps = (tokens / total_ms * 1000) if total_ms > 0 else 0
        status.set_line("requests", "Requests", str(request_count[0]))
        status.set_line("tokens", "Tokens", f"{total_tokens[0]:,}")
        status.log(f"{npc}: {tokens} tok @ {tps:.1f} tk/s", "success")

    if tunnel:
        tunnel.on_request_complete = on_request

    # Command palette handler
    def cmd_chat():
        return "chat"

    def cmd_monitor():
        return "monitor"

    def cmd_quit():
        return "quit"

    commands = [
        ("chat", "Chat with an NPC", cmd_chat),
        ("monitor", "View server stats", cmd_monitor),
        ("quit", "Exit Loreguard", cmd_quit),
    ]

    # Main loop
    running = True

    def handle_signal(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    import select

    try:
        while running:
            # Check for `/` key (non-blocking)
            if sys.platform != 'win32':
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = sys.stdin.read(1)
                    if ch == '/':
                        status.stop()
                        palette = CommandPalette(commands)
                        result = palette.run()
                        status.start()

                        if result:
                            action = result()
                            if action == "quit":
                                running = False
                            elif action == "chat" and not dev_mode:
                                status.stop()
                                from .npc_chat import run_npc_chat
                                try:
                                    await run_npc_chat(api_token=token, tunnel=tunnel)
                                except KeyboardInterrupt:
                                    pass
                                status.start()
                else:
                    await asyncio.sleep(0.1)
            else:
                await asyncio.sleep(0.5)

    except KeyboardInterrupt:
        pass

    # Cleanup
    status.set_line("server", "llama-server", "Stopping...")
    status.stop()

    llama.stop()
    if tunnel:
        try:
            await tunnel.disconnect()
        except:
            pass

    # Exit alt screen
    console.set_alt_screen(False)
    show_cursor()
    print_success("Goodbye!")
    return 0


async def run_wizard() -> int:
    """Run the wizard in full-screen alternate buffer mode."""
    app = get_app()
    try:
        # Detect hardware first
        hardware = detect_hardware()
        app.hardware_info = _format_hardware(hardware)

        # Start full-screen app
        app.start()

        # Step 1
        token, worker_id, dev_mode = await step_authentication(app)
        if token is None:
            return 1

        # Step 2
        model_path = await step_model_selection(hardware, app)
        if model_path is None:
            return 1

        # Step 3
        nli_enabled = await step_nli_setup(app)

        # Step 4
        return await step_start(model_path, token, worker_id, dev_mode, nli_enabled, app)

    except KeyboardInterrupt:
        return 1
    finally:
        app.stop()


def main(verbose: bool = False):
    """Entry point."""
    global _verbose
    _verbose = verbose

    log_file = _configure_logging(verbose)
    if log_file:
        print(f"Debug logging to: {log_file}")

    try:
        exit_code = asyncio.run(run_wizard())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        show_cursor()
        console.set_alt_screen(False)
        sys.exit(1)


if __name__ == "__main__":
    verbose = any(a in ('-v', '--verbose') for a in sys.argv[1:])
    main(verbose=verbose)
