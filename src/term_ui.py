"""Terminal UI with colors and inline rendering (doesn't clear history)."""

import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Callable


# ═══════════════════════════════════════════════════════════════════════════════
# ANSI Colors - CDE/Motif inspired palette
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    """ANSI color codes matching CDE/Motif aesthetic."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # CDE-inspired colors
    CYAN = "\033[36m"
    BRIGHT_CYAN = "\033[96m"
    YELLOW = "\033[33m"
    BRIGHT_YELLOW = "\033[93m"
    GREEN = "\033[32m"
    BRIGHT_GREEN = "\033[92m"
    RED = "\033[31m"
    BRIGHT_RED = "\033[91m"
    MAGENTA = "\033[35m"
    BRIGHT_MAGENTA = "\033[95m"
    PINK = "\033[38;5;205m"  # Hot pink
    PURPLE = "\033[38;5;141m"  # Light purple
    WHITE = "\033[37m"
    BRIGHT_WHITE = "\033[97m"
    GRAY = "\033[90m"

    # Semantic colors for hierarchy
    MUTED = "\033[38;5;245m"      # Readable gray for labels (brighter than GRAY)
    LABEL = "\033[38;5;110m"      # Muted blue for labels
    TEXT = "\033[38;5;252m"       # Light gray for normal text
    ACCENT = "\033[38;5;216m"     # Peach/salmon for accents

    # Backgrounds
    BG_DARK = "\033[48;5;236m"    # Dark gray background
    BG_SELECTED = "\033[48;5;24m" # Dark teal for selection


def is_windows() -> bool:
    return os.name == 'nt'


def supports_color() -> bool:
    """Check if terminal supports colors."""
    if is_windows():
        # Enable ANSI on Windows 10+
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            return True
        except:
            return os.environ.get('TERM') is not None
    return sys.stdout.isatty()


def get_terminal_size() -> tuple[int, int]:
    """Get terminal size (columns, rows)."""
    try:
        size = os.get_terminal_size()
        return size.columns, size.lines
    except OSError:
        return 80, 24


# ═══════════════════════════════════════════════════════════════════════════════
# Cursor and screen control
# ═══════════════════════════════════════════════════════════════════════════════

def hide_cursor():
    sys.stdout.write('\033[?25l')
    sys.stdout.flush()


def show_cursor():
    sys.stdout.write('\033[?25h')
    sys.stdout.flush()


def move_cursor_up(n: int):
    if n > 0:
        sys.stdout.write(f'\033[{n}A')


def move_cursor_down(n: int):
    if n > 0:
        sys.stdout.write(f'\033[{n}B')


def move_cursor_to_column(col: int):
    sys.stdout.write(f'\033[{col}G')


def clear_line():
    sys.stdout.write('\033[2K')


def save_cursor():
    sys.stdout.write('\033[s')


def restore_cursor():
    sys.stdout.write('\033[u')


# ═══════════════════════════════════════════════════════════════════════════════
# Input handling
# ═══════════════════════════════════════════════════════════════════════════════

def getch() -> str:
    """Get a single character from stdin without echo."""
    if is_windows():
        import msvcrt
        ch = msvcrt.getch()
        if ch in (b'\x00', b'\xe0'):
            ch2 = msvcrt.getch()
            if ch2 == b'H': return 'UP'
            elif ch2 == b'P': return 'DOWN'
            elif ch2 == b'K': return 'LEFT'
            elif ch2 == b'M': return 'RIGHT'
            return ''
        return ch.decode('utf-8', errors='ignore')
    else:
        import termios
        import tty
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                ch2 = sys.stdin.read(1)
                if ch2 == '[':
                    ch3 = sys.stdin.read(1)
                    if ch3 == 'A': return 'UP'
                    elif ch3 == 'B': return 'DOWN'
                    elif ch3 == 'C': return 'RIGHT'
                    elif ch3 == 'D': return 'LEFT'
                return 'ESC'
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def kbhit() -> bool:
    """Check if a key has been pressed (non-blocking)."""
    if is_windows():
        import msvcrt
        return msvcrt.kbhit()
    else:
        import select
        return select.select([sys.stdin], [], [], 0)[0] != []


def check_for_cancel() -> bool:
    """Check if ESC or Ctrl+C was pressed (non-blocking). Returns True if cancel requested."""
    if not kbhit():
        return False
    key = getch()
    return key in ('ESC', '\x1b', '\x03', 'q')


# ═══════════════════════════════════════════════════════════════════════════════
# Box drawing with CDE style
# ═══════════════════════════════════════════════════════════════════════════════

class Box:
    """Box drawing characters."""
    # Single line
    H = "─"
    V = "│"
    TL = "┌"
    TR = "┐"
    BL = "└"
    BR = "┘"

    # Double line (for emphasis)
    DH = "═"
    DV = "║"
    DTL = "╔"
    DTR = "╗"
    DBL = "╚"
    DBR = "╝"


# ═══════════════════════════════════════════════════════════════════════════════
# Panel - renders inline without clearing terminal history
# ═══════════════════════════════════════════════════════════════════════════════

class Panel:
    """
    A panel that renders in place without clearing terminal history.
    Maintains cursor position and redraws content inline.
    """

    def __init__(self, height: int = 15):
        self.height = height
        self.width = min(get_terminal_size()[0] - 2, 80)
        self._lines_drawn = 0
        self._initialized = False
        self.use_color = supports_color()

    def _c(self, color: str) -> str:
        """Return color code if colors supported."""
        return color if self.use_color else ""

    def _make_space(self):
        """Print empty lines to make space for the panel."""
        if not self._initialized:
            # Print empty lines to create space
            print("\n" * (self.height - 1), end="")
            self._initialized = True
            self._lines_drawn = self.height

    def _move_to_start(self):
        """Move cursor to the start of our panel area."""
        if self._lines_drawn > 0:
            move_cursor_up(self._lines_drawn)
        move_cursor_to_column(1)

    def _write_line(self, line: str):
        """Write a line and move to next."""
        clear_line()
        sys.stdout.write(line)
        sys.stdout.write("\n")

    def render(self, lines: List[str], title: str = "", footer: str = ""):
        """
        Render the panel with given content lines.
        Stays in place, doesn't scroll terminal.
        """
        self._make_space()
        self._move_to_start()

        c = self._c
        width = self.width

        # Colors
        border_color = c(Colors.CYAN)
        title_color = c(Colors.BRIGHT_CYAN + Colors.BOLD)
        footer_color = c(Colors.WHITE)
        reset = c(Colors.RESET)

        rendered_lines = []

        # Top border with title
        if title:
            title_display = f" {title} "
            padding_left = (width - len(title_display) - 2) // 2
            padding_right = width - len(title_display) - 2 - padding_left
            top = f"{border_color}{Box.TL}{Box.H * padding_left}{reset}{title_color}{title_display}{reset}{border_color}{Box.H * padding_right}{Box.TR}{reset}"
        else:
            top = f"{border_color}{Box.TL}{Box.H * (width - 2)}{Box.TR}{reset}"
        rendered_lines.append(top)

        # Content lines
        content_height = self.height - 2  # Account for top and bottom borders
        if footer:
            content_height -= 1  # Account for footer

        for i in range(content_height):
            if i < len(lines):
                line = lines[i]
                # Strip ANSI codes for length calculation
                import re
                visible_len = len(re.sub(r'\033\[[0-9;]*m', '', line))
                padding = width - visible_len - 4  # 2 for borders, 2 for inner padding
                if padding < 0:
                    # Truncate line
                    line = line[:width - 7] + "..."
                    padding = 0
                content = f"{border_color}{Box.V}{reset}  {line}{' ' * padding}{border_color}{Box.V}{reset}"
            else:
                content = f"{border_color}{Box.V}{reset}{' ' * (width - 2)}{border_color}{Box.V}{reset}"
            rendered_lines.append(content)

        # Footer line (inside box)
        if footer:
            footer_visible_len = len(re.sub(r'\033\[[0-9;]*m', '', footer))
            footer_padding = width - footer_visible_len - 4
            footer_line = f"{border_color}{Box.V}{reset}  {footer_color}{footer}{reset}{' ' * footer_padding}{border_color}{Box.V}{reset}"
            rendered_lines.append(footer_line)

        # Bottom border
        bottom = f"{border_color}{Box.BL}{Box.H * (width - 2)}{Box.BR}{reset}"
        rendered_lines.append(bottom)

        # Write all lines
        for line in rendered_lines:
            self._write_line(line)

        self._lines_drawn = len(rendered_lines)
        sys.stdout.flush()

    def clear(self):
        """Clear the panel area."""
        if self._lines_drawn > 0:
            self._move_to_start()
            for _ in range(self._lines_drawn):
                clear_line()
                sys.stdout.write("\n")
            move_cursor_up(self._lines_drawn)
            self._lines_drawn = 0
            self._initialized = False
            sys.stdout.flush()


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
# Menu - Arrow key selection
# ═══════════════════════════════════════════════════════════════════════════════

class Menu:
    """Interactive arrow-key menu rendered in a panel."""

    def __init__(
        self,
        items: List[MenuItem],
        title: str = "",
        prompt: str = "",
        allow_cancel: bool = True,
    ):
        self.items = items
        self.title = title
        self.prompt = prompt
        self.allow_cancel = allow_cancel
        self.selected = 0
        self.scroll_offset = 0
        self.use_color = supports_color()

        # Calculate panel height based on items
        self.visible_items = min(len(items), 8)
        self.panel = Panel(height=self.visible_items + 5)  # items + prompt + spacing + footer

    def _c(self, color: str) -> str:
        return color if self.use_color else ""

    def _build_lines(self) -> List[str]:
        """Build the content lines for the panel."""
        c = self._c
        lines = []

        # Prompt
        if self.prompt:
            lines.append(f"{c(Colors.TEXT)}{self.prompt}{c(Colors.RESET)}")
            lines.append("")

        # Scroll handling
        if self.selected < self.scroll_offset:
            self.scroll_offset = self.selected
        elif self.selected >= self.scroll_offset + self.visible_items:
            self.scroll_offset = self.selected - self.visible_items + 1

        # Scroll indicator (top)
        if self.scroll_offset > 0:
            lines.append(f"{c(Colors.MUTED)}    ↑ more{c(Colors.RESET)}")

        # Calculate max label width for alignment
        max_label_width = max((len(item.label) for item in self.items), default=0)

        # Menu items
        for i in range(self.visible_items):
            idx = self.scroll_offset + i
            if idx >= len(self.items):
                break

            item = self.items[idx]
            is_selected = idx == self.selected

            # Pad label to align tags
            padding = " " * (max_label_width - len(item.label))

            # Build item display
            if is_selected:
                prefix = f"{c(Colors.BRIGHT_MAGENTA)}❯{c(Colors.RESET)} "
                label = f"{c(Colors.BRIGHT_WHITE + Colors.BOLD)}{item.label}{c(Colors.RESET)}{padding}"
            else:
                prefix = "  "
                label = f"{c(Colors.TEXT)}{item.label}{c(Colors.RESET)}{padding}"

            # Tag (installed, size, etc.)
            if item.tag:
                if "installed" in item.tag.lower():
                    tag = f"  {c(Colors.BRIGHT_GREEN)}[{item.tag}]{c(Colors.RESET)}"
                elif "recommended" in item.tag.lower():
                    tag = f"  {c(Colors.BRIGHT_YELLOW)}[{item.tag}]{c(Colors.RESET)}"
                elif "experimental" in item.tag.lower():
                    tag = f"  {c(Colors.BRIGHT_MAGENTA)}[{item.tag}]{c(Colors.RESET)}"
                elif "GB" in item.tag:
                    # Size - show in accent color (readable but not dominant)
                    tag = f"  {c(Colors.ACCENT)}[{item.tag}]{c(Colors.RESET)}"
                else:
                    tag = f"  {c(Colors.MUTED)}[{item.tag}]{c(Colors.RESET)}"
            else:
                tag = ""

            lines.append(f"{prefix}{label}{tag}")

        # Scroll indicator (bottom)
        if self.scroll_offset + self.visible_items < len(self.items):
            lines.append(f"{c(Colors.MUTED)}    ↓ more{c(Colors.RESET)}")

        # Description of selected item (highlight it since it's contextual)
        if self.items[self.selected].description:
            lines.append("")
            lines.append(f"{c(Colors.LABEL)}  {self.items[self.selected].description}{c(Colors.RESET)}")

        return lines

    def _build_footer(self) -> str:
        """Build the footer text."""
        c = self._c
        parts = [
            f"{c(Colors.CYAN)}↑↓{c(Colors.MUTED)} navigate",
            f"{c(Colors.CYAN)}Enter{c(Colors.MUTED)} select",
        ]
        if self.allow_cancel:
            parts.append(f"{c(Colors.CYAN)}Esc{c(Colors.MUTED)} cancel")
        return f"{c(Colors.RESET)}  ".join(parts) + c(Colors.RESET)

    def run(self) -> Optional[MenuItem]:
        """Run the menu and return selected item or None if cancelled."""
        if not self.items:
            return None

        hide_cursor()
        try:
            while True:
                lines = self._build_lines()
                self.panel.render(lines, title=self.title, footer=self._build_footer())

                key = getch()

                if key == 'UP' or key == 'k':
                    if self.selected > 0:
                        self.selected -= 1
                        while self.selected > 0 and self.items[self.selected].disabled:
                            self.selected -= 1

                elif key == 'DOWN' or key == 'j':
                    if self.selected < len(self.items) - 1:
                        self.selected += 1
                        while self.selected < len(self.items) - 1 and self.items[self.selected].disabled:
                            self.selected += 1

                elif key == '\r' or key == '\n' or key == ' ':
                    if not self.items[self.selected].disabled:
                        self.panel.clear()
                        return self.items[self.selected]

                elif key == 'ESC' or key == 'q':
                    if self.allow_cancel:
                        self.panel.clear()
                        return None

                elif key == '\x03':  # Ctrl+C
                    self.panel.clear()
                    raise KeyboardInterrupt()

        finally:
            show_cursor()


# ═══════════════════════════════════════════════════════════════════════════════
# Input Field
# ═══════════════════════════════════════════════════════════════════════════════

class InputField:
    """Text input field rendered in a panel."""

    def __init__(
        self,
        prompt: str = "Enter value:",
        default: str = "",
        password: bool = False,
        validator: Optional[Callable[[str], Optional[str]]] = None,
    ):
        self.prompt = prompt
        self.value = default
        self.password = password
        self.validator = validator
        self.error: Optional[str] = None
        self.use_color = supports_color()
        self.panel = Panel(height=7)

    def _c(self, color: str) -> str:
        return color if self.use_color else ""

    def _build_lines(self) -> List[str]:
        c = self._c
        lines = []

        # Prompt
        lines.append(f"{c(Colors.TEXT)}{self.prompt}{c(Colors.RESET)}")
        lines.append("")

        # Input field with cursor
        display_value = '•' * len(self.value) if self.password else self.value
        cursor = f"{c(Colors.BRIGHT_MAGENTA)}▋{c(Colors.RESET)}"
        lines.append(f"  {c(Colors.BRIGHT_WHITE)}{display_value}{cursor}")

        # Error message
        if self.error:
            lines.append("")
            lines.append(f"  {c(Colors.BRIGHT_RED)}✗ {self.error}{c(Colors.RESET)}")

        return lines

    def _build_footer(self) -> str:
        c = self._c
        return f"{c(Colors.CYAN)}Enter{c(Colors.MUTED)} submit  {c(Colors.CYAN)}Esc{c(Colors.MUTED)} cancel{c(Colors.RESET)}"

    def run(self, title: str = "") -> Optional[str]:
        """Run the input and return value or None if cancelled."""
        hide_cursor()
        try:
            while True:
                lines = self._build_lines()
                self.panel.render(lines, title=title, footer=self._build_footer())

                key = getch()

                if key == '\r' or key == '\n':
                    if self.validator:
                        self.error = self.validator(self.value)
                        if self.error:
                            continue
                    self.panel.clear()
                    return self.value

                elif key == 'ESC':
                    self.panel.clear()
                    return None

                elif key == '\x7f' or key == '\x08':  # Backspace
                    if self.value:
                        self.value = self.value[:-1]
                        self.error = None

                elif key == '\x03':  # Ctrl+C
                    self.panel.clear()
                    raise KeyboardInterrupt()

                elif len(key) == 1 and ord(key) >= 32:
                    self.value += key
                    self.error = None

        finally:
            show_cursor()


# ═══════════════════════════════════════════════════════════════════════════════
# Progress Display
# ═══════════════════════════════════════════════════════════════════════════════

class ProgressDisplay:
    """Progress bar in a panel."""

    def __init__(self, title: str = "", total: int = 100, subtitle: str = "", footer: str = ""):
        self.title = title
        self.subtitle = subtitle
        self.total = total
        self.current = 0
        self.message = ""
        self.footer = footer
        self.use_color = supports_color()
        self.panel = Panel(height=9)

    def _c(self, color: str) -> str:
        return color if self.use_color else ""

    def update(self, current: int, message: str = ""):
        """Update progress."""
        self.current = current
        self.message = message
        self._draw()

    def _draw(self):
        c = self._c
        pct = self.current / self.total if self.total > 0 else 0

        # Progress bar
        bar_width = 40
        filled = int(bar_width * pct)
        bar = f"{c(Colors.BRIGHT_MAGENTA)}{'█' * filled}{c(Colors.MUTED)}{'░' * (bar_width - filled)}{c(Colors.RESET)}"

        lines = []

        # Subtitle (e.g., URL)
        if self.subtitle:
            # Truncate long URLs
            display_url = self.subtitle
            if len(display_url) > 60:
                display_url = display_url[:57] + "..."
            lines.append(f"{c(Colors.MUTED)}{display_url}{c(Colors.RESET)}")
            lines.append("")

        # Progress message
        if self.message:
            lines.append(f"{c(Colors.TEXT)}{self.message}{c(Colors.RESET)}")
        else:
            lines.append("")

        # Progress bar
        lines.append("")
        lines.append(f"  {bar}  {c(Colors.BRIGHT_WHITE)}{pct * 100:.0f}%{c(Colors.RESET)}")

        self.panel.render(lines, title=self.title, footer=self.footer)

    def clear(self):
        self.panel.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# Status Display
# ═══════════════════════════════════════════════════════════════════════════════

class StatusDisplay:
    """Status display panel with labeled values."""

    def __init__(self, title: str = "", height: int = 12):
        self.title = title
        self.lines: List[tuple[str, str, str]] = []  # (label, value, color)
        self.footer = ""
        self.use_color = supports_color()
        self.panel = Panel(height=height)

    def _c(self, color: str) -> str:
        return color if self.use_color else ""

    def set_title(self, title: str):
        self.title = title

    def set_line(self, index: int, label: str, value: str, color: str = ""):
        """Set a status line."""
        while len(self.lines) <= index:
            self.lines.append(("", "", ""))
        self.lines[index] = (label, value, color)

    def set_footer(self, footer: str):
        self.footer = footer

    def draw(self):
        """Draw the status panel."""
        c = self._c

        # Find max label width
        max_label = max((len(l[0]) for l in self.lines if l[0]), default=0)

        content_lines = []
        for label, value, color in self.lines:
            if label:
                # Status indicators - values are important, labels are secondary
                if value.startswith("✓"):
                    value_color = c(Colors.BRIGHT_GREEN)
                elif value.startswith("✗"):
                    value_color = c(Colors.BRIGHT_RED)
                elif "..." in value or "ing" in value.lower():
                    value_color = c(Colors.YELLOW)
                else:
                    value_color = c(color) if color else c(Colors.TEXT)

                line = f"{c(Colors.LABEL)}{label.rjust(max_label)}: {c(Colors.RESET)}{value_color}{value}{c(Colors.RESET)}"
            else:
                line = ""
            content_lines.append(line)

        footer = f"{c(Colors.MUTED)}{self.footer}{c(Colors.RESET)}" if self.footer else ""
        self.panel.render(content_lines, title=self.title, footer=footer)

    def clear(self):
        self.panel.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# Simple print helpers for consistent styling
# ═══════════════════════════════════════════════════════════════════════════════

class _NoColors:
    """Dummy class for when colors aren't supported."""
    RESET = BOLD = DIM = ""
    CYAN = BRIGHT_CYAN = YELLOW = BRIGHT_YELLOW = ""
    GREEN = BRIGHT_GREEN = RED = BRIGHT_RED = ""
    MAGENTA = BRIGHT_MAGENTA = PINK = PURPLE = ""
    WHITE = BRIGHT_WHITE = GRAY = ""
    MUTED = LABEL = TEXT = ACCENT = ""
    BG_DARK = BG_SELECTED = ""

_no_colors = _NoColors()


def print_success(message: str):
    """Print a success message."""
    c = Colors if supports_color() else _no_colors
    print(f"{c.BRIGHT_GREEN}✓{c.RESET} {c.BRIGHT_WHITE}{message}{c.RESET}")


def print_error(message: str):
    """Print an error message."""
    c = Colors if supports_color() else _no_colors
    print(f"{c.BRIGHT_RED}✗{c.RESET} {c.BRIGHT_WHITE}{message}{c.RESET}")


def print_info(message: str):
    """Print an info message."""
    c = Colors if supports_color() else _no_colors
    print(f"{c.BRIGHT_CYAN}→{c.RESET} {c.BRIGHT_WHITE}{message}{c.RESET}")


def print_header(title: str):
    """Print a styled header."""
    c = Colors if supports_color() else _no_colors
    width = min(get_terminal_size()[0] - 2, 60)
    line = "─" * width
    print(f"\n{c.BRIGHT_CYAN}{line}{c.RESET}")
    print(f"{c.BRIGHT_MAGENTA}{c.BOLD}  {title}{c.RESET}")
    print(f"{c.BRIGHT_CYAN}{line}{c.RESET}\n")
