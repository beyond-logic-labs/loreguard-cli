"""LOREGUARD banner widget with ASCII art and cyan-pink gradient."""

from textual.widgets import Static
from textual.reactive import reactive
from rich.text import Text
from rich.style import Style

from ..styles import FG

# Gradient colors from CYAN to PINK (top to bottom)
GRADIENT_COLORS = [
    "#8BE9FD",  # Cyan
    "#9ADEF9",
    "#A9D3F5",
    "#B8C8F1",
    "#C7BDED",
    "#D6B2E9",
    "#E5A7E5",
    "#F49CE1",
    "#FF79C6",  # Pink
]

# ASCII art for LOREGUARD - big block letters
ASCII_LOGO = r"""
██       ██████  ██████  ███████  ██████  ██    ██  █████  ██████  ██████
██      ██    ██ ██   ██ ██      ██       ██    ██ ██   ██ ██   ██ ██   ██
██      ██    ██ ██████  █████   ██   ███ ██    ██ ███████ ██████  ██   ██
██      ██    ██ ██   ██ ██      ██    ██ ██    ██ ██   ██ ██   ██ ██   ██
███████  ██████  ██   ██ ███████  ██████   ██████  ██   ██ ██   ██ ██████
""".strip().split("\n")


def interpolate_color(color1: str, color2: str, t: float) -> str:
    """Interpolate between two hex colors. t=0 returns color1, t=1 returns color2."""
    # Parse hex colors
    r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
    r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)

    # Interpolate
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)

    return f"#{r:02x}{g:02x}{b:02x}"


def get_gradient_color(position: float) -> str:
    """Get color from gradient at position (0.0 to 1.0)."""
    if position <= 0:
        return GRADIENT_COLORS[0]
    if position >= 1:
        return GRADIENT_COLORS[-1]

    # Find the two colors to interpolate between
    segment = position * (len(GRADIENT_COLORS) - 1)
    idx = int(segment)
    t = segment - idx

    if idx >= len(GRADIENT_COLORS) - 1:
        return GRADIENT_COLORS[-1]

    return interpolate_color(GRADIENT_COLORS[idx], GRADIENT_COLORS[idx + 1], t)


class LoreguardBanner(Static):
    """Banner widget displaying LOREGUARD ASCII art with cyan-pink gradient."""

    def __init__(self) -> None:
        super().__init__()

    def render(self) -> Text:
        """Render the LOREGUARD banner with gradient."""
        text = Text()

        # Calculate the width of the ASCII art
        max_width = max(len(line) for line in ASCII_LOGO)

        # Top border with gradient
        border_char = "#"
        border_width = max_width + 4  # padding on sides

        # Top border line
        for i, char in enumerate(border_char * border_width):
            color = get_gradient_color(i / border_width)
            text.append(char, style=Style(color=color))
        text.append("\n")

        # Empty line with border
        text.append(border_char, style=Style(color=GRADIENT_COLORS[0]))
        text.append(" " * (border_width - 2))
        text.append(border_char + "\n", style=Style(color=GRADIENT_COLORS[-1]))

        # ASCII art lines with vertical gradient
        total_lines = len(ASCII_LOGO)
        for line_idx, line in enumerate(ASCII_LOGO):
            # Left border
            v_progress = line_idx / max(1, total_lines - 1)
            left_color = get_gradient_color(v_progress * 0.5)  # First half of gradient
            text.append(border_char + " ", style=Style(color=left_color))

            # The ASCII art line with horizontal gradient
            padded_line = line.ljust(max_width)
            for char_idx, char in enumerate(padded_line):
                # Horizontal gradient across the line
                h_progress = char_idx / max(1, len(padded_line) - 1)
                color = get_gradient_color(h_progress)
                text.append(char, style=Style(color=color, bold=True))

            # Right border
            right_color = get_gradient_color(0.5 + v_progress * 0.5)  # Second half of gradient
            text.append(" " + border_char + "\n", style=Style(color=right_color))

        # Empty line with border
        text.append(border_char, style=Style(color=GRADIENT_COLORS[4]))
        text.append(" " * (border_width - 2))
        text.append(border_char + "\n", style=Style(color=GRADIENT_COLORS[4]))

        # Subtitle line (white text)
        subtitle = "Local inference for your game NPCs"
        sub_padding = (border_width - 2 - len(subtitle)) // 2
        text.append(border_char, style=Style(color=GRADIENT_COLORS[5]))
        text.append(" " * sub_padding)
        text.append(subtitle, style=Style(color=FG))  # White text
        remaining = border_width - 2 - sub_padding - len(subtitle)
        text.append(" " * remaining)
        text.append(border_char + "\n", style=Style(color=GRADIENT_COLORS[5]))

        # Empty line with border
        text.append(border_char, style=Style(color=GRADIENT_COLORS[6]))
        text.append(" " * (border_width - 2))
        text.append(border_char + "\n", style=Style(color=GRADIENT_COLORS[6]))

        # Bottom border line with gradient
        for i, char in enumerate(border_char * border_width):
            color = get_gradient_color(i / border_width)
            text.append(char, style=Style(color=color))

        return text
