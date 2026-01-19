"""LOREGUARD banner widget - minimal style inspired by Crush."""

from textual.widgets import Static
from rich.text import Text
from rich.style import Style

from ..styles import FG_DIM, PINK

# Simple stylized logo
LOGO = r"""
 _    ___  ___ ___ ___ _  _   _   ___ ___
| |  / _ \| _ \ __/ __| || | /_\ | _ \   \
| |_| (_) |   / _| (_ | || |/ _ \|   / |) |
|____\___/|_|_\___\___|___|/_/ \_\_|_\___/
""".strip().split("\n")


def get_gradient_color(position: float) -> str:
    """Get grayscale gradient with hint of pink at the end."""
    # Grayscale from light to slightly pink
    if position < 0.7:
        # Gray gradient
        gray = int(100 + (180 - 100) * (1 - position / 0.7))
        return f"#{gray:02x}{gray:02x}{gray:02x}"
    else:
        # Transition to muted pink
        t = (position - 0.7) / 0.3
        gray = int(100 * (1 - t))
        r = int(100 + (200 - 100) * t)
        g = int(100 * (1 - t * 0.5))
        b = int(100 + (150 - 100) * t)
        return f"#{r:02x}{g:02x}{b:02x}"


class LoreguardBanner(Static):
    """Compact banner widget - minimal gray style."""

    DEFAULT_CSS = """
    LoreguardBanner {
        height: 6;
        width: 100%;
        padding: 0 1;
    }
    """

    def __init__(self, version: str = "0.11.0") -> None:
        super().__init__()
        self._version = version

    def render(self) -> Text:
        """Render minimal banner."""
        text = Text()

        # Line 1: Branding + version
        text.append("Beyond Logic Labs™", style=f"bold {PINK}")
        text.append(f"  v{self._version}", style=FG_DIM)
        text.append("\n")

        # Logo lines with subtle gradient
        for line_idx, line in enumerate(LOGO):
            for char_idx, char in enumerate(line):
                h_progress = char_idx / max(1, len(line) - 1)
                color = get_gradient_color(h_progress)
                text.append(char, style=Style(color=color))

            if line_idx < len(LOGO) - 1:
                text.append("\n")

        return text
