"""Auth menu modal - Crush-style floating centered dialog."""

import socket
from typing import TYPE_CHECKING

import httpx
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static, ListView, ListItem, Label
from rich.text import Text

from ..styles import PURPLE, CYAN, PINK, FG, FG_DIM, GREEN, RED
from ..widgets.banner import get_gradient_color

if TYPE_CHECKING:
    from ..app import LoreguardApp


def gradient_line(width: int) -> Text:
    """Create a gradient line of slashes."""
    text = Text()
    for i in range(width):
        color = get_gradient_color(i / width)
        text.append("/", style=color)
    return text


class AuthMenuModal(ModalScreen[tuple | None]):
    """Floating auth menu modal like Crush's command palette."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("enter", "select", "Select", show=False),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("j", "cursor_down", "Down", show=False),
    ]

    DEFAULT_CSS = f"""
    AuthMenuModal {{
        align: center middle;
        background: transparent;
    }}

    AuthMenuModal > Vertical {{
        width: 60;
        height: auto;
        border: thick $background 80%;
        background: $surface;
        padding: 1;
    }}

    AuthMenuModal .modal-header {{
        height: 1;
        padding: 0 1;
        background: transparent;
    }}

    AuthMenuModal .modal-title {{
        padding: 0 1;
    }}

    AuthMenuModal Input {{
        width: 100%;
        border: none;
        background: transparent;
        padding: 0 1;
        margin-bottom: 1;
    }}

    AuthMenuModal ListView {{
        height: auto;
        max-height: 10;
        padding: 0;
        background: transparent;
    }}

    AuthMenuModal ListView > ListItem {{
        padding: 0 1;
        height: 1;
    }}

    AuthMenuModal ListView > ListItem.-selected {{
        background: {PURPLE};
        color: {FG};
        text-style: bold;
    }}

    AuthMenuModal .modal-footer {{
        height: 1;
        padding: 0 1;
        color: {FG_DIM};
        background: transparent;
    }}

    AuthMenuModal .status-line {{
        padding: 0 1;
        height: 1;
    }}
    """

    def __init__(self) -> None:
        super().__init__()
        self._mode = "menu"  # "menu" or "token"
        self._validating = False

    def compose(self) -> ComposeResult:
        """Compose the modal."""
        with Vertical():
            yield Static(self._render_header(), classes="modal-header")
            yield Input(placeholder="> Type to filter", id="filter-input")
            yield ListView(
                ListItem(Label(self._render_option("Paste Token", "Enter your API token")), id="opt-token"),
                ListItem(Label(self._render_option("Dev Mode", "Test locally without backend")), id="opt-dev"),
                id="auth-list",
            )
            yield Static("", id="status-line", classes="status-line")
            yield Static(self._render_footer(), classes="modal-footer")

    def _render_header(self) -> Text:
        """Render header with title and gradient line."""
        text = Text()
        text.append("Step 1/4 ", style=f"bold {PINK}")
        text.append("Auth ", style=f"bold {FG}")
        # Add gradient hashes
        for i in range(40):
            color = get_gradient_color(i / 40)
            text.append("#", style=color)
        return text

    def _render_option(self, title: str, desc: str) -> Text:
        """Render a menu option."""
        text = Text()
        text.append(title, style=f"bold {FG}")
        return text

    def _render_footer(self) -> Text:
        """Render footer with key hints."""
        text = Text()
        text.append("tab", style=f"bold {FG}")
        text.append(" switch selection ", style=FG_DIM)
        text.append("  ")
        text.append("↑↓", style=f"bold {FG}")
        text.append(" choose ", style=FG_DIM)
        text.append("  ")
        text.append("enter", style=f"bold {FG}")
        text.append(" confirm ", style=FG_DIM)
        text.append("  ")
        text.append("esc", style=f"bold {FG}")
        text.append(" cancel", style=FG_DIM)
        return text

    def on_mount(self) -> None:
        """Focus the list on mount."""
        self.query_one("#auth-list", ListView).focus()

    def action_cursor_up(self) -> None:
        """Move cursor up."""
        if self._mode == "menu":
            self.query_one("#auth-list", ListView).action_cursor_up()

    def action_cursor_down(self) -> None:
        """Move cursor down."""
        if self._mode == "menu":
            self.query_one("#auth-list", ListView).action_cursor_down()

    def action_select(self) -> None:
        """Handle selection."""
        if self._mode == "token":
            # Submit token
            token_input = self.query_one("#filter-input", Input)
            token = token_input.value.strip()
            if token:
                self._validate_token(token)
            return

        if self._mode == "menu":
            auth_list = self.query_one("#auth-list", ListView)
            selected = auth_list.highlighted_child
            if selected:
                if selected.id == "opt-dev":
                    self.dismiss(("dev_mock_token", "dev-worker", True))
                elif selected.id == "opt-token":
                    self._switch_to_token_input()

    def action_cancel(self) -> None:
        """Cancel."""
        if self._mode == "token":
            self._switch_to_menu()
        else:
            self.dismiss(None)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle list selection."""
        if self._mode == "menu":
            if event.item.id == "opt-dev":
                self.dismiss(("dev_mock_token", "dev-worker", True))
            elif event.item.id == "opt-token":
                self._switch_to_token_input()

    def _switch_to_token_input(self) -> None:
        """Switch to token input mode."""
        self._mode = "token"

        # Update header
        header = self.query_one(".modal-header", Static)
        text = Text()
        text.append("Enter Token ", style=f"bold {PINK}")
        for i in range(38):
            color = get_gradient_color(i / 38)
            text.append("#", style=color)
        header.update(text)

        # Update input
        input_widget = self.query_one("#filter-input", Input)
        input_widget.placeholder = "Paste your API token here..."
        input_widget.password = True
        input_widget.value = ""
        input_widget.focus()

        # Hide list
        auth_list = self.query_one("#auth-list", ListView)
        auth_list.display = False

        # Update status
        status = self.query_one("#status-line", Static)
        status.update(Text("Get your token at loreguard.com/dashboard", style=FG_DIM))

    def _switch_to_menu(self) -> None:
        """Switch back to menu mode."""
        self._mode = "menu"

        # Update header
        header = self.query_one(".modal-header", Static)
        header.update(self._render_header())

        # Update input
        input_widget = self.query_one("#filter-input", Input)
        input_widget.placeholder = "> Type to filter"
        input_widget.password = False
        input_widget.value = ""

        # Show list
        auth_list = self.query_one("#auth-list", ListView)
        auth_list.display = True
        auth_list.focus()

        # Clear status
        status = self.query_one("#status-line", Static)
        status.update("")

    def _validate_token(self, token: str) -> None:
        """Validate the token."""
        if self._validating:
            return
        self._validating = True

        status = self.query_one("#status-line", Static)
        status.update(Text("Validating token...", style=CYAN))

        self.run_worker(self._do_validate_token(token))

    async def _do_validate_token(self, token: str) -> None:
        """Worker to validate token."""
        status = self.query_one("#status-line", Static)

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
                    status.update(Text(f"Authenticated as {name}", style=f"bold {GREEN}"))
                    import asyncio
                    await asyncio.sleep(0.5)
                    self.dismiss((token, hostname, False))
                else:
                    status.update(Text("Invalid token", style=f"bold {RED}"))
                    self._validating = False
        except Exception as e:
            status.update(Text(f"Connection error: {e}", style=f"bold {RED}"))
            self._validating = False

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        if self._mode == "token":
            token = event.value.strip()
            if token:
                self._validate_token(token)
