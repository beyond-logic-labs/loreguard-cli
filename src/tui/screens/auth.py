"""Authentication screen - Step 1/4."""

import socket
from typing import TYPE_CHECKING

import httpx
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Static, ListView, ListItem, Label
from textual.worker import Worker, get_current_worker
from rich.text import Text

from ..widgets.banner import LoreguardBanner
from ..widgets.hardware_info import HardwareInfo
from ..styles import CYAN, PINK, GREEN, RED, FG_DIM

if TYPE_CHECKING:
    from ..app import LoreguardApp


class AuthScreen(Screen):
    """Authentication screen with menu and token input."""

    BINDINGS = [
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("enter", "select", "Select", show=True),
    ]

    def compose(self) -> ComposeResult:
        """Compose the auth screen layout."""
        yield LoreguardBanner()
        yield HardwareInfo()
        yield Static("", classes="spacer")
        yield Static("Step 1/4: Authentication", classes="screen-title")
        yield Static("Choose how to connect", classes="screen-subtitle")
        yield Center(
            Vertical(
                ListView(
                    ListItem(Label("Paste token"), id="opt-token"),
                    ListItem(Label("Dev mode"), id="opt-dev"),
                    id="auth-menu",
                ),
                id="auth-container",
            )
        )
        yield Static("", id="status-message")
        yield Footer()

    def on_mount(self) -> None:
        """Focus the menu on mount."""
        menu = self.query_one("#auth-menu", ListView)
        menu.focus()

    def action_cursor_up(self) -> None:
        """Move cursor up in menu."""
        menu = self.query_one("#auth-menu", ListView)
        menu.action_cursor_up()

    def action_cursor_down(self) -> None:
        """Move cursor down in menu."""
        menu = self.query_one("#auth-menu", ListView)
        menu.action_cursor_down()

    def action_select(self) -> None:
        """Handle selection."""
        menu = self.query_one("#auth-menu", ListView)
        selected = menu.highlighted_child
        if selected:
            if selected.id == "opt-dev":
                self._enter_dev_mode()
            elif selected.id == "opt-token":
                self._show_token_input()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle list item selection."""
        if event.item.id == "opt-dev":
            self._enter_dev_mode()
        elif event.item.id == "opt-token":
            self._show_token_input()

    def _enter_dev_mode(self) -> None:
        """Enter dev mode without authentication."""
        app: "LoreguardApp" = self.app  # type: ignore
        status = self.query_one("#status-message", Static)
        status.update(Text("Dev mode enabled", style=f"bold {GREEN}"))
        app.proceed_to_model_select("dev_mock_token", "dev-worker")

    def _show_token_input(self) -> None:
        """Show the token input modal."""
        from ..modals.token_input import TokenInputModal

        def handle_token(token: str | None) -> None:
            if token:
                self._validate_token(token)

        self.app.push_screen(TokenInputModal(), handle_token)

    def _validate_token(self, token: str) -> None:
        """Validate the token with the backend."""
        status = self.query_one("#status-message", Static)
        status.update(Text("Validating token...", style=f"{CYAN}"))
        self.run_worker(self._do_validate_token(token), exclusive=True)

    async def _do_validate_token(self, token: str) -> None:
        """Worker to validate token asynchronously."""
        status = self.query_one("#status-message", Static)
        app: "LoreguardApp" = self.app  # type: ignore

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
                    app.proceed_to_model_select(token, hostname)
                else:
                    status.update(Text("Authentication failed - invalid token", style=f"bold {RED}"))
        except Exception as e:
            status.update(Text(f"Connection error: {e}", style=f"bold {RED}"))
