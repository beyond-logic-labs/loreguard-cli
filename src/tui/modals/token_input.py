"""Token input modal for authentication."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from ..styles import PURPLE, FG_DIM


class TokenInputModal(ModalScreen[str | None]):
    """Modal for entering API token."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("enter", "submit", "Submit", show=False),
    ]

    DEFAULT_CSS = f"""
    TokenInputModal {{
        align: center middle;
        background: transparent;
    }}

    TokenInputModal > Vertical {{
        width: 60;
        height: auto;
        border: thick $background 80%;
        background: $surface;
        padding: 1 2;
    }}

    TokenInputModal .modal-title {{
        text-align: center;
        text-style: bold;
        color: {PURPLE};
        padding-bottom: 1;
    }}

    TokenInputModal .modal-hint {{
        text-align: center;
        color: {FG_DIM};
        padding-bottom: 1;
    }}

    TokenInputModal .modal-footer {{
        text-align: center;
        color: {FG_DIM};
        padding-top: 1;
    }}

    TokenInputModal Input {{
        width: 100%;
    }}
    """

    def compose(self) -> ComposeResult:
        """Compose the modal layout."""
        with Vertical():
            yield Static("Enter API Token", classes="modal-title")
            yield Static("Get your token at loreguard.com/dashboard", classes="modal-hint")
            yield Input(placeholder="Paste your token here...", password=True, id="token-input")
            yield Static("enter submit • esc cancel", classes="modal-footer")

    def on_mount(self) -> None:
        """Focus the input on mount."""
        self.query_one("#token-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        token = event.value.strip()
        if token:
            self.dismiss(token)
        else:
            # Flash the input to indicate error
            input_widget = self.query_one("#token-input", Input)
            input_widget.add_class("error")
            self.set_timer(0.5, lambda: input_widget.remove_class("error"))

    def action_cancel(self) -> None:
        """Cancel and close the modal."""
        self.dismiss(None)

    def action_submit(self) -> None:
        """Submit the token."""
        input_widget = self.query_one("#token-input", Input)
        token = input_widget.value.strip()
        if token:
            self.dismiss(token)
