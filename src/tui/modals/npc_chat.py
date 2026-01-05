"""NPC Chat modal - test chat with local LLM."""

from typing import TYPE_CHECKING

import httpx
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, Static, Label
from rich.text import Text

from ..styles import PURPLE, CYAN, PINK, FG, FG_DIM, GREEN
from ..widgets.banner import get_gradient_color

if TYPE_CHECKING:
    from ..app import LoreguardApp


class NPCChatModal(ModalScreen[None]):
    """Modal for chatting with a test NPC via local LLM."""

    BINDINGS = [
        Binding("escape", "close", "Close", show=False),
        Binding("enter", "send", "Send", show=False),
    ]

    DEFAULT_CSS = f"""
    NPCChatModal {{
        align: center middle;
        background: transparent;
    }}

    NPCChatModal > Vertical {{
        width: 70;
        height: auto;
        max-height: 18;
        border: thick $background 80%;
        background: $surface;
        padding: 1;
    }}

    NPCChatModal .modal-header {{
        height: 1;
        padding: 0 1;
    }}

    NPCChatModal .chat-container {{
        height: 1fr;
        padding: 0 1;
        background: #1E1F29;
    }}

    NPCChatModal .chat-message {{
        padding: 0;
        margin: 0 0 1 0;
    }}

    NPCChatModal Input {{
        width: 100%;
        border: none;
        background: transparent;
        padding: 0 1;
    }}

    NPCChatModal .modal-footer {{
        height: 1;
        padding: 0 1;
        color: {FG_DIM};
    }}

    NPCChatModal .status-line {{
        height: 1;
        padding: 0 1;
    }}
    """

    # Default NPC persona
    DEFAULT_SYSTEM_PROMPT = """You are Garrett, a grizzled blacksmith in a medieval fantasy village. You speak in a gruff but friendly manner, often mentioning your work at the forge. You have knowledge of weapons, armor, and local village gossip. Keep responses concise (2-3 sentences)."""

    def __init__(self, npc_name: str = "Garrett", system_prompt: str | None = None) -> None:
        super().__init__()
        self._npc_name = npc_name
        self._system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self._messages: list[dict] = []
        self._generating = False

    def compose(self) -> ComposeResult:
        """Compose the chat modal."""
        with Vertical():
            yield Static(self._render_header(), classes="modal-header")
            yield VerticalScroll(id="chat-container", classes="chat-container")
            yield Static("", id="status-line", classes="status-line")
            yield Input(placeholder=f"> Say something to {self._npc_name}...", id="chat-input")
            yield Static(self._render_footer(), classes="modal-footer")

    def _render_header(self) -> Text:
        """Render header with NPC name and gradient line."""
        text = Text()
        text.append(f"Chat with {self._npc_name} ", style=f"bold {PINK}")
        # Add gradient hashes
        remaining = 68 - len(self._npc_name) - 11
        for i in range(remaining):
            color = get_gradient_color(i / remaining)
            text.append("#", style=color)
        return text

    def _render_footer(self) -> Text:
        """Render footer with key hints."""
        text = Text()
        text.append("enter", style=f"bold {FG}")
        text.append(" send ", style=FG_DIM)
        text.append("  ")
        text.append("esc", style=f"bold {FG}")
        text.append(" close", style=FG_DIM)
        return text

    def on_mount(self) -> None:
        """Focus input and show welcome message."""
        self.query_one("#chat-input", Input).focus()
        self._add_npc_message("*looks up from the forge* Ah, a visitor! What brings ye to my smithy today?")

    def _add_message(self, role: str, content: str) -> None:
        """Add a message to the chat display."""
        container = self.query_one("#chat-container", VerticalScroll)

        text = Text()
        if role == "user":
            text.append("You: ", style=f"bold {CYAN}")
            text.append(content, style=FG)
        else:
            text.append(f"{self._npc_name}: ", style=f"bold {PINK}")
            text.append(content, style=FG)

        label = Label(text, classes="chat-message")
        container.mount(label)
        container.scroll_end(animate=False)

    def _add_npc_message(self, content: str) -> None:
        """Add an NPC message."""
        self._messages.append({"role": "assistant", "content": content})
        self._add_message("npc", content)

    def _add_user_message(self, content: str) -> None:
        """Add a user message."""
        self._messages.append({"role": "user", "content": content})
        self._add_message("user", content)

    def action_send(self) -> None:
        """Send the message."""
        if self._generating:
            return

        input_widget = self.query_one("#chat-input", Input)
        message = input_widget.value.strip()

        if not message:
            return

        input_widget.value = ""
        self._add_user_message(message)
        self._generate_response(message)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        self.action_send()

    def _generate_response(self, user_message: str) -> None:
        """Generate NPC response using local LLM."""
        self._generating = True
        status = self.query_one("#status-line", Static)
        status.update(Text(f"{self._npc_name} is thinking...", style=CYAN))

        self.run_worker(self._do_generate(user_message))

    async def _do_generate(self, user_message: str) -> None:
        """Worker to generate response."""
        status = self.query_one("#status-line", Static)

        try:
            # Build messages for the LLM
            messages = [{"role": "system", "content": self._system_prompt}]
            messages.extend(self._messages)

            # Call local llama-server
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "http://127.0.0.1:8080/v1/chat/completions",
                    json={
                        "messages": messages,
                        "max_tokens": 256,
                        "temperature": 0.8,
                        "stream": False,
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if content:
                        self._add_npc_message(content.strip())
                        status.update("")
                    else:
                        status.update(Text("No response generated", style=FG_DIM))
                else:
                    status.update(Text(f"LLM error: {response.status_code}", style="#FF5555"))

        except httpx.ConnectError:
            status.update(Text("Cannot connect to llama-server", style="#FF5555"))
        except Exception as e:
            status.update(Text(f"Error: {e}", style="#FF5555"))
        finally:
            self._generating = False

    def action_close(self) -> None:
        """Close the chat modal."""
        self.dismiss(None)
