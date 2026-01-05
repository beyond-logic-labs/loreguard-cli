"""NPC Chat widget - embedded chat panel for main screen.

Uses the Loreguard API for NPC conversations:
- POST /api/chat - Chat with an NPC (grounded responses)
"""

from typing import TYPE_CHECKING

import httpx
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static, Input, Label
from textual.binding import Binding
from rich.text import Text

from ..styles import PURPLE, CYAN, PINK, FG, FG_DIM, GREEN
from .banner import get_gradient_color

if TYPE_CHECKING:
    from ..app import LoreguardApp

# Loreguard API
LOREGUARD_API_URL = "https://api.loreguard.com"


class NPCChat(Vertical):
    """Embedded NPC chat widget for the main screen.

    Uses the Loreguard API for grounded NPC conversations.
    """

    BINDINGS = [
        Binding("escape", "close_chat", "Close", show=False),
    ]

    DEFAULT_CSS = f"""
    NPCChat {{
        width: 100%;
        height: 1fr;
        border: solid {FG_DIM};
        background: transparent;
        display: none;
    }}

    NPCChat.visible {{
        display: block;
    }}

    NPCChat .chat-header {{
        height: 1;
        padding: 0 1;
    }}

    NPCChat .chat-container {{
        height: 1fr;
        padding: 0 1;
        background: transparent;
    }}

    NPCChat .chat-message {{
        padding: 0;
        margin: 0;
    }}

    NPCChat Input {{
        width: 100%;
        border: none;
        background: transparent;
        padding: 0;
    }}

    NPCChat .chat-footer {{
        height: 1;
        padding: 0 1;
        color: {FG_DIM};
    }}

    NPCChat .status-line {{
        height: 1;
        padding: 0 1;
    }}
    """

    def __init__(self) -> None:
        super().__init__()
        self._npc_id: str = ""
        self._npc_name: str = "NPC"
        self._api_token: str = ""
        self._verbose: bool = False
        self._messages: list[dict] = []
        self._generating = False
        self._visible = False

    def compose(self) -> ComposeResult:
        """Compose the chat widget."""
        yield Static(self._render_header(), classes="chat-header")
        yield VerticalScroll(id="npc-chat-container", classes="chat-container")
        yield Static("", id="npc-status-line", classes="status-line")
        yield Input(placeholder=f"> Say something to {self._npc_name}...", id="npc-chat-input")
        yield Static(self._render_footer(), classes="chat-footer")

    def _render_header(self) -> Text:
        """Render header with NPC name and gradient line."""
        text = Text()
        text.append(f"Chat with {self._npc_name} ", style=f"bold {PINK}")
        remaining = 60 - len(self._npc_name) - 11
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

    def show(self) -> None:
        """Show the chat widget."""
        self._visible = True
        self.add_class("visible")
        # Add class to parent screen to adjust layout
        if self.screen:
            self.screen.add_class("chat-open")
        # Focus the input - no welcome message for API-based NPCs
        # The NPC's personality is defined in Loreguard, not here
        self.query_one("#npc-chat-input", Input).focus()

    def hide(self) -> None:
        """Hide the chat widget."""
        self._visible = False
        self.remove_class("visible")
        # Remove class from parent screen
        if self.screen:
            self.screen.remove_class("chat-open")

    def toggle(self) -> None:
        """Toggle chat visibility."""
        if self._visible:
            self.hide()
        else:
            self.show()

    def set_npc(self, npc_id: str, name: str, api_token: str, verbose: bool = False) -> None:
        """Set the NPC and reset chat.

        Args:
            npc_id: Character ID from Loreguard API
            name: Display name of the NPC
            api_token: API token for authentication
            verbose: Enable verbose mode (show pipeline debug info)
        """
        self._npc_id = npc_id
        self._npc_name = name
        self._api_token = api_token
        self._verbose = verbose
        self._messages = []

        # Update header
        header = self.query_one(".chat-header", Static)
        header.update(self._render_header())

        # Update input placeholder
        input_widget = self.query_one("#npc-chat-input", Input)
        input_widget.placeholder = f"> Say something to {name}..."

        # Clear chat container
        container = self.query_one("#npc-chat-container", VerticalScroll)
        container.remove_children()

        # Clear status
        status = self.query_one("#npc-status-line", Static)
        status.update("")

    @property
    def is_visible(self) -> bool:
        """Check if chat is visible."""
        return self._visible

    def _add_message(self, role: str, content: str) -> None:
        """Add a message to the chat display."""
        container = self.query_one("#npc-chat-container", VerticalScroll)

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

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        if event.input.id != "npc-chat-input":
            return

        if self._generating:
            return

        message = event.value.strip()
        if not message:
            return

        event.input.value = ""
        self._add_user_message(message)
        self._generate_response(message)

    def _generate_response(self, user_message: str) -> None:
        """Generate NPC response using local LLM."""
        self._generating = True
        status = self.query_one("#npc-status-line", Static)
        status.update(Text(f"{self._npc_name} is thinking...", style=CYAN))

        self.run_worker(self._do_generate(user_message))

    async def _do_generate(self, user_message: str) -> None:
        """Worker to generate response using Loreguard API."""
        status = self.query_one("#npc-status-line", Static)

        if not self._npc_id or not self._api_token:
            status.update(Text("Not connected to an NPC", style="#FF5555"))
            self._generating = False
            return

        try:
            # Build history for API
            history = []
            for msg in self._messages:
                history.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

            # Build request payload
            payload = {
                "character_id": self._npc_id,
                "message": user_message,
                "history": history,
                "player_id": "tui_user",
            }

            # Add verbose flag to request pipeline debug info
            if self._verbose:
                payload["verbose"] = True

            # Call Loreguard API
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{LOREGUARD_API_URL}/api/chat",
                    headers={
                        "Authorization": f"Bearer {self._api_token}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )

                if response.status_code == 200:
                    data = response.json()
                    speech = data.get("speech", "")
                    verified = data.get("verified", False)

                    if speech:
                        # Show debug info if verbose mode
                        if self._verbose:
                            self._show_debug_info(data)

                        # Add verification indicator
                        indicator = "●" if verified else "○"
                        self._add_npc_message(f"{speech} {indicator}")
                        status.update("")
                    else:
                        status.update(Text("No response generated", style=FG_DIM))
                elif response.status_code == 401:
                    status.update(Text("Authentication failed - invalid token", style="#FF5555"))
                elif response.status_code == 429:
                    try:
                        data = response.json()
                        limit = data.get("requests_limit", "?")
                        used = data.get("requests_used", "?")
                        status.update(Text(f"Rate limited: {used}/{limit} requests", style="#FF5555"))
                    except Exception:
                        status.update(Text("Rate limited - please wait", style="#FF5555"))
                elif response.status_code == 404:
                    status.update(Text(f"NPC '{self._npc_name}' not found", style="#FF5555"))
                else:
                    # Try to get full error details from response
                    try:
                        err_text = response.text
                        # Try to parse as JSON for cleaner display
                        try:
                            err_data = response.json()
                            err_msg = str(err_data)
                        except Exception:
                            err_msg = err_text[:500] if err_text else "No response body"

                        # Show in debug area if verbose, otherwise in status
                        if self._verbose:
                            self._show_api_error(response.status_code, err_msg, dict(response.headers))
                        status.update(Text(f"API error {response.status_code}: {err_msg[:100]}", style="#FF5555"))
                    except Exception as e:
                        status.update(Text(f"API error {response.status_code}: {e}", style="#FF5555"))

        except httpx.ConnectError:
            status.update(Text("Cannot connect to Loreguard API", style="#FF5555"))
        except httpx.TimeoutException:
            status.update(Text("Request timed out - try again", style="#FF5555"))
        except Exception as e:
            status.update(Text(f"Error: {e}", style="#FF5555"))
        finally:
            self._generating = False

    def on_pass_update(self, payload: dict) -> None:
        """Handle pass_update message from backend (verbose mode).

        Called by the tunnel when it receives pass updates via WebSocket.
        """
        if not self._verbose or not self._visible:
            return

        container = self.query_one("#npc-chat-container", VerticalScroll)

        pass_num = payload.get("pass", "?")
        name = payload.get("name", "Unknown")
        duration = payload.get("durationMs", 0)
        skipped = payload.get("skipped", False)
        error = payload.get("error", False)
        retry_of = payload.get("retryOf", 0)

        text = Text()
        retry_suffix = f" (Retry {retry_of})" if retry_of > 0 else ""

        if skipped:
            text.append(f"[Pass {pass_num}] {name}: Skipped", style=FG_DIM)
        elif error:
            text.append(f"[Pass {pass_num}] {name}: Error{retry_suffix}", style="#FF5555")
            if error_msg := payload.get("errorMsg"):
                text.append(f"\n  {error_msg}", style="#FF5555")
        else:
            text.append(f"[Pass {pass_num}] {name} ({duration}ms){retry_suffix}", style=CYAN)

            # Query rewrite (Pass 1)
            if query_rewrite := payload.get("queryRewrite"):
                text.append(f"\n  Query rewrite: {query_rewrite}", style=FG_DIM)

            # Sources (Pass 1)
            if sources := payload.get("sources"):
                text.append(f"\n  Sources ({len(sources)}):", style=FG_DIM)
                for src in sources[:5]:  # Show first 5
                    score = src.get("score", 0)
                    path = src.get("path", "")
                    src_id = src.get("id", "?")
                    text.append(f"\n    [{src_id}] {path} (score: {score:.2f})", style=FG_DIM)

            # Evidence blocks (Pass 4)
            if evidence_blocks := payload.get("evidenceBlocks"):
                text.append(f"\n  Evidence ({len(evidence_blocks)}):", style=FG_DIM)
                for block in evidence_blocks[:3]:  # Show first 3
                    block_id = block.get("id", "?")
                    block_text = block.get("text", "")[:80]
                    text.append(f"\n    [{block_id}] \"{block_text}...\"", style=FG_DIM)

            # Verdict (Pass 2.5/4.5)
            if verdict := payload.get("verdict"):
                if verdict == "APPROVED":
                    faith = payload.get("faithfulness", 0)
                    text.append(f"\n  ✓ APPROVED (faithfulness: {faith:.2f})", style=GREEN)
                else:
                    issues = payload.get("issues", [])
                    text.append(f"\n  ✗ ISSUES_FOUND ({len(issues)} issue(s))", style="#FF5555")
                    for issue in issues[:3]:
                        claim = issue.get("claim", "")[:40]
                        text.append(f"\n    - \"{claim}...\"", style=FG_DIM)

            # Output (Pass 2/4 - internal monologue or speech)
            if output := payload.get("output"):
                output_preview = output[:200] + "..." if len(output) > 200 else output
                text.append(f"\n  Output: {output_preview}", style=FG_DIM)

        label = Label(text, classes="chat-message")
        container.mount(label)
        container.scroll_end(animate=False)

    def _show_api_error(self, status_code: int, body: str, headers: dict) -> None:
        """Show full API error details in verbose mode."""
        container = self.query_one("#npc-chat-container", VerticalScroll)

        text = Text()
        text.append(f"─── API ERROR {status_code} ───\n", style="#FF5555")
        text.append(f"Body: {body}\n", style=FG_DIM)
        text.append("Headers:\n", style=FG_DIM)
        for key, value in headers.items():
            text.append(f"  {key}: {value}\n", style=FG_DIM)
        text.append("─────────────────────", style="#FF5555")

        label = Label(text, classes="chat-message")
        container.mount(label)
        container.scroll_end(animate=False)

    def _show_debug_info(self, data: dict) -> None:
        """Show debug info from API response in verbose mode."""
        container = self.query_one("#npc-chat-container", VerticalScroll)

        # Build debug text
        debug_lines = []

        latency = data.get("latency_ms")
        if latency:
            debug_lines.append(f"latency: {latency}ms")

        verified = data.get("verified")
        if verified is not None:
            debug_lines.append(f"verified: {verified}")

        retries = data.get("retries")
        if retries:
            debug_lines.append(f"retries: {retries}")

        thoughts = data.get("thoughts")
        if thoughts:
            debug_lines.append(f"thoughts: {thoughts}")

        citations = data.get("citations", [])
        if citations:
            debug_lines.append(f"citations: {len(citations)}")
            for cit in citations[:3]:  # Show first 3
                claim = cit.get("claim", "")[:50]
                verified_cit = "✓" if cit.get("verified") else "✗"
                debug_lines.append(f"  {verified_cit} {claim}...")

        if debug_lines:
            text = Text()
            text.append("─── debug ───\n", style=FG_DIM)
            for line in debug_lines:
                text.append(f"{line}\n", style=FG_DIM)
            text.append("─────────────", style=FG_DIM)

            label = Label(text, classes="chat-message")
            container.mount(label)
            container.scroll_end(animate=False)

    def action_close_chat(self) -> None:
        """Close the chat widget."""
        self.hide()
