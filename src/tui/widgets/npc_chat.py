"""NPC Chat widget - embedded chat panel for main screen.

Uses the local proxy for NPC conversations with token streaming:
- POST /api/chat - Chat with an NPC (grounded responses via SSE streaming)
"""

import json
from typing import TYPE_CHECKING

import httpx
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, VerticalScroll
from textual.widgets import Static, Input
from textual.binding import Binding
from rich.text import Text

from ..styles import CYAN, FG, FG_DIM, GREEN
from ...runtime import RuntimeInfo

if TYPE_CHECKING:
    from ..app import LoreguardApp

# Fallback to cloud API if local proxy unavailable
LOREGUARD_API_URL = "https://api.loreguard.com"


def get_local_proxy_url() -> str | None:
    """Get the local proxy URL from runtime.json.

    Returns:
        URL like "http://127.0.0.1:54321" or None if not available.
    """
    info = RuntimeInfo.load()
    if info and info.port:
        return f"http://127.0.0.1:{info.port}"
    return None


class DebugPassWidget(Static):
    """A single collapsible debug pass entry."""

    DEFAULT_CSS = f"""
    DebugPassWidget {{
        width: 100%;
        height: auto;
        padding: 0;
        margin: 0;
    }}

    DebugPassWidget .pass-header {{
        width: 100%;
        height: 1;
    }}

    DebugPassWidget .pass-details {{
        padding: 0 0 0 2;
        color: {FG_DIM};
    }}
    """

    def __init__(self, pass_num: str, name: str, payload: dict) -> None:
        super().__init__()
        self._pass_num = pass_num
        self._name = name
        self._payload = payload
        self._expanded = True  # Start expanded

    def compose(self) -> ComposeResult:
        yield Static(self._render_header(), classes="pass-header")
        # Show details by default (expanded)
        details_text = self._render_details()
        if details_text:
            yield Static(details_text, classes="pass-details")

    def _render_header(self) -> Text:
        """Render the pass header line."""
        text = Text()
        payload = self._payload
        duration = payload.get("durationMs", 0)
        skipped = payload.get("skipped", False)
        error = payload.get("error", False)
        retry_of = payload.get("retryOf", 0)
        retry_suffix = f" ↻{retry_of}" if retry_of > 0 else ""

        # Expand/collapse indicator
        indicator = "▼" if self._expanded else "►"

        if skipped:
            text.append(f"{indicator} ", style="#555566")
            text.append(f"Pass {self._pass_num}", style="#555566")
            text.append(f" {self._name} ", style="#555566")
            text.append("Skipped", style="#555566")
        elif error:
            text.append(f"{indicator} ", style="#AA5555")
            text.append(f"Pass {self._pass_num}", style="#AA5555")
            text.append(f" {self._name} ", style="#AA5555")
            text.append(f"Error{retry_suffix}", style="#AA5555")
        else:
            text.append(f"{indicator} ", style="#888899")
            text.append(f"Pass {self._pass_num}", style="#AAAAAA")
            text.append(f" {self._name} ", style="#888899")
            text.append(f"({duration}ms){retry_suffix}", style="#666677")

        return text

    def _render_details(self) -> Text:
        """Render the expanded details."""
        text = Text()
        payload = self._payload

        # Error message
        if error_msg := payload.get("errorMsg"):
            text.append(f"{error_msg}\n", style="#FF5555")

        # Intent output (Pass 0)
        if output := payload.get("output"):
            text.append(f"Output:\n", style=FG_DIM)
            for line in output.split("\n"):
                text.append(f"  {line}\n", style=FG)

        # Query rewrite (Pass 1)
        if query_rewrite := payload.get("queryRewrite"):
            text.append(f"Query: {query_rewrite}\n", style=FG)

        # Sources (Pass 1)
        if sources := payload.get("sources"):
            text.append(f"Sources ({len(sources)}):\n", style=FG_DIM)
            for src in sources:
                score = src.get("score", 0)
                path = src.get("path", "")
                text.append(f"  {path} ({score:.2f})\n", style=FG_DIM)

        # Evidence blocks (Pass 4)
        if evidence_blocks := payload.get("evidenceBlocks"):
            text.append(f"Evidence ({len(evidence_blocks)}):\n", style=FG_DIM)
            for block in evidence_blocks:
                block_text = block.get("text", "")
                text.append(f"  \"{block_text}\"\n", style=FG_DIM)

        # Citation answer (Pass 4)
        if citation_answer := payload.get("citationAnswer"):
            text.append(f"Citation answer:\n", style=FG_DIM)
            for line in citation_answer.split("\n"):
                text.append(f"  {line}\n", style=FG)

        # Verdict (Pass 2.5/4.5)
        if verdict := payload.get("verdict"):
            if verdict == "APPROVED":
                faith = payload.get("faithfulness")
                if faith is not None:
                    text.append(f"✓ APPROVED (faithfulness: {faith:.2f})\n", style=GREEN)
                else:
                    text.append(f"✓ APPROVED\n", style=GREEN)
            else:
                issues = payload.get("issues", [])
                text.append(f"✗ ISSUES ({len(issues)})\n", style="#FF5555")
                for issue in issues:
                    claim = issue.get("claim", "")
                    claim_type = issue.get("claimType", "")
                    severity = issue.get("severity", "")
                    type_info = f" [{claim_type}]" if claim_type else ""
                    sev_info = f" ({severity})" if severity else ""
                    text.append(f"  - {claim}{type_info}{sev_info}\n", style=FG_DIM)

        # Fail-closed info (Pass 4.5)
        if fail_closed := payload.get("failClosed"):
            reason = fail_closed.get("reason", "UNKNOWN")
            claims_stripped = fail_closed.get("claimsStripped", [])
            original_len = fail_closed.get("originalLen", 0)
            final_len = fail_closed.get("finalLen", 0)
            text.append(f"⚠ FAIL-CLOSED: {reason}\n", style="#FFAA00")
            text.append(f"  Stripped {len(claims_stripped)} claims\n", style=FG_DIM)
            text.append(f"  {original_len} → {final_len} chars\n", style=FG_DIM)
            for claim in claims_stripped:
                text.append(f"  • {claim}\n", style=FG_DIM)

        return text

    def on_click(self) -> None:
        """Toggle expanded state."""
        self._expanded = not self._expanded

        # Update header
        header = self.query_one(".pass-header", Static)
        header.update(self._render_header())

        # Show/hide details
        try:
            details = self.query_one(".pass-details", Static)
            details.remove()
        except Exception:
            if self._expanded:
                details = Static(self._render_details(), classes="pass-details")
                self.mount(details)


class DebugPanel(Vertical):
    """Right panel showing pipeline debug information."""

    DEFAULT_CSS = """
    DebugPanel {
        width: 1fr;
        height: 100%;
        padding: 0 1;
        margin: 0;
        border: solid #333344;
        background: #1a1b24;
    }

    DebugPanel .debug-header {
        height: 1;
        width: 100%;
        padding: 0;
        border-bottom: solid #333344;
    }

    DebugPanel .debug-scroll {
        height: 1fr;
        width: 100%;
        padding: 0;
    }

    DebugPanel .debug-summary {
        height: 1;
        width: 100%;
        padding: 0;
        border-top: solid #333344;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._passes: list[DebugPassWidget] = []
        self._summary_data: dict = {}

    def compose(self) -> ComposeResult:
        header = Text()
        header.append("Pipeline Debug ", style="bold #888899")
        header.append("[^D]", style=FG_DIM)
        yield Static(header, classes="debug-header")
        yield VerticalScroll(id="debug-scroll", classes="debug-scroll")
        yield Static("", id="debug-summary", classes="debug-summary")

    def clear(self) -> None:
        """Clear all debug info."""
        self._passes = []
        self._summary_data = {}
        try:
            scroll = self.query_one("#debug-scroll", VerticalScroll)
            scroll.remove_children()
            summary = self.query_one("#debug-summary", Static)
            summary.update("")
        except Exception:
            pass

    def add_pass(self, payload: dict) -> None:
        """Add a pipeline pass to the debug panel."""
        pass_num = str(payload.get("pass", "?"))
        name = payload.get("name", "Unknown")

        widget = DebugPassWidget(pass_num, name, payload)
        self._passes.append(widget)

        try:
            scroll = self.query_one("#debug-scroll", VerticalScroll)
            scroll.mount(widget)
            scroll.scroll_end(animate=False)
        except Exception:
            pass

    def set_summary(self, total_ms: int, backend_ms: int, tokens: int, verified: bool) -> None:
        """Set the summary line at the bottom."""
        self._summary_data = {
            "total_ms": total_ms,
            "backend_ms": backend_ms,
            "tokens": tokens,
            "verified": verified,
        }

        text = Text()
        indicator = "●" if verified else "○"
        text.append(f"✓ ", style=GREEN)
        text.append(f"{total_ms}ms total  {backend_ms}ms backend  {tokens} tok  {indicator}", style=FG_DIM)

        try:
            summary = self.query_one("#debug-summary", Static)
            summary.update(text)
        except Exception:
            pass

    def set_status(self, message: str, style: str = CYAN) -> None:
        """Set a status message in the summary."""
        try:
            summary = self.query_one("#debug-summary", Static)
            summary.update(Text(message, style=style))
        except Exception:
            pass


class NPCChat(Vertical):
    """Embedded NPC chat widget for the main screen.

    Uses the Loreguard API for grounded NPC conversations.
    Supports two-column layout in verbose mode with debug panel.
    """

    BINDINGS = [
        Binding("escape", "close_chat", "Close", show=False),
        Binding("ctrl+d", "toggle_debug", "Toggle Debug", show=False),
        Binding("ctrl+e", "switch_scenario", "Switch Scenario", show=False),
    ]

    DEFAULT_CSS = f"""
    NPCChat {{
        width: 100%;
        height: 1fr;
        background: transparent;
        display: none;
        padding: 0 1;
    }}

    NPCChat.visible {{
        display: block;
    }}

    NPCChat .chat-header {{
        height: 1;
        padding: 0 1;
    }}

    NPCChat .chat-main {{
        height: 1fr;
        width: 100%;
    }}

    NPCChat .chat-column {{
        width: 1fr;
        height: 100%;
        border: solid #333344;
        background: #1a1b24;
        margin: 0 1 0 0;
    }}

    NPCChat .chat-column-header {{
        height: 1;
        padding: 0 1;
        border-bottom: solid #333344;
    }}

    NPCChat .chat-container {{
        height: 1fr;
        padding: 0 1;
        background: transparent;
        overflow-x: hidden;
        overflow-y: auto;
    }}

    NPCChat .chat-message {{
        padding: 0;
        margin: 0;
        width: 100%;
        height: auto;
    }}

    NPCChat .user-message {{
        padding: 0;
        margin: 0;
    }}

    NPCChat .npc-message {{
        padding: 0;
        margin: 0;
    }}

    NPCChat Input {{
        width: 100%;
        border: none;
        border-top: solid #333344;
        background: transparent;
        padding: 0 1;
        margin: 0;
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

    NPCChat DebugPanel {{
        display: none;
    }}

    NPCChat.debug-visible DebugPanel {{
        display: block;
    }}

    NPCChat.debug-visible .chat-column {{
        width: 1fr;
    }}
    """

    def __init__(self) -> None:
        super().__init__()
        self._npc_id: str = ""
        self._npc_name: str = "NPC"
        self._api_token: str = ""
        self._verbose: bool = False
        self._messages: list[dict] = []
        self._player_handle = "Player"
        self._generating = False
        self._visible = False
        self._debug_visible = False
        # Scenario support
        self._scenarios: list[dict] = []  # [{id, name, isDefault}, ...]
        self._scenario_id: str | None = None

    def compose(self) -> ComposeResult:
        """Compose the chat widget with optional debug panel."""
        yield Static(self._render_header(), classes="chat-header")

        with Horizontal(classes="chat-main"):
            with Vertical(classes="chat-column"):
                yield Static(self._render_chat_column_header(), classes="chat-column-header")
                yield VerticalScroll(id="npc-chat-container", classes="chat-container")
                yield Static("", id="npc-status-line", classes="status-line")
                yield Input(placeholder=f"> Say something to {self._npc_name}...", id="npc-chat-input")
            yield DebugPanel()

        yield Static(self._render_footer(), classes="chat-footer")

    def _render_chat_column_header(self) -> Text:
        """Render the chat column header."""
        text = Text()
        text.append("Conversation", style="bold #888899")
        return text

    def _render_header(self) -> Text:
        """Render header with NPC name and current scenario."""
        text = Text()
        text.append(f"Chat with ", style=FG_DIM)
        text.append(f"{self._npc_name}", style=f"bold {FG}")
        # Show current scenario if set
        if self._scenario_id and self._scenarios:
            scenario = next((s for s in self._scenarios if s.get("id") == self._scenario_id), None)
            if scenario:
                text.append(f"  ", style=FG_DIM)
                text.append(f"[{scenario.get('name', self._scenario_id)}]", style=CYAN)
        return text

    def _render_footer(self) -> Text:
        """Render footer with key hints."""
        text = Text()
        text.append(" enter ", style=f"bold {FG} on #44475A")
        text.append(" send", style=FG_DIM)
        text.append("   ")
        text.append(" esc ", style=f"bold {FG} on #44475A")
        text.append(" close", style=FG_DIM)
        text.append("   ")
        text.append(" ^E ", style=f"bold {FG} on #44475A")
        text.append(" scenario", style=FG_DIM)
        text.append("   ")
        text.append(" ^D ", style=f"bold {FG} on #44475A")
        text.append(" debug", style=FG_DIM)
        return text

    def show(self) -> None:
        """Show the chat widget."""
        self._visible = True
        self.add_class("visible")
        # Add class to parent screen to adjust layout
        if self.screen:
            self.screen.add_class("chat-open")
        # Show debug panel if verbose mode
        if self._verbose:
            self._debug_visible = True
            self.add_class("debug-visible")
        # Focus the input
        self.query_one("#npc-chat-input", Input).focus()

    def hide(self) -> None:
        """Hide the chat widget."""
        self._visible = False
        self.remove_class("visible")
        self.remove_class("debug-visible")
        # Remove class from parent screen
        if self.screen:
            self.screen.remove_class("chat-open")

    def toggle(self) -> None:
        """Toggle chat visibility."""
        if self._visible:
            self.hide()
        else:
            self.show()

    def action_toggle_debug(self) -> None:
        """Toggle debug panel visibility."""
        self._debug_visible = not self._debug_visible
        if self._debug_visible:
            self._verbose = True  # Enable verbose API requests when debug is shown
            self.add_class("debug-visible")
        else:
            self.remove_class("debug-visible")

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
        # Reset scenarios
        self._scenarios = []
        self._scenario_id = None

        # Update header
        header = self.query_one(".chat-header", Static)
        header.update(self._render_header())

        # Update footer (shows ^D hint in verbose mode)
        footer = self.query_one(".chat-footer", Static)
        footer.update(self._render_footer())

        # Update input placeholder
        input_widget = self.query_one("#npc-chat-input", Input)
        input_widget.placeholder = f"> Say something to {name}..."

        # Clear chat container
        container = self.query_one("#npc-chat-container", VerticalScroll)
        container.remove_children()

        # Clear status
        status = self.query_one("#npc-status-line", Static)
        status.update("")

        # Clear debug panel
        try:
            debug_panel = self.query_one(DebugPanel)
            debug_panel.clear()
        except Exception:
            pass

        # Fetch scenarios for this NPC
        self.run_worker(self._fetch_scenarios())

    @property
    def is_visible(self) -> bool:
        """Check if chat is visible."""
        return self._visible

    async def _fetch_scenarios(self) -> None:
        """Fetch available scenarios for the current NPC."""
        if not self._npc_id or not self._api_token:
            return

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{LOREGUARD_API_URL}/api/engine/characters/scenarios/{self._npc_id}",
                    headers={"Authorization": f"Bearer {self._api_token}"},
                )
                if response.status_code == 200:
                    data = response.json()
                    self._scenarios = data.get("scenarios", [])
                    # Set default scenario if available
                    default = next((s for s in self._scenarios if s.get("isDefault")), None)
                    if default:
                        self._scenario_id = default.get("id")
                    elif self._scenarios:
                        self._scenario_id = self._scenarios[0].get("id")
                    # Update header to show scenario
                    header = self.query_one(".chat-header", Static)
                    header.update(self._render_header())
        except Exception:
            # Scenarios are optional, don't fail if not available
            pass

    def action_switch_scenario(self) -> None:
        """Open scenario selection palette."""
        if not self._visible:
            return

        if not self._scenarios:
            # Show message that no scenarios are available
            try:
                status = self.query_one("#npc-status-line", Static)
                status.update(Text("No scenarios available for this NPC", style=FG_DIM))
            except Exception:
                pass
            return

        from ..modals.unified_palette import UnifiedPaletteModal, PaletteItem

        # Build scenario items
        items = []
        for scenario in self._scenarios:
            scenario_id = scenario.get("id", "")
            name = scenario.get("name", scenario_id)
            is_current = scenario_id == self._scenario_id
            is_default = scenario.get("isDefault", False)

            desc = ""
            if is_current:
                desc = "current"
            elif is_default:
                desc = "default"

            items.append(PaletteItem(
                id=f"scenario-{scenario_id}",
                title=name,
                description=desc,
                category="scenario",
                data=scenario_id,
            ))

        def handle_result(result: tuple | None) -> None:
            if result and result[0] == "scenario":
                self._scenario_id = result[1]
                # Update header
                header = self.query_one(".chat-header", Static)
                header.update(self._render_header())
                # Focus input
                self.query_one("#npc-chat-input", Input).focus()

        self.app.push_screen(
            UnifiedPaletteModal(
                title="Select Scenario",
                items=items,
                show_models=False,
                show_commands=False,
            ),
            handle_result,
        )

    def _add_message(self, role: str, content: str) -> None:
        """Add a message to the chat display."""
        container = self.query_one("#npc-chat-container", VerticalScroll)

        text = Text()
        if role == "user":
            text.append("You: ", style=f"bold {FG}")
            text.append(content, style=FG_DIM)
            css_class = "chat-message user-message"
        else:
            text.append(f"{self._npc_name}: ", style=f"bold {FG}")
            text.append(content, style=FG)
            css_class = "chat-message npc-message"

        widget = Static(text, classes=css_class)
        container.mount(widget)
        container.scroll_end(animate=False)

    def _build_history(self) -> list[dict[str, str]]:
        """Build history payload for API requests."""
        history: list[dict[str, str]] = []
        for msg in self._messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history.append({"role": role, "content": content})
        return history

    def _add_npc_message(self, content: str) -> None:
        """Add an NPC message."""
        self._messages.append({"role": "assistant", "content": content})
        self._add_message("npc", content)

    def _add_user_message(self, content: str) -> None:
        """Add a user message."""
        self._messages.append({"role": "user", "content": content})
        self._add_message("user", content)

    def _submit_message(self, input_widget: Input) -> None:
        """Submit a chat message from the input widget."""
        if self._generating:
            return

        message = input_widget.value.strip()
        if not message:
            return

        input_widget.clear()
        input_widget.cursor_position = 0
        input_widget.focus()
        self._add_user_message(message)

        # Clear debug panel for new request
        if self._verbose:
            try:
                debug_panel = self.query_one(DebugPanel)
                debug_panel.clear()
                debug_panel.set_status("Processing...")
            except Exception:
                pass

        self._generate_response(message)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        self._submit_message(event.input)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes - / opens palette, /scenario switches scenario."""
        if self._visible and event.input.id == "npc-chat-input":
            value = event.input.value.lower()
            if value == "/":
                self._open_palette()
            elif value == "/scenario":
                event.input.clear()
                self.action_switch_scenario()

    def on_key(self, event) -> None:
        """Handle key events - capture Ctrl+D for debug toggle, Ctrl+E for scenario."""
        key = event.key
        if key in ("ctrl+d", "ctrl_d") and self._visible:
            self.action_toggle_debug()
            event.prevent_default()
            event.stop()
        elif key in ("ctrl+e", "ctrl_e") and self._visible:
            self.action_switch_scenario()
            event.prevent_default()
            event.stop()

    def _open_palette(self) -> None:
        """Open the unified palette for NPC commands."""
        if not self.screen:
            return

        try:
            from ..modals.unified_palette import UnifiedPaletteModal

            def handle_result(result: tuple | None) -> None:
                if not result:
                    return

                category, value = result
                if category == "command":
                    self._handle_command(value)

            self.app.push_screen(
                UnifiedPaletteModal(title="NPC Commands", show_models=False, show_commands=True),
                handle_result
            )
        except Exception:
            pass

    def _handle_command(self, cmd_id: str) -> None:
        """Handle command execution."""
        if cmd_id == "cmd-quit":
            self.app.exit()
        elif cmd_id == "cmd-scenario":
            self.action_switch_scenario()

    def _generate_response(self, user_message: str) -> None:
        """Generate NPC response using local LLM."""
        self._generating = True
        status = self.query_one("#npc-status-line", Static)
        status.update(Text(f"{self._npc_name} is thinking...", style=CYAN))

        self.run_worker(self._do_generate(user_message))

    async def _do_generate(self, user_message: str) -> None:
        """Worker to generate response using local proxy with SSE streaming."""
        status = self.query_one("#npc-status-line", Static)
        container = self.query_one("#npc-chat-container", VerticalScroll)

        if not self._npc_id or not self._api_token:
            status.update(Text("Not connected to an NPC", style="#FF5555"))
            self._generating = False
            return

        try:
            history = self._build_history()
            payload = {
                "character_id": self._npc_id,
                "message": user_message,
                "player_handle": self._player_handle,
                "player_id": self._player_handle,
                "history": history,
                "context": history,
            }

            # Add scenario if selected
            if self._scenario_id:
                payload["scenario_id"] = self._scenario_id

            if self._verbose:
                payload["verbose"] = True

            try:
                await self._do_generate_streaming(payload, status, container)
                return
            except httpx.ConnectError as e:
                local_url = get_local_proxy_url()
                if self._verbose:
                    if local_url:
                        status.update(Text(f"Local proxy failed ({local_url}): {e}", style=FG_DIM))
                    else:
                        status.update(Text("No runtime.json found - SDK server not running", style=FG_DIM))
                else:
                    status.update(Text("Local proxy unavailable, using cloud...", style=FG_DIM))
            except Exception as e:
                local_url = get_local_proxy_url()
                if self._verbose:
                    status.update(Text(f"Local proxy error ({local_url}): {type(e).__name__}: {e}", style="#FF5555"))
                else:
                    status.update(Text("Local proxy error, using cloud...", style=FG_DIM))

            await self._do_generate_cloud(payload, status)

        except httpx.ConnectError:
            status.update(Text("Cannot connect to Loreguard API", style="#FF5555"))
        except httpx.TimeoutException:
            status.update(Text("Request timed out - try again", style="#FF5555"))
        except Exception as e:
            status.update(Text(f"Error: {e}", style="#FF5555"))
        finally:
            self._generating = False

    async def _do_generate_streaming(self, payload: dict, status: Static, container: VerticalScroll) -> None:
        """Generate response using local proxy with SSE streaming."""
        import time
        client_start_time = time.time()

        local_proxy_url = get_local_proxy_url()
        if not local_proxy_url:
            raise httpx.ConnectError("Local proxy URL not found in runtime.json")

        # Widget created lazily on first token
        streaming_text: Text | None = None
        streaming_widget: Static | None = None

        tokens_received = 0
        speech = ""
        verified = False
        final_data = {}

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{local_proxy_url}/api/chat",
                headers={
                    "Authorization": f"Bearer {self._api_token}",
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
                json=payload,
            ) as response:
                if response.status_code != 200:
                    streaming_widget.remove()
                    err_text = await response.aread()
                    raise Exception(f"API error {response.status_code}: {err_text.decode()[:100]}")

                event_type = ""
                async for line in response.aiter_lines():
                    line = line.strip()

                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        data_str = line[5:].strip()
                        if not data_str:
                            continue

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        if event_type == "token":
                            token = data.get("t", "")
                            if token:
                                # Create widget on first token
                                if streaming_widget is None:
                                    streaming_text = Text()
                                    streaming_text.append(f"{self._npc_name}: ", style=f"bold {FG}")
                                    streaming_widget = Static(streaming_text, classes="chat-message npc-message")
                                    container.mount(streaming_widget)
                                tokens_received += 1
                                speech += token
                                streaming_text.append(token, style=FG)
                                streaming_widget.update(streaming_text)
                                container.scroll_end(animate=False)
                                status.update(Text(f"Streaming... ({tokens_received} tokens)", style=CYAN))

                        elif event_type == "done":
                            final_data = data
                            speech = data.get("speech", speech)
                            verified = data.get("verified", False)

                        elif event_type == "error":
                            error_msg = data.get("error", "Unknown error")
                            if streaming_widget is not None:
                                streaming_widget.remove()
                            raise Exception(error_msg)

        # Update final message with verification indicator
        if speech and streaming_widget is not None:
            indicator = "●" if verified else "○"
            final_text = Text()
            final_text.append(f"{self._npc_name}: ", style=f"bold {FG}")
            final_text.append(f"{speech} ", style=FG)
            final_text.append(indicator, style=GREEN if verified else FG_DIM)
            streaming_widget.update(final_text)

            self._messages.append({"role": "assistant", "content": speech})

            # Update debug panel summary
            if self._verbose and final_data:
                backend_latency = final_data.get("latency_ms", 0)
                client_latency = int((time.time() - client_start_time) * 1000)
                try:
                    debug_panel = self.query_one(DebugPanel)
                    debug_panel.set_summary(client_latency, backend_latency, tokens_received, verified)
                except Exception:
                    pass

            backend_latency = final_data.get("latency_ms", 0)
            client_latency = int((time.time() - client_start_time) * 1000)
            status.update(Text(f"Done ({client_latency}ms, {tokens_received} tokens)", style=GREEN))
        else:
            if streaming_widget is not None:
                streaming_widget.remove()
            status.update(Text("No response generated", style=FG_DIM))

    async def _do_generate_cloud(self, payload: dict, status: Static) -> None:
        """Fallback: Generate response using cloud API (no streaming)."""
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
                try:
                    err_text = response.text
                    try:
                        err_data = response.json()
                        err_msg = str(err_data)
                    except Exception:
                        err_msg = err_text[:500] if err_text else "No response body"
                    status.update(Text(f"API error {response.status_code}: {err_msg[:100]}", style="#FF5555"))
                except Exception as e:
                    status.update(Text(f"API error {response.status_code}: {e}", style="#FF5555"))

    def on_pass_update(self, payload: dict) -> None:
        """Handle pass_update message from backend (verbose mode).

        Called by the tunnel when it receives pass updates via WebSocket.
        """
        if not self._verbose or not self._visible:
            return

        # Add pass to debug panel instead of chat
        try:
            debug_panel = self.query_one(DebugPanel)
            debug_panel.add_pass(payload)
        except Exception:
            pass

    def action_close_chat(self) -> None:
        """Close the chat widget."""
        self.hide()
