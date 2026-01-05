"""Main screen - persistent base with banner that stays visible."""

import logging
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Static
from textual.containers import Container

from ..widgets.banner import LoreguardBanner
from ..widgets.hardware_info import HardwareInfo
from ..widgets.server_monitor import ServerMonitor
from ..widgets.npc_chat import NPCChat
from ...config import LoreguardConfig

if TYPE_CHECKING:
    from ..app import LoreguardApp

log = logging.getLogger(__name__)


def _resolve_backend_model_id(filename_stem: str) -> str:
    """Map local model filename to backend-accepted model ID.

    Backend accepts specific model IDs like qwen3-4b, qwen3-8b, external.
    We map the local model filename to the closest match.
    """
    MODEL_MAPPINGS = {
        "qwen3-4b": "qwen3-4b",
        "qwen3-8b": "qwen3-8b",
        "qwen3-0.6b": "qwen3-0.6b",
        "qwen3-1.7b": "qwen3-4b",  # Map to closest
        "llama-3": "llama-3.1-8b",
        "mistral": "mistral-7b",
        "phi-3": "phi-3",
        "tinyllama": "tinyllama",
    }

    search_str = filename_stem.lower()
    for pattern, backend_id in MODEL_MAPPINGS.items():
        if pattern in search_str:
            log.debug(f"Model ID mapped: {search_str} -> {backend_id}")
            return backend_id

    # Fallback to 'external' for custom/unknown models
    log.debug(f"Using 'external' model ID for: {search_str}")
    return "external"


class MainScreen(Screen):
    """Main screen with persistent banner - modals float on top."""

    BINDINGS = [
        Binding("/", "open_palette", "Commands", show=True),
        Binding("ctrl+p", "open_palette", "Commands", show=False),
        Binding("ctrl+n", "open_chat", "Chat", show=False),
        Binding("ctrl+m", "open_monitor", "Monitor", show=False),
        Binding("ctrl+l", "switch_model", "Model", show=False),
        Binding("ctrl+t", "toggle_theme", "Theme", show=False),
        Binding("ctrl+c", "quit", "Quit", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._status_text = "Starting..."

    DEFAULT_CSS = """
    MainScreen {
        layout: vertical;
        overflow: hidden;
    }

    MainScreen LoreguardBanner {
        width: 100%;
        height: 11;
        content-align: center top;
    }

    MainScreen HardwareInfo {
        width: 100%;
        height: 1;
        text-align: center;
        content-align: center middle;
    }

    MainScreen #main-content {
        width: 100%;
        height: 1fr;
        padding: 1 2;
    }

    MainScreen #main-status {
        width: 100%;
    }

    MainScreen NPCChat {
        width: 100%;
        height: 1fr;
        margin: 0 2;
    }

    MainScreen.chat-open #main-content {
        height: auto;
        max-height: 3;
    }

    MainScreen ServerMonitor {
        width: 100%;
        height: 1;
        dock: bottom;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the main screen with banner."""
        yield LoreguardBanner()
        yield HardwareInfo()
        with Container(id="main-content"):
            yield Static("", id="main-status")
        yield NPCChat()
        yield ServerMonitor()
        yield Footer()

    def on_mount(self) -> None:
        """Check for saved config or show auth modal."""
        # Delay to allow the banner to render first
        self.set_timer(0.1, self._check_saved_config)

    def _check_saved_config(self) -> None:
        """Check if we have saved configuration."""
        config = LoreguardConfig.load()
        app: "LoreguardApp" = self.app  # type: ignore

        if config.has_saved_config():
            # Use saved config
            app.api_token = config.api_token
            app.model_path = config.get_model_path_obj()
            app.dev_mode = config.dev_mode

            if config.dev_mode:
                app.worker_id = "dev-worker"
                self._update_status(f"Using saved config: {app.model_path.name if app.model_path else 'unknown'}")
            else:
                import socket
                app.worker_id = socket.gethostname().split(".")[0]
                self._update_status(f"Using saved config: {app.model_path.name if app.model_path else 'unknown'}")

            # Skip auth/model selection, go straight to services
            self._show_nli_setup()
        else:
            # No saved config, show auth modal
            self._show_auth_modal()

    def _show_auth_modal(self) -> None:
        """Show authentication modal."""
        from ..modals.auth_menu import AuthMenuModal

        def handle_auth(result: tuple | None) -> None:
            if result:
                token, worker_id, dev_mode = result
                app: "LoreguardApp" = self.app  # type: ignore
                app.api_token = token
                app.worker_id = worker_id
                app.dev_mode = dev_mode
                self._show_model_select()

        self.app.push_screen(AuthMenuModal(), handle_auth)

    def _show_model_select(self) -> None:
        """Show model selection modal."""
        from ..modals.unified_palette import UnifiedPaletteModal

        def handle_model(result: tuple | None) -> None:
            if result and result[0] == "model":
                model_path = result[1]
                app: "LoreguardApp" = self.app  # type: ignore
                app.model_path = model_path

                # Save config
                config = LoreguardConfig.load()
                config.api_token = app.api_token
                config.set_model_path(model_path)
                config.dev_mode = app.dev_mode
                config.save()

                self._show_nli_setup()

        # Show unified palette with only models (no commands during setup)
        self.app.push_screen(
            UnifiedPaletteModal(title="Step 2/4 Model", show_models=True, show_commands=False),
            handle_model
        )

    def _show_nli_setup(self) -> None:
        """Show NLI setup (progress in status, not modal)."""
        self._update_status("Setting up NLI model...")
        self.run_worker(self._do_nli_setup())

    async def _do_nli_setup(self) -> None:
        """Setup NLI model."""
        import asyncio
        import concurrent.futures
        from ...nli import is_nli_model_available, download_nli_model
        from ...wizard import suppress_external_output

        if is_nli_model_available():
            self._update_status("NLI model ready")
            await asyncio.sleep(0.5)
            self._start_services()
            return

        self._update_status("Downloading NLI model (~1.4GB)...")

        try:
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                with suppress_external_output():
                    success = await loop.run_in_executor(pool, download_nli_model)

            if success:
                self._update_status("NLI model ready")
            else:
                self._update_status("NLI download failed - continuing without")

            await asyncio.sleep(0.5)
            self._start_services()

        except Exception as e:
            self._update_status(f"NLI setup failed: {e}")
            await asyncio.sleep(1)
            self._start_services()

    def _start_services(self) -> None:
        """Start llama-server and backend connection."""
        self._update_status("Starting services...")
        self.run_worker(self._do_start_services())

    async def _do_start_services(self) -> None:
        """Worker to start services."""
        import asyncio
        from ...llama_server import LlamaServerProcess, is_llama_server_installed, download_llama_server
        from ...wizard import suppress_external_output

        app: "LoreguardApp" = self.app  # type: ignore

        if not app.model_path:
            self._update_status("No model selected")
            return

        # Download llama-server if needed
        if not is_llama_server_installed():
            self._update_status("Downloading llama-server...")
            try:
                await download_llama_server(lambda msg, prog: None)
                self._update_status("llama-server downloaded")
            except Exception as e:
                self._update_status(f"Failed: {e}")
                return

        # Start llama-server
        self._update_status("Starting llama-server...")
        app._llama_process = LlamaServerProcess(app.model_path, port=8080)
        app._llama_process.start()

        self._update_status("Loading model...")
        ready = await app._llama_process.wait_for_ready(timeout=120.0)

        if not ready:
            app._llama_process.stop()
            self._update_status("llama-server failed to start")
            return

        self._update_status("llama-server running on :8080")

        # Connect backend if not dev mode
        if not app.dev_mode:
            await self._connect_backend()
        else:
            self._update_status("Ready (dev mode - local only)")
            self._update_connection_status("dev")

    async def _connect_backend(self) -> None:
        """Connect to Loreguard backend."""
        import asyncio
        from ...tunnel import BackendTunnel
        from ...llm import LLMProxy
        from ...nli import NLIService
        from ...wizard import suppress_external_output

        app: "LoreguardApp" = self.app  # type: ignore

        self._update_status("Connecting to backend...")
        self._update_connection_status("connecting")

        try:
            llm_proxy = LLMProxy("http://127.0.0.1:8080")

            # Load NLI service
            nli_service = None
            try:
                nli_service = NLIService()
                with suppress_external_output():
                    model_loaded = nli_service.load_model()
                if not model_loaded:
                    nli_service = None
            except Exception:
                nli_service = None

            model_id = _resolve_backend_model_id(app.model_path.stem) if app.model_path else "unknown"

            def log_callback(msg: str, lvl: str) -> None:
                # Always show errors and warnings
                if lvl in ("error", "warn"):
                    self._update_status(msg)

                # Update connection status based on messages
                if "Registered as worker" in msg:
                    self._update_status("Ready - connected to Loreguard")
                    self._update_connection_status("connected")
                elif "Connection failed" in msg or "Connection rejected" in msg:
                    self._update_connection_status("disconnected")
                elif "Connection closed" in msg or "Disconnected" in msg:
                    self._update_connection_status("disconnected")
                elif "Connecting to" in msg:
                    self._update_status("Connecting to backend...")
                    self._update_connection_status("connecting")
                elif "Reconnecting" in msg:
                    self._update_connection_status("connecting")

            app._tunnel = BackendTunnel(
                backend_url="wss://api.loreguard.com/workers",
                llm_proxy=llm_proxy,
                worker_id=app.worker_id,
                worker_token=app.api_token,
                model_id=model_id,
                nli_service=nli_service,
                log_callback=log_callback,
                max_retries=0,  # Single try, no retries
            )

            # Wire up request completion callback to update server monitor
            def on_request_complete(npc: str, tokens: int, ttft_ms: float, total_ms: float) -> None:
                tps = (tokens / total_ms * 1000) if total_ms > 0 else 0
                self._update_server_monitor(tokens, tps)

            app._tunnel.on_request_complete = on_request_complete

            # Wire up pass update callback to chat widget (for verbose mode)
            def on_pass_update(payload: dict) -> None:
                try:
                    chat = self.query_one(NPCChat)
                    chat.on_pass_update(payload)
                except Exception:
                    pass

            app._tunnel.on_pass_update = on_pass_update

            # Start connection in background - let log_callback handle status updates
            asyncio.create_task(app._tunnel.connect())

        except Exception as e:
            self._update_status(f"Backend error: {e}")
            self._update_connection_status("disconnected")

    def _update_status(self, text: str) -> None:
        """Update the status text."""
        self._status_text = text
        try:
            status = self.query_one("#main-status", Static)
            status.update(f"  {text}")
        except Exception:
            pass

    def _update_connection_status(self, status: str) -> None:
        """Update the connection status indicator."""
        try:
            hw_info = self.query_one(HardwareInfo)
            hw_info.set_connection_status(status)
        except Exception:
            pass

    def _update_server_monitor(self, tokens: int, tps: float) -> None:
        """Update the server monitor with request stats."""
        try:
            monitor = self.query_one(ServerMonitor)
            monitor.record_request(tokens, tps)
        except Exception:
            pass

    def action_open_palette(self) -> None:
        """Open unified search palette."""
        from ..modals.unified_palette import UnifiedPaletteModal

        def handle_result(result: tuple | None) -> None:
            if not result:
                return

            category, value = result
            app: "LoreguardApp" = self.app  # type: ignore

            if category == "command":
                self._handle_command(value)
            elif category == "model":
                # Model selected - update app state
                app.model_path = value
                self._update_status(f"Model: {value.name}")
                # If services not started, start them
                if not app._llama_process:
                    self._start_services()

        self.app.push_screen(UnifiedPaletteModal(title="Commands", show_models=False), handle_result)

    def _handle_command(self, cmd_id: str) -> None:
        """Handle command execution."""
        if cmd_id == "cmd-quit":
            self.app.exit()
        elif cmd_id == "cmd-chat":
            self._open_npc_chat()
        elif cmd_id == "cmd-monitor":
            self._update_status("Server monitor not yet implemented")
        elif cmd_id == "cmd-switch-model":
            self._switch_model()
        elif cmd_id == "cmd-change-token":
            self._change_token()
        elif cmd_id == "cmd-logout":
            self._logout()
        elif cmd_id == "cmd-restart":
            self._restart_server()
        elif cmd_id == "cmd-theme":
            self._open_theme_picker()
        elif cmd_id == "cmd-keys":
            # Show key bindings
            bindings = []
            for binding in self.BINDINGS:
                bindings.append(f"{binding.key}: {binding.description}")
            self._update_status(" | ".join(bindings))
        elif cmd_id == "cmd-screenshot":
            path = self.app.save_screenshot()
            self._update_status(f"Screenshot saved: {path}")

    def _open_npc_chat(self) -> None:
        """Open NPC picker - fetches NPCs from Loreguard API."""
        app: "LoreguardApp" = self.app  # type: ignore

        # Check if we have an API token
        if not app.api_token:
            self._update_status("Login first to chat with NPCs")
            return

        # Fetch NPCs asynchronously
        self._update_status("Loading NPCs...")
        self.run_worker(self._fetch_and_show_npcs())

    async def _fetch_and_show_npcs(self) -> None:
        """Fetch NPCs from API and show picker."""
        from ..modals.unified_palette import UnifiedPaletteModal, PaletteItem
        import httpx

        app: "LoreguardApp" = self.app  # type: ignore

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    "https://api.loreguard.com/api/characters",
                    headers={"Authorization": f"Bearer {app.api_token}"},
                )

                if response.status_code == 200:
                    characters = response.json()

                    if not characters:
                        self._update_status("No NPCs registered. Create NPCs at loreguard.com first.")
                        return

                    # Create NPC items
                    npc_items = []
                    for char in characters:
                        npc_items.append(PaletteItem(
                            id=f"npc-{char['id']}",
                            title=char["name"],
                            description="",
                            category="npc",
                            shortcut="",
                            icon="",
                            data=char,  # Store full character data
                        ))

                    def handle_npc(result: tuple | None) -> None:
                        if result:
                            category, value = result
                            if category == "npc":
                                npc_id = value.replace("npc-", "")
                                # Find the character data
                                char_data = next((c for c in characters if c["id"] == npc_id), None)
                                if char_data:
                                    self._start_npc_chat(npc_id, char_data["name"])

                    self._update_status("")
                    self.app.push_screen(
                        UnifiedPaletteModal(title="Chat with NPC", items=npc_items, show_models=False, show_commands=False),
                        handle_npc
                    )
                elif response.status_code == 401:
                    self._update_status("Invalid token. Please re-authenticate.")
                else:
                    self._update_status(f"Failed to fetch NPCs: {response.status_code}")

        except httpx.ConnectError:
            self._update_status("Cannot connect to Loreguard API")
        except Exception as e:
            self._update_status(f"Error: {e}")

    def _start_npc_chat(self, npc_id: str, npc_name: str) -> None:
        """Start chat with selected NPC using Loreguard API."""
        app: "LoreguardApp" = self.app  # type: ignore

        try:
            chat = self.query_one(NPCChat)
            chat.set_npc(npc_id, npc_name, app.api_token, app.verbose)
            chat.toggle()
        except Exception:
            pass

    def _switch_model(self) -> None:
        """Switch to a different model."""
        from ..modals.unified_palette import UnifiedPaletteModal

        app: "LoreguardApp" = self.app  # type: ignore

        def handle_model(result: tuple | None) -> None:
            if result and result[0] == "model":
                model_path = result[1]

                # Stop current server if running
                if app._llama_process:
                    self._update_status("Stopping current server...")
                    app._llama_process.stop()
                    app._llama_process = None

                # Update model
                app.model_path = model_path

                # Save config
                config = LoreguardConfig.load()
                config.set_model_path(model_path)
                config.save()

                self._update_status(f"Switched to {model_path.name}")
                self._start_services()

        self.app.push_screen(
            UnifiedPaletteModal(title="Switch Model", show_models=True, show_commands=False),
            handle_model
        )

    def _open_theme_picker(self) -> None:
        """Open theme picker."""
        from ..modals.unified_palette import UnifiedPaletteModal, PaletteItem

        # Get available themes
        themes = list(self.app.available_themes.keys())
        current_theme = self.app.theme

        # Create theme items
        theme_items = []
        for theme_name in sorted(themes):
            icon = "✓" if theme_name == current_theme else ""
            theme_items.append(PaletteItem(
                id=f"theme-{theme_name}",
                title=theme_name,
                description="",
                category="theme",
                shortcut="",
                icon=icon,
            ))

        def handle_theme(result: tuple | None) -> None:
            if result:
                category, value = result
                if category == "theme":
                    theme_name = value.replace("theme-", "")
                    self.app.theme = theme_name
                    self._update_status(f"Theme: {theme_name}")

        self.app.push_screen(
            UnifiedPaletteModal(title="Theme", items=theme_items, show_models=False, show_commands=False),
            handle_theme
        )

    def _change_token(self) -> None:
        """Change the API token."""
        from ..modals.auth_menu import AuthMenuModal

        app: "LoreguardApp" = self.app  # type: ignore

        def handle_auth(result: tuple | None) -> None:
            if result:
                token, worker_id, dev_mode = result
                app.api_token = token
                app.worker_id = worker_id
                app.dev_mode = dev_mode

                # Save config
                config = LoreguardConfig.load()
                config.api_token = token
                config.dev_mode = dev_mode
                config.save()

                self._update_status("Token updated")

                # Reconnect to backend if not dev mode
                if not dev_mode and app._tunnel:
                    self._update_status("Reconnecting to backend...")
                    self.run_worker(self._reconnect_backend())

        self.app.push_screen(AuthMenuModal(), handle_auth)

    async def _reconnect_backend(self) -> None:
        """Reconnect to the backend with new credentials."""
        app: "LoreguardApp" = self.app  # type: ignore

        if app._tunnel:
            await app._tunnel.disconnect()
            app._tunnel = None

        await self._connect_backend()

    def _logout(self) -> None:
        """Clear saved credentials and restart setup."""
        app: "LoreguardApp" = self.app  # type: ignore

        # Stop services
        if app._llama_process:
            app._llama_process.stop()
            app._llama_process = None

        if app._tunnel:
            self.run_worker(app._tunnel.disconnect())
            app._tunnel = None

        # Clear config
        config = LoreguardConfig.load()
        config.api_token = ""
        config.model_path = ""
        config.dev_mode = False
        config.save()

        # Clear app state
        app.api_token = ""
        app.worker_id = ""
        app.model_path = None
        app.dev_mode = False

        self._update_status("Logged out")
        self._update_connection_status("")

        # Show auth modal again
        self.set_timer(0.5, self._show_auth_modal)

    def _restart_server(self) -> None:
        """Restart the llama-server."""
        app: "LoreguardApp" = self.app  # type: ignore

        if app._llama_process:
            self._update_status("Restarting llama-server...")
            app._llama_process.stop()
            app._llama_process = None
            self._start_services()
        else:
            self._update_status("No server running")

    def action_open_chat(self) -> None:
        """Open NPC chat."""
        self._open_npc_chat()

    def action_open_monitor(self) -> None:
        """Open server monitor."""
        self._update_status("Server monitor not yet implemented")

    def action_switch_model(self) -> None:
        """Switch model."""
        self._switch_model()

    def action_toggle_theme(self) -> None:
        """Open theme picker."""
        self._open_theme_picker()

    def action_quit(self) -> None:
        """Quit the app."""
        self.app.exit()
