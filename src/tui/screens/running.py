"""Running screen - Step 4/4."""

import asyncio
import concurrent.futures
from typing import TYPE_CHECKING, Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Static, RichLog
from textual.containers import Vertical
from textual.worker import Worker
from rich.text import Text

from ..widgets.banner import LoreguardBanner
from ..widgets.hardware_info import HardwareInfo
from ..widgets.footer import LoreguardFooter
from ..styles import CYAN, GREEN, YELLOW, RED, FG_DIM

if TYPE_CHECKING:
    from ..app import LoreguardApp


class RunningScreen(Screen):
    """Main running screen with status panel and activity log."""

    BINDINGS = [
        Binding("/", "command_palette", "Commands", show=True),
        Binding("ctrl+c", "quit", "Quit", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._llama_process = None
        self._tunnel = None
        self._request_count = 0
        self._total_tokens = 0

    def compose(self) -> ComposeResult:
        """Compose the running screen."""
        yield LoreguardBanner()
        yield HardwareInfo()
        yield Static("", classes="spacer")
        yield Vertical(
            Static("", id="status-server"),
            Static("", id="status-model"),
            Static("", id="status-backend"),
            Static("", id="status-nli"),
            Static("", id="status-intent"),
            Static("", id="status-mode"),
            Static("", classes="spacer"),
            Static("", id="status-requests"),
            Static("", id="status-tokens"),
            id="status-panel",
        )
        yield Static("", classes="spacer")
        yield RichLog(id="activity-log", highlight=True, markup=True, wrap=True)
        yield LoreguardFooter()

    def on_mount(self) -> None:
        """Start services on mount."""
        self._update_status("server", "llama-server", "Starting...")
        self._update_status("requests", "Requests", "0")
        self._update_status("tokens", "Tokens", "0")
        self.run_worker(self._start_services(), exclusive=True)

    def _update_status(self, status_id: str, label: str, value: str, status_type: str = "normal") -> None:
        """Update a status line."""
        try:
            widget = self.query_one(f"#status-{status_id}", Static)
        except Exception:
            return

        text = Text()
        text.append(f"{label}: ", style=FG_DIM)

        color = FG_DIM
        if status_type == "success":
            color = GREEN
        elif status_type == "warning":
            color = YELLOW
        elif status_type == "error":
            color = RED
        elif status_type == "info":
            color = CYAN

        text.append(value, style=color)
        widget.update(text)

    def _log(self, message: str, level: str = "info") -> None:
        """Add a message to the activity log."""
        try:
            log_widget = self.query_one("#activity-log", RichLog)
        except Exception:
            return

        prefix = ""
        if level == "error":
            prefix = f"[{RED}]ERROR:[/] "
        elif level == "warning":
            prefix = f"[{YELLOW}]WARN:[/] "
        elif level == "success":
            prefix = f"[{GREEN}]OK:[/] "
        elif level == "info":
            prefix = f"[{CYAN}]INFO:[/] "

        log_widget.write(f"{prefix}{message}")

    async def _start_services(self) -> None:
        """Start llama-server and connect to backend."""
        from ...llama_server import LlamaServerProcess, is_llama_server_installed, download_llama_server
        from ...wizard import suppress_external_output

        app: "LoreguardApp" = self.app  # type: ignore

        if not app.model_path:
            self._log("No model selected", "error")
            return

        # Download llama-server if needed
        if not is_llama_server_installed():
            self._update_status("server", "llama-server", "Downloading...", "info")
            try:
                def on_progress(msg: str, prog):
                    if prog:
                        self._update_status("server", "llama-server", f"Downloading... {int(prog.percent)}%", "info")

                await download_llama_server(on_progress)
                self._update_status("server", "llama-server", "Downloaded", "success")
            except Exception as e:
                self._update_status("server", "llama-server", f"Failed: {e}", "error")
                return

        # Start llama-server
        self._update_status("server", "llama-server", "Starting...")
        self._update_status("model", "Model", app.model_path.name)
        self._log(f"Starting llama-server with {app.model_path.name}", "info")

        self._llama_process = LlamaServerProcess(app.model_path, port=8080)
        self._llama_process.start()

        # Wait for model to load with progress updates
        self._log("Loading LLM model, this may take a minute...", "info")
        import time
        import httpx

        start = time.time()
        timeout = 120.0
        ready = False

        async with httpx.AsyncClient() as client:
            while time.time() - start < timeout:
                elapsed = int(time.time() - start)
                self._update_status("server", "llama-server", f"Loading model... ({elapsed}s)", "info")

                # Check if process died
                if self._llama_process.process and self._llama_process.process.poll() is not None:
                    self._log("llama-server process terminated unexpectedly", "error")
                    break

                try:
                    response = await client.get(
                        f"http://127.0.0.1:{self._llama_process.port}/health",
                        timeout=5.0
                    )
                    if response.status_code == 200:
                        ready = True
                        break
                    # 503 means "loading" - server is up but model not ready yet
                except httpx.RequestError:
                    pass

                await asyncio.sleep(1.0)

        if not ready:
            self._llama_process.stop()
            elapsed = int(time.time() - start)
            self._update_status("server", "llama-server", f"Failed after {elapsed}s", "error")
            self._log(f"llama-server failed to start after {elapsed}s", "error")
            return

        elapsed = int(time.time() - start)
        self._update_status("server", "llama-server", f"Running on :8080 ({elapsed}s)", "success")
        self._log(f"LLM ready in {elapsed}s", "success")

        # Connect backend
        if not app.dev_mode:
            self._update_status("backend", "Backend", "Connecting...", "info")
            try:
                from ...tunnel import BackendTunnel
                from ...llm import LLMProxy
                from ...nli import NLIService
                from ...intent_classifier import IntentClassifier

                llm_proxy = LLMProxy("http://127.0.0.1:8080")

                # Load NLI service (run in thread pool to not block event loop)
                nli_service = None
                self._update_status("nli", "NLI", "Loading...", "info")
                self._log("Loading NLI model...", "info")
                try:
                    nli_service = NLIService()
                    loop = asyncio.get_event_loop()
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        with suppress_external_output():
                            model_loaded = await loop.run_in_executor(pool, nli_service.load_model)
                    if model_loaded:
                        self._update_status("nli", "NLI", f"Ready ({nli_service.device})", "success")
                        self._log(f"NLI ready on {nli_service.device}", "success")
                    else:
                        self._update_status("nli", "NLI", "Failed", "warning")
                        self._log("NLI model failed to load", "warning")
                        nli_service = None
                except Exception as e:
                    self._update_status("nli", "NLI", f"Error: {e}", "warning")
                    self._log(f"NLI error: {e}", "error")
                    nli_service = None

                # Load Intent Classifier (ADR-0010)
                intent_classifier = None
                self._update_status("intent", "Intent", "Loading...", "info")
                self._log("Loading Intent classifier...", "info")
                try:
                    intent_classifier = IntentClassifier()
                    loop = asyncio.get_event_loop()
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        with suppress_external_output():
                            intent_loaded = await loop.run_in_executor(pool, intent_classifier.load_model)
                    if intent_loaded:
                        self._update_status("intent", "Intent", f"Ready ({intent_classifier.device})", "success")
                        self._log(f"Intent classifier ready on {intent_classifier.device}", "success")
                    else:
                        self._update_status("intent", "Intent", "Failed", "warning")
                        self._log("Intent classifier failed to load", "warning")
                        intent_classifier = None
                except Exception as e:
                    self._update_status("intent", "Intent", f"Error: {e}", "warning")
                    self._log(f"Intent classifier error: {e}", "error")
                    intent_classifier = None

                # Get model ID for backend
                model_id = app.model_path.stem

                self._tunnel = BackendTunnel(
                    backend_url="wss://api.loreguard.com/workers",
                    llm_proxy=llm_proxy,
                    worker_id=app.worker_id,
                    worker_token=app.api_token,
                    model_id=model_id,
                    nli_service=nli_service,
                    intent_classifier=intent_classifier,
                    log_callback=self._log,
                )
                self._tunnel.on_request_complete = self._on_request_complete

                asyncio.create_task(self._tunnel.connect())
                await asyncio.sleep(2)
                self._update_status("backend", "Backend", "Connected", "success")
                self._log("Connected to Loreguard backend", "success")

            except Exception as e:
                self._update_status("backend", "Backend", f"Error: {e}", "error")
                self._log(f"Backend connection failed: {e}", "error")
        else:
            self._update_status("mode", "Mode", "Dev (local only)", "info")
            self._log("Running in dev mode (local only)", "info")

        self._log("Ready for NPC requests", "success")

    def _on_request_complete(self, npc: str, tokens: int, ttft_ms: float, total_ms: float) -> None:
        """Handle completed request."""
        self._request_count += 1
        self._total_tokens += tokens
        tps = (tokens / total_ms * 1000) if total_ms > 0 else 0

        self._update_status("requests", "Requests", str(self._request_count))
        self._update_status("tokens", "Tokens", f"{self._total_tokens:,}")
        self._log(f"{npc}: {tokens} tok @ {tps:.1f} tk/s", "success")

    def action_command_palette(self) -> None:
        """Open the command palette."""
        from ..modals.command_palette import CommandPaletteModal

        def handle_command(result: str | None) -> None:
            if result == "quit":
                self.app.exit()
            elif result == "chat":
                self._run_npc_chat()

        self.app.push_screen(CommandPaletteModal(), handle_command)

    def _run_npc_chat(self) -> None:
        """Run NPC chat modal."""
        app: "LoreguardApp" = self.app  # type: ignore
        if app.dev_mode:
            self._log("Chat not available in dev mode", "warning")
            return

        # For now, log a message - full chat modal can be implemented later
        self._log("Opening NPC chat...", "info")

    def action_quit(self) -> None:
        """Handle quit action."""
        self._cleanup()
        self.app.exit()

    def _cleanup(self) -> None:
        """Cleanup resources."""
        if self._llama_process:
            self._llama_process.stop()
        if self._tunnel:
            asyncio.create_task(self._tunnel.disconnect())
