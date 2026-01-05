"""NLI setup screen - Step 3/4."""

import asyncio
import concurrent.futures
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Footer, Static, ProgressBar
from textual.containers import Center, Vertical
from rich.text import Text

from ..widgets.banner import LoreguardBanner
from ..widgets.hardware_info import HardwareInfo
from ..styles import CYAN, GREEN, YELLOW, RED

if TYPE_CHECKING:
    from ..app import LoreguardApp


class NLISetupScreen(Screen):
    """NLI model setup screen with automatic download."""

    def compose(self) -> ComposeResult:
        """Compose the NLI setup screen."""
        yield LoreguardBanner()
        yield HardwareInfo()
        yield Static("", classes="spacer")
        yield Static("Step 3/4: NLI Model Setup", classes="screen-title")
        yield Static("Setting up Natural Language Inference model", classes="screen-subtitle")
        yield Center(
            Vertical(
                Static("Checking NLI model...", id="nli-status"),
                ProgressBar(id="nli-progress", show_percentage=True),
                id="nli-container",
            )
        )
        yield Footer()

    def on_mount(self) -> None:
        """Start NLI setup on mount."""
        progress = self.query_one("#nli-progress", ProgressBar)
        progress.display = False
        self.run_worker(self._setup_nli(), exclusive=True)

    async def _setup_nli(self) -> None:
        """Setup NLI model."""
        from ...nli import is_nli_model_available, download_nli_model
        from ...wizard import suppress_external_output

        status = self.query_one("#nli-status", Static)
        progress = self.query_one("#nli-progress", ProgressBar)
        app: "LoreguardApp" = self.app  # type: ignore

        status.update(Text("Checking NLI model...", style=f"{CYAN}"))

        # Check if already available
        if is_nli_model_available():
            status.update(Text("NLI model ready", style=f"bold {GREEN}"))
            await asyncio.sleep(0.5)
            app.proceed_to_running()
            return

        # Need to download
        status.update(Text("Downloading NLI model (~1.4GB)...", style=f"{CYAN}"))
        progress.display = True
        progress.update(total=100, progress=0)

        try:
            # Run sync download in thread to not block
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                # Suppress transformers output during download
                with suppress_external_output():
                    success = await loop.run_in_executor(pool, download_nli_model)

            if success:
                status.update(Text("NLI model ready", style=f"bold {GREEN}"))
                progress.update(progress=100)
            else:
                status.update(Text("NLI download failed - continuing without", style=f"{YELLOW}"))

            progress.display = False
            await asyncio.sleep(0.5)
            app.proceed_to_running()

        except Exception as e:
            status.update(Text(f"NLI setup failed: {e}", style=f"{YELLOW}"))
            progress.display = False
            await asyncio.sleep(1)
            # Continue anyway
            app.proceed_to_running()
