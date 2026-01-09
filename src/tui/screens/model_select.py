"""Model selection screen - Step 2/4."""

from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Vertical
from textual.screen import Screen
from textual.widgets import Static, ListView, ListItem, Label, Input, ProgressBar
from rich.text import Text

from ..widgets.banner import LoreguardBanner
from ..widgets.hardware_info import HardwareInfo, HardwareData, detect_hardware
from ..widgets.footer import LoreguardFooter
from ..styles import CYAN, PINK, GREEN, YELLOW, RED, FG_DIM, FG

if TYPE_CHECKING:
    from ..app import LoreguardApp


class ModelSelectScreen(Screen):
    """Model selection screen with filterable list."""

    BINDINGS = [
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("enter", "select", "Select", show=True),
        Binding("escape", "back", "Back", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._hardware: HardwareData | None = None
        self._models_dir: Path | None = None

    def compose(self) -> ComposeResult:
        """Compose the model selection screen."""
        yield LoreguardBanner()
        yield HardwareInfo()
        yield Static("", classes="spacer")
        yield Static("Step 2/4: Model Selection", classes="screen-title")
        yield Static("Choose a model to run (type to filter)", classes="screen-subtitle")
        yield Center(
            Vertical(
                Input(placeholder="> Type to filter...", id="model-filter"),
                ListView(id="model-list"),
                id="model-container",
            )
        )
        yield Static("", id="status-message")
        yield ProgressBar(id="download-progress", show_percentage=True)
        yield LoreguardFooter()

    def on_mount(self) -> None:
        """Load models and focus the filter."""
        from ..widgets.hardware_info import detect_hardware
        from ...llama_server import get_models_dir

        self._hardware = detect_hardware()
        self._models_dir = get_models_dir()
        self._load_models()

        # Hide progress bar initially
        self.query_one("#download-progress", ProgressBar).display = False

        # Focus filter input
        self.query_one("#model-filter", Input).focus()

    def _load_models(self, filter_text: str = "") -> None:
        """Load models into the list."""
        from ...models_registry import SUPPORTED_MODELS

        model_list = self.query_one("#model-list", ListView)
        model_list.clear()

        filter_lower = filter_text.lower()

        for model in SUPPORTED_MODELS:
            # Filter by name or description
            if filter_lower and filter_lower not in model.name.lower() and filter_lower not in model.description.lower():
                continue

            # Build display text
            is_installed = self._models_dir and (self._models_dir / model.filename).exists()
            tags = []
            if is_installed:
                tags.append("installed")
            else:
                tags.append(f"{model.size_gb:.1f}GB")
            if model.recommended:
                tags.append("recommended")

            tag_str = " • ".join(tags)
            label_text = Text()
            # Use white for names, with green checkmark for installed
            if is_installed:
                label_text.append("✓ ", style=f"bold {GREEN}")
            label_text.append(model.name, style=f"bold {FG}")
            label_text.append(f"  [{tag_str}]", style=FG_DIM)

            # Sanitize ID for Textual (replace . with _)
            safe_id = model.id.replace(".", "_")
            item = ListItem(Label(label_text), id=f"model-{safe_id}")
            item.model_id = model.id  # type: ignore (keep original for lookup)
            model_list.append(item)

        # Add custom path option
        custom_label = Text()
        custom_label.append("Custom path...", style="bold")
        custom_label.append("  [use your own .gguf file]", style=FG_DIM)
        custom_item = ListItem(Label(custom_label), id="model-__custom__")
        custom_item.model_id = "__custom__"  # type: ignore
        model_list.append(custom_item)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle filter input changes."""
        if event.input.id == "model-filter":
            self._load_models(event.value)

    def action_cursor_up(self) -> None:
        """Move cursor up in list."""
        model_list = self.query_one("#model-list", ListView)
        model_list.action_cursor_up()

    def action_cursor_down(self) -> None:
        """Move cursor down in list."""
        model_list = self.query_one("#model-list", ListView)
        model_list.action_cursor_down()

    def action_select(self) -> None:
        """Handle selection."""
        model_list = self.query_one("#model-list", ListView)
        selected = model_list.highlighted_child
        if selected and hasattr(selected, "model_id"):
            self._handle_model_selection(selected.model_id)  # type: ignore

    def action_back(self) -> None:
        """Go back to previous screen."""
        self.app.pop_screen()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle list item selection."""
        if hasattr(event.item, "model_id"):
            self._handle_model_selection(event.item.model_id)  # type: ignore

    def _handle_model_selection(self, model_id: str) -> None:
        """Handle model selection."""
        if model_id == "__custom__":
            self._show_custom_path_input()
        else:
            self._select_model(model_id)

    def _show_custom_path_input(self) -> None:
        """Show input for custom model path."""
        # Simple inline approach - replace filter with path input
        filter_input = self.query_one("#model-filter", Input)
        filter_input.placeholder = "Enter path to .gguf file..."
        filter_input.value = ""
        filter_input.id = "custom-path-input"
        filter_input.focus()

        # Update list to show hint
        model_list = self.query_one("#model-list", ListView)
        model_list.clear()
        model_list.append(ListItem(Label("Press Enter to use path, Escape to cancel")))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        if event.input.id == "custom-path-input":
            path_str = event.value.strip()
            if path_str:
                model_path = Path(path_str.replace("~", str(Path.home())))
                if model_path.exists() and model_path.suffix == ".gguf":
                    self._proceed_with_model(model_path)
                else:
                    status = self.query_one("#status-message", Static)
                    status.update(Text(f"File not found or not a .gguf: {model_path}", style=f"bold {RED}"))

    def _select_model(self, model_id: str) -> None:
        """Select a model by ID."""
        from ...models_registry import SUPPORTED_MODELS

        model = next((m for m in SUPPORTED_MODELS if m.id == model_id), None)
        if not model or not self._models_dir:
            return

        model_path = self._models_dir / model.filename

        if model_path.exists():
            self._proceed_with_model(model_path)
        else:
            self._download_model(model, model_path)

    def _download_model(self, model, model_path: Path) -> None:
        """Download the model."""
        status = self.query_one("#status-message", Static)
        status.update(Text(f"Downloading {model.name}...", style=f"{CYAN}"))

        progress = self.query_one("#download-progress", ProgressBar)
        progress.display = True

        self.run_worker(self._do_download(model, model_path), exclusive=True)

    async def _do_download(self, model, model_path: Path) -> None:
        """Worker to download model."""
        status = self.query_one("#status-message", Static)
        progress = self.query_one("#download-progress", ProgressBar)

        model_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:
                async with client.stream("GET", model.url) as response:
                    total = model.size_bytes or int(response.headers.get("content-length", 0))
                    downloaded = 0

                    progress.update(total=total, progress=0)

                    with open(model_path, "wb") as f:
                        async for chunk in response.aiter_bytes(1024 * 1024):
                            f.write(chunk)
                            downloaded += len(chunk)
                            progress.update(progress=downloaded)

                            mb_done = downloaded // 1024 // 1024
                            mb_total = total // 1024 // 1024
                            pct = int(downloaded / total * 100) if total > 0 else 0
                            status.update(Text(f"Downloading {model.name}... {pct}% ({mb_done}MB / {mb_total}MB)", style=f"{CYAN}"))

            status.update(Text(f"Downloaded: {model.name}", style=f"bold {GREEN}"))
            progress.display = False
            self._proceed_with_model(model_path)

        except Exception as e:
            status.update(Text(f"Download failed: {e}", style=f"bold {RED}"))
            progress.display = False
            if model_path.exists():
                model_path.unlink()

    def _proceed_with_model(self, model_path: Path) -> None:
        """Proceed to NLI setup with selected model."""
        app: "LoreguardApp" = self.app  # type: ignore
        app.proceed_to_nli_setup(model_path)
