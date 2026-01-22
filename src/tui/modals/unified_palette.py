"""Unified palette modal - single search interface for commands, models, and more."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Input, Static, ListView, ListItem, Label, ProgressBar
from rich.text import Text

from ..styles import GREEN, RED

if TYPE_CHECKING:
    from ..app import LoreguardApp

# Crush-style colors
MAGENTA = "#FF79C6"       # Bright magenta for headers
CYAN = "#50FA7B"          # Cyan/green for prompt
ITEM_FG = "#FFFFFF"       # White for items
DESC_FG = "#9090A0"       # Lighter gray for descriptions/shortcuts
BORDER_COLOR = "#6B5B95"  # Subtle purple border
SELECTION_BG = "#7C3AED"  # Vivid purple for selection
BG_COLOR = "#1E1F29"
INPUT_BG = "#1E1F29"


class PaletteItem:
    """An item in the unified palette."""

    def __init__(
        self,
        id: str,
        title: str,
        description: str = "",
        category: str = "command",
        shortcut: str = "",
        icon: str = "",
        data: Any = None,
        is_section: bool = False,
    ) -> None:
        self.id = id
        self.title = title
        self.description = description
        self.category = category
        self.shortcut = shortcut
        self.icon = icon
        self.data = data
        self.is_section = is_section


class UnifiedPaletteModal(ModalScreen[tuple[str, Any] | None]):
    """Unified floating palette for commands, models, and actions."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("enter", "select", "Select", show=False),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("j", "cursor_down", "Down", show=False),
    ]

    DEFAULT_CSS = f"""
    UnifiedPaletteModal {{
        align: center middle;
        background: transparent;
        padding-bottom: 5;
    }}

    UnifiedPaletteModal > Vertical {{
        width: 70;
        height: auto;
        max-height: 22;
        border: solid {BORDER_COLOR};
        background: {BG_COLOR};
        padding: 0;
    }}

    UnifiedPaletteModal .modal-header {{
        height: 1;
        padding: 0 1;
        margin-bottom: 1;
    }}

    UnifiedPaletteModal .input-row {{
        height: 2;
        padding: 0 1;
        margin: 0 0 0 0;
        width: 100%;
        background: transparent;
    }}

    UnifiedPaletteModal .input-prompt {{
        width: 2;
        height: 1;
        padding: 0;
        background: transparent;
    }}

    #palette-filter {{
        border: hidden;
        background: {BG_COLOR};
        color: {ITEM_FG};
        padding: 0;
        margin: 0;
        width: 1fr;
        height: 1;
    }}

    #palette-filter:focus {{
        border: hidden;
        background: {BG_COLOR};
    }}

    UnifiedPaletteModal ListView {{
        height: auto;
        max-height: 9;
        padding: 0;
        background: transparent;
        scrollbar-size: 1 1;
    }}

    UnifiedPaletteModal ListView > ListItem {{
        padding: 0 1;
        height: 1;
        background: transparent;
    }}

    UnifiedPaletteModal ListView > ListItem.-highlight {{
        background: {SELECTION_BG};
    }}

    UnifiedPaletteModal ListView > ListItem.--highlight {{
        background: {SELECTION_BG};
    }}

    #palette-list > ListItem.-highlight {{
        background: {SELECTION_BG};
    }}

    #palette-list > ListItem.--highlight {{
        background: {SELECTION_BG};
    }}

    UnifiedPaletteModal .modal-footer {{
        height: 2;
        min-height: 2;
        padding: 0 1;
        text-align: left;
        content-align: left middle;
    }}

    UnifiedPaletteModal .status-line {{
        padding: 0 1;
        height: 0;
        display: none;
    }}

    UnifiedPaletteModal .status-line.visible {{
        height: auto;
        display: block;
    }}

    UnifiedPaletteModal ProgressBar {{
        padding: 0 1;
        display: none;
    }}

    UnifiedPaletteModal ProgressBar.visible {{
        display: block;
    }}
    """

    def __init__(
        self,
        title: str = "Search",
        items: list[PaletteItem] | None = None,
        show_models: bool = True,
        show_commands: bool = True,
        show_adapters: bool = False,
    ) -> None:
        super().__init__()
        self._title = title
        self._custom_items = items or []
        self._show_models = show_models
        self._show_commands = show_commands
        self._show_adapters = show_adapters
        self._all_items: list[PaletteItem] = []
        self._filtered_items: list[PaletteItem] = []
        self._models_dir: Path | None = None
        self._downloading = False
        self._download_path: Path | None = None  # Track current download for cleanup

    def compose(self) -> ComposeResult:
        """Compose the modal."""
        with Vertical():
            yield Static(self._render_header(), classes="modal-header")
            with Horizontal(classes="input-row"):
                yield Static(self._render_input_prompt(), classes="input-prompt")
                yield Input(placeholder="Type to filter", id="palette-filter")
            yield ListView(id="palette-list")
            yield Static("", id="status-line", classes="status-line")
            yield ProgressBar(id="download-progress", show_percentage=True)
            yield Static(self._render_footer(), classes="modal-footer")

    def _render_header(self) -> Text:
        """Render header with title."""
        text = Text()
        text.append(f"{self._title}", style=f"bold {SELECTION_BG}")
        return text

    def _render_input_prompt(self) -> Text:
        """Render the input prompt."""
        text = Text()
        text.append(">", style=f"bold {CYAN}")
        return text

    def _render_footer(self) -> Text:
        """Render footer with key hints."""
        text = Text()
        text.append("↑↓", style=DESC_FG)
        text.append(" choose", style=DESC_FG)
        text.append("  · ", style=DESC_FG)
        text.append("enter", style=DESC_FG)
        text.append(" confirm", style=DESC_FG)
        text.append("  · ", style=DESC_FG)
        text.append("esc", style=DESC_FG)
        text.append(" cancel", style=DESC_FG)
        return text

    def _check_model_fit(self, model_size_gb: float, hardware) -> str:
        """Check if model fits in available VRAM.

        Returns: 'fits', 'tight', 'too_big', or 'unknown'
        """
        if not hardware:
            return "unknown"

        # Use VRAM if available, otherwise RAM (unified memory on Apple Silicon)
        available = hardware.gpu_vram_gb if hardware.gpu_vram_gb else hardware.ram_gb
        if not available:
            return "unknown"

        # Minimum required = model size + 1GB overhead
        min_required = model_size_gb + 1.0

        if available >= min_required + 4:
            return "fits"  # Comfortable (4GB+ headroom)
        elif available >= min_required:
            return "tight"  # Will work but tight
        else:
            return "too_big"  # Won't fit

    def on_mount(self) -> None:
        """Load items on mount."""
        self._load_all_items()
        self._populate_list()
        self.query_one("#palette-filter", Input).focus()

    def _load_all_items(self) -> None:
        """Load all available items (commands + models)."""
        self._all_items = []

        # Add custom items first
        self._all_items.extend(self._custom_items)

        # Add commands
        if self._show_commands:
            self._all_items.extend([
                PaletteItem("cmd-chat", "Chat with NPC", "", "command", "ctrl+n"),
                PaletteItem("cmd-scenario", "Switch Scenario", "", "command", "ctrl+e"),
                PaletteItem("cmd-switch-model", "Switch Model", "", "command", "ctrl+l"),
                PaletteItem("cmd-theme", "Change Theme", "", "command", "ctrl+t"),
                PaletteItem("cmd-change-token", "Change Token", "", "command"),
                PaletteItem("cmd-restart", "Restart Server", "", "command"),
                PaletteItem("cmd-config", "Show Config Path", "", "command"),
                PaletteItem("cmd-logout", "Logout", "", "command"),
                PaletteItem("cmd-quit", "Quit", "", "command", "ctrl+c"),
            ])

        # Add models
        if self._show_models:
            from ...llama_server import get_models_dir
            from ...models_registry import SUPPORTED_MODELS
            from ...hf_discovery import discover_models
            from ..widgets.hardware_info import detect_hardware

            self._models_dir = get_models_dir()

            # Detect hardware for compatibility checking
            hardware = detect_hardware()

            # Get models from HF discovery (includes dynamic models like pipeline-v3)
            # Fall back to static registry if discovery fails
            try:
                all_models = discover_models(use_cache=True)
                if not all_models:
                    all_models = list(SUPPORTED_MODELS)
            except Exception:
                all_models = list(SUPPORTED_MODELS)

            # Sort: most recent first, then by size descending
            def model_sort_key(m):
                # Primary: sort by recency (days_ago), None goes last
                days = m.days_ago if m.days_ago is not None else 9999
                # Secondary: larger models first (descending)
                return (days, -m.size_gb)

            all_models.sort(key=model_sort_key)

            # Separate installed vs available
            installed = []
            available = []

            for model in all_models:
                # Check if file exists AND is complete (at least 90% of expected size)
                model_path = self._models_dir / model.filename if self._models_dir else None
                if model_path and model_path.exists():
                    actual_size = model_path.stat().st_size
                    expected_size = model.size_bytes or 0
                    # Allow 10% tolerance for size estimation differences
                    is_installed = expected_size == 0 or actual_size >= expected_size * 0.9
                else:
                    is_installed = False

                # Check hardware compatibility
                fit_status = self._check_model_fit(model.size_gb, hardware)

                # Calculate minimum VRAM (model size + 1GB overhead)
                min_vram = int(model.size_gb + 1)

                # Build age string
                age_str = ""
                if model.days_ago is not None:
                    if model.days_ago == 0:
                        age_str = "today"
                    elif model.days_ago == 1:
                        age_str = "1d ago"
                    else:
                        age_str = f"{model.days_ago}d ago"

                # Build description with size and VRAM requirement
                if is_installed:
                    desc = "installed"
                else:
                    desc = f"{model.size_gb:.1f}GB · ≥{min_vram}GB VRAM"

                item = PaletteItem(
                    id=f"model-{model.id}",
                    title=model.name,
                    description=desc,
                    category="model",
                    icon=age_str,  # Store age for gray rendering
                    data=model,
                    shortcut=fit_status,  # Store fit status for coloring
                )

                if is_installed:
                    installed.append(item)
                else:
                    available.append(item)

            if installed:
                self._all_items.append(PaletteItem("section-installed", "Installed Models", is_section=True))
                self._all_items.extend(installed)

            if available:
                self._all_items.append(PaletteItem("section-available", "Available Models", is_section=True))
                self._all_items.extend(available)

        # Add adapters (LoRA)
        if self._show_adapters:
            from ...llama_server import get_models_dir
            from ...models_registry import SUPPORTED_ADAPTERS

            self._models_dir = get_models_dir()

            self._all_items.append(PaletteItem("section-adapters", "Adapters", is_section=True))
            self._all_items.append(PaletteItem(
                id="adapter-none",
                title="No Adapter",
                description="skip",
                category="adapter",
            ))

            for adapter in SUPPORTED_ADAPTERS:
                is_installed = self._models_dir and (self._models_dir / adapter.filename).exists()

                self._all_items.append(PaletteItem(
                    id=f"adapter-{adapter.id}",
                    title=adapter.name,
                    description=f"{adapter.size_mb:.0f}MB" if not is_installed else "installed",
                    category="adapter",
                    icon="✓" if is_installed else "",
                    data=adapter,
                ))

    def _populate_list(self, filter_text: str = "") -> None:
        """Populate the list with filtered items."""
        palette_list = self.query_one("#palette-list", ListView)
        palette_list.clear()

        filter_lower = filter_text.lower()
        self._filtered_items = []

        for item in self._all_items:
            # Skip section headers when filtering
            if filter_lower and item.is_section:
                continue

            # Filter
            if filter_lower:
                if (filter_lower not in item.title.lower() and
                    filter_lower not in item.description.lower() and
                    filter_lower not in item.category.lower()):
                    continue

            self._filtered_items.append(item)

            # Build display text
            line_width = 65

            if item.is_section:
                # Section header - skip empty separators
                if not item.title:
                    continue
                text = Text()
                text.append(item.title, style=DESC_FG)  # Gray for section titles
                list_item = ListItem(Label(text))
                list_item.palette_item = item  # type: ignore
                list_item.disabled = True
                palette_list.append(list_item)
                continue

            # Regular item
            title_str = item.title
            text = Text()

            # For models, show: Name ... Age (dim gray) Size/VRAM (dim colored) right-aligned
            if item.category == "model":
                fit_status = item.shortcut  # We stored fit status here
                age_str = item.icon  # We stored age here
                desc_str = item.description

                # Color based on fit status (with dim for smaller appearance)
                if fit_status == "fits":
                    desc_color = f"dim {GREEN}"
                elif fit_status == "tight":
                    desc_color = "dim #F1FA8C"  # Yellow
                elif fit_status == "too_big":
                    desc_color = f"dim {RED}"
                else:
                    desc_color = f"dim {DESC_FG}"

                # Right-align: Name takes remaining space, then age + desc on right
                total_width = 65
                right_part = f"{age_str}  {desc_str}" if age_str else desc_str
                right_len = len(right_part)

                # Calculate padding to push right part to the end
                padding_needed = total_width - len(title_str) - right_len
                if padding_needed < 1:
                    padding_needed = 1
                    # Truncate name if needed
                    max_name = total_width - right_len - 1
                    if max_name > 0:
                        title_str = title_str[:max_name]

                text.append(title_str, style=ITEM_FG)
                text.append(" " * padding_needed)
                if age_str:
                    text.append(age_str, style=f"dim {DESC_FG}")
                    text.append("  ")
                text.append(desc_str, style=desc_color)
            else:
                # Commands and other items
                shortcut_str = item.shortcut or ""
                padding_needed = line_width - len(title_str) - len(shortcut_str)
                if padding_needed < 1:
                    padding_needed = 1

                text.append(title_str, style=ITEM_FG)
                text.append(" " * padding_needed)
                if shortcut_str:
                    text.append(shortcut_str, style=DESC_FG)

            list_item = ListItem(Label(text))
            list_item.palette_item = item  # type: ignore
            palette_list.append(list_item)

        # Highlight first non-section item
        for i, item in enumerate(self._filtered_items):
            if not item.is_section:
                palette_list.index = i
                break

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle filter input."""
        if event.input.id == "palette-filter" and not self._downloading:
            self._populate_list(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle enter key on input."""
        if event.input.id == "palette-filter":
            self.action_select()

    def on_key(self, event) -> None:
        """Handle key events."""
        if self._downloading:
            return

        if event.key == "up":
            self._move_cursor(-1)
            event.prevent_default()
            event.stop()
        elif event.key == "down":
            self._move_cursor(1)
            event.prevent_default()
            event.stop()
        elif event.key == "escape":
            self.dismiss(None)
            event.prevent_default()
            event.stop()

    def _move_cursor(self, direction: int) -> None:
        """Move cursor, skipping section headers."""
        palette_list = self.query_one("#palette-list", ListView)
        current = palette_list.index or 0
        new_index = current + direction

        while 0 <= new_index < len(self._filtered_items):
            if not self._filtered_items[new_index].is_section:
                palette_list.index = new_index
                return
            new_index += direction

    def action_cursor_up(self) -> None:
        """Move cursor up."""
        if not self._downloading:
            self._move_cursor(-1)

    def action_cursor_down(self) -> None:
        """Move cursor down."""
        if not self._downloading:
            self._move_cursor(1)

    def action_select(self) -> None:
        """Handle selection."""
        if self._downloading:
            return

        palette_list = self.query_one("#palette-list", ListView)
        selected = palette_list.highlighted_child
        if selected and hasattr(selected, "palette_item"):
            item = selected.palette_item  # type: ignore
            if not item.is_section:
                self._handle_selection(item)

    def action_cancel(self) -> None:
        """Cancel."""
        if self._downloading:
            # Clean up partial download
            if self._download_path and self._download_path.exists():
                try:
                    self._download_path.unlink()
                except Exception:
                    pass
            self._downloading = False
            self._download_path = None
        self.dismiss(None)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle list selection."""
        if self._downloading:
            return
        if hasattr(event.item, "palette_item"):
            item = event.item.palette_item  # type: ignore
            if not item.is_section:
                self._handle_selection(item)

    def _handle_selection(self, item: PaletteItem) -> None:
        """Handle item selection based on category."""
        if item.category == "command":
            self.dismiss(("command", item.id))

        elif item.category == "model":
            model = item.data
            if not model or not self._models_dir:
                return

            model_path = self._models_dir / model.filename

            if model_path.exists():
                self.dismiss(("model", model_path))
            else:
                self._download_model(model, model_path)

        elif item.category == "adapter":
            adapter = item.data
            if adapter is None:
                self.dismiss(("adapter", None))
            elif not self._models_dir:
                return
            else:
                adapter_path = self._models_dir / adapter.filename
                if adapter_path.exists():
                    self.dismiss(("adapter", adapter_path))
                else:
                    self._download_adapter(adapter, adapter_path)

        elif item.category == "theme":
            self.dismiss(("theme", item.id))

        elif item.category == "npc":
            self.dismiss(("npc", item.id))

        elif item.category == "scenario":
            self.dismiss(("scenario", item.data))

        else:
            self.dismiss(("action", item.id))

    def _download_model(self, model, model_path: Path) -> None:
        """Download the model."""
        self._downloading = True
        self._download_path = model_path  # Track for cleanup on cancel

        status = self.query_one("#status-line", Static)
        status.update(Text(f"Downloading {model.name}...", style=ITEM_FG))

        progress = self.query_one("#download-progress", ProgressBar)
        progress.add_class("visible")

        self.run_worker(self._do_download(model, model_path))

    async def _do_download(self, model, model_path: Path) -> None:
        """Worker to download model."""
        status = self.query_one("#status-line", Static)
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

                            pct = int(downloaded / total * 100) if total > 0 else 0
                            mb = downloaded // 1024 // 1024
                            total_mb = total // 1024 // 1024
                            status.update(Text(f"Downloading {model.name}... {pct}% ({mb}/{total_mb} MB)", style=ITEM_FG))

            status.update(Text(f"Downloaded {model.name}", style=GREEN))
            progress.remove_class("visible")
            self._downloading = False
            self._download_path = None

            import asyncio
            await asyncio.sleep(0.5)
            self.dismiss(("model", model_path))

        except Exception as e:
            status.update(Text(f"Download failed: {e}", style=RED))
            progress.remove_class("visible")
            self._downloading = False
            self._download_path = None
            if model_path.exists():
                model_path.unlink()

    def _download_adapter(self, adapter, adapter_path: Path) -> None:
        """Download an adapter."""
        self._downloading = True
        self._download_path = adapter_path  # Track for cleanup on cancel

        status = self.query_one("#status-line", Static)
        status.update(Text(f"Downloading {adapter.name}...", style=ITEM_FG))

        progress = self.query_one("#download-progress", ProgressBar)
        progress.add_class("visible")

        self.run_worker(self._do_download_adapter(adapter, adapter_path))

    async def _do_download_adapter(self, adapter, adapter_path: Path) -> None:
        """Worker to download adapter."""
        status = self.query_one("#status-line", Static)
        progress = self.query_one("#download-progress", ProgressBar)

        adapter_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:
                async with client.stream("GET", adapter.url) as response:
                    total = adapter.size_bytes or int(response.headers.get("content-length", 0))
                    downloaded = 0

                    progress.update(total=total, progress=0)

                    with open(adapter_path, "wb") as f:
                        async for chunk in response.aiter_bytes(1024 * 1024):
                            f.write(chunk)
                            downloaded += len(chunk)
                            progress.update(progress=downloaded)

                            pct = int(downloaded / total * 100) if total > 0 else 0
                            mb = downloaded // 1024 // 1024
                            total_mb = total // 1024 // 1024
                            status.update(Text(f"Downloading {adapter.name}... {pct}% ({mb}/{total_mb} MB)", style=ITEM_FG))

            status.update(Text(f"Downloaded {adapter.name}", style=GREEN))
            progress.remove_class("visible")
            self._downloading = False
            self._download_path = None

            import asyncio
            await asyncio.sleep(0.5)
            self.dismiss(("adapter", adapter_path))

        except Exception as e:
            status.update(Text(f"Download failed: {e}", style=RED))
            progress.remove_class("visible")
            self._downloading = False
            self._download_path = None
            if adapter_path.exists():
                adapter_path.unlink()
