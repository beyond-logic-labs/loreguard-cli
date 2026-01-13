"""Unified palette modal - single search interface for commands, models, and more."""

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Any

import httpx
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static, ListView, ListItem, Label, ProgressBar
from rich.text import Text

from ..styles import PURPLE, CYAN, PINK, FG, FG_DIM, GREEN, YELLOW, RED
from ..widgets.banner import get_gradient_color

if TYPE_CHECKING:
    from ..app import LoreguardApp


class PaletteItem:
    """An item in the unified palette."""

    def __init__(
        self,
        id: str,
        title: str,
        description: str = "",
        category: str = "command",  # "command", "model", "action"
        shortcut: str = "",
        icon: str = "",
        data: Any = None,
    ) -> None:
        self.id = id
        self.title = title
        self.description = description
        self.category = category
        self.shortcut = shortcut
        self.icon = icon
        self.data = data  # Extra data (e.g., model info)


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
    }}

    UnifiedPaletteModal > Vertical {{
        width: 65;
        height: auto;
        max-height: 24;
        border: solid {FG_DIM};
        background: #282A36;
        padding: 0;
    }}

    UnifiedPaletteModal .modal-header {{
        height: 1;
        padding: 0 1;
        background: transparent;
    }}

    UnifiedPaletteModal Input {{
        border: tall #282A36;
        background: #282A36;
        color: {FG};
        padding: 0 1;
    }}

    UnifiedPaletteModal Input:focus {{
        border: tall #282A36;
    }}

    UnifiedPaletteModal ListView {{
        height: auto;
        max-height: 14;
        padding: 0;
        background: transparent;
    }}

    UnifiedPaletteModal ListView > ListItem {{
        padding: 0 1;
        height: 1;
        background: transparent;
    }}

    UnifiedPaletteModal ListView > ListItem.-selected {{
        background: {PURPLE};
        color: #F8F8F2;
    }}

    UnifiedPaletteModal .modal-footer {{
        height: auto;
        padding: 0 1;
        color: {FG_DIM};
    }}

    UnifiedPaletteModal .status-line {{
        padding: 0 1;
        height: auto;
    }}

    UnifiedPaletteModal ProgressBar {{
        padding: 0 1;
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

    def compose(self) -> ComposeResult:
        """Compose the modal."""
        with Vertical():
            yield Static(self._render_header(), classes="modal-header")
            yield Input(placeholder="Type to filter", id="palette-filter")
            yield ListView(id="palette-list")
            yield Static("", id="status-line", classes="status-line")
            yield ProgressBar(id="download-progress", show_percentage=True)
            yield Static(self._render_footer(), classes="modal-footer")

    def _render_header(self) -> Text:
        """Render header with title and gradient line."""
        text = Text()
        text.append(f"{self._title} ", style=f"bold {PINK}")
        # Add gradient hashes
        remaining = 60 - len(self._title) - 1
        for i in range(remaining):
            color = get_gradient_color(i / remaining)
            text.append("#", style=color)
        return text

    def _render_footer(self) -> Text:
        """Render footer with key hints."""
        text = Text()
        text.append("↑↓", style=f"bold {FG}")
        text.append(" choose ", style=FG_DIM)
        text.append("  ")
        text.append("enter", style=f"bold {FG}")
        text.append(" confirm ", style=FG_DIM)
        text.append("  ")
        text.append("esc", style=f"bold {FG}")
        text.append(" cancel", style=FG_DIM)
        return text

    def on_mount(self) -> None:
        """Load items on mount."""
        self._load_all_items()
        self._populate_list()

        # Hide progress bar initially
        self.query_one("#download-progress", ProgressBar).display = False

        # Focus the input field
        self.query_one("#palette-filter", Input).focus()

    def _load_all_items(self) -> None:
        """Load all available items (commands + models)."""
        self._all_items = []

        # Add custom items first
        self._all_items.extend(self._custom_items)

        # Add commands
        if self._show_commands:
            self._all_items.extend([
                PaletteItem("cmd-chat", "Chat with NPC", "", "command", "ctrl+n", ""),
                PaletteItem("cmd-monitor", "Server Monitor", "", "command", "ctrl+m", ""),
                PaletteItem("cmd-switch-model", "Switch Model", "", "command", "ctrl+l", ""),
                PaletteItem("cmd-change-token", "Change Token", "", "command", "", ""),
                PaletteItem("cmd-restart", "Restart Server", "", "command", "", ""),
                PaletteItem("cmd-logout", "Logout", "", "command", "", ""),
                PaletteItem("cmd-theme", "Change Theme", "", "command", "ctrl+t", ""),
                PaletteItem("cmd-config", "Show Config Path", "", "command", "", ""),
                PaletteItem("cmd-keys", "Keys", "", "command", "", ""),
                PaletteItem("cmd-screenshot", "Screenshot", "", "command", "", ""),
                PaletteItem("cmd-quit", "Quit", "", "command", "ctrl+c", ""),
            ])

        # Add models
        if self._show_models:
            from ...llama_server import get_models_dir
            from ...models_registry import SUPPORTED_MODELS

            self._models_dir = get_models_dir()

            for model in SUPPORTED_MODELS:
                is_installed = self._models_dir and (self._models_dir / model.filename).exists()

                tags = []
                if is_installed:
                    tags.append("installed")
                else:
                    tags.append(f"{model.size_gb:.1f}GB")
                if model.recommended:
                    tags.append("recommended")

                icon = "✓" if is_installed else ""

                self._all_items.append(PaletteItem(
                    id=f"model-{model.id}",
                    title=model.name,
                    description=" • ".join(tags),
                    category="model",
                    icon=icon,
                    data=model,
                ))

        # Add adapters (LoRA)
        if self._show_adapters:
            from ...llama_server import get_models_dir
            from ...models_registry import SUPPORTED_ADAPTERS

            self._models_dir = get_models_dir()

            # Add "No adapter" option first
            self._all_items.append(PaletteItem(
                id="adapter-none",
                title="No Adapter",
                description="skip",
                category="adapter",
                icon="",
                data=None,
            ))

            for adapter in SUPPORTED_ADAPTERS:
                is_installed = self._models_dir and (self._models_dir / adapter.filename).exists()

                tags = []
                if is_installed:
                    tags.append("installed")
                else:
                    tags.append(f"{adapter.size_mb:.0f}MB")
                if adapter.recommended:
                    tags.append("recommended")

                icon = "✓" if is_installed else ""

                self._all_items.append(PaletteItem(
                    id=f"adapter-{adapter.id}",
                    title=adapter.name,
                    description=" • ".join(tags),
                    category="adapter",
                    icon=icon,
                    data=adapter,
                ))

    def _populate_list(self, filter_text: str = "") -> None:
        """Populate the list with filtered items."""
        palette_list = self.query_one("#palette-list", ListView)
        palette_list.clear()

        filter_lower = filter_text.lower()
        self._filtered_items = []

        for item in self._all_items:
            # Filter
            if filter_lower:
                if (filter_lower not in item.title.lower() and
                    filter_lower not in item.description.lower() and
                    filter_lower not in item.category.lower()):
                    continue

            self._filtered_items.append(item)

            # Build display text - fixed width of 60 chars
            line_width = 60

            # Icon
            icon_str = ""
            if item.icon:
                icon_str = f"{item.icon} "

            # Title
            title_str = item.title

            # Description/tags
            desc_str = ""
            if item.description:
                desc_str = f"  [{item.description}]"

            # Calculate left side length
            left_side = icon_str + title_str + desc_str
            left_len = len(left_side)

            # Shortcut (right-aligned)
            shortcut_str = item.shortcut if item.shortcut else ""
            shortcut_len = len(shortcut_str)

            # Calculate padding
            padding_needed = line_width - left_len - shortcut_len
            if padding_needed < 1:
                padding_needed = 1

            # Build the text
            text = Text()
            if icon_str:
                if item.icon == "✓":
                    text.append(icon_str, style=f"bold {GREEN}")
                else:
                    text.append(icon_str)
            text.append(title_str, style=f"bold {FG}")
            if desc_str:
                text.append(desc_str, style=FG_DIM)
            text.append(" " * padding_needed)
            if shortcut_str:
                text.append(shortcut_str, style="#888888")

            list_item = ListItem(Label(text))
            list_item.palette_item = item  # type: ignore
            palette_list.append(list_item)

        # Highlight first item
        if self._filtered_items:
            palette_list.index = 0

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle filter input."""
        if event.input.id == "palette-filter" and not self._downloading:
            self._populate_list(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle enter key on input - select current item."""
        if event.input.id == "palette-filter":
            self.action_select()

    def on_key(self, event) -> None:
        """Handle key events - capture arrow keys even when input is focused."""
        if self._downloading:
            return

        if event.key == "up":
            self.query_one("#palette-list", ListView).action_cursor_up()
            event.prevent_default()
            event.stop()
        elif event.key == "down":
            self.query_one("#palette-list", ListView).action_cursor_down()
            event.prevent_default()
            event.stop()
        elif event.key == "escape":
            self.dismiss(None)
            event.prevent_default()
            event.stop()

    def action_cursor_up(self) -> None:
        """Move cursor up."""
        if not self._downloading:
            self.query_one("#palette-list", ListView).action_cursor_up()

    def action_cursor_down(self) -> None:
        """Move cursor down."""
        if not self._downloading:
            self.query_one("#palette-list", ListView).action_cursor_down()

    def action_select(self) -> None:
        """Handle selection."""
        if self._downloading:
            return

        palette_list = self.query_one("#palette-list", ListView)
        selected = palette_list.highlighted_child
        if selected and hasattr(selected, "palette_item"):
            self._handle_selection(selected.palette_item)  # type: ignore

    def action_cancel(self) -> None:
        """Cancel."""
        if not self._downloading:
            self.dismiss(None)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle list selection."""
        if self._downloading:
            return
        if hasattr(event.item, "palette_item"):
            self._handle_selection(event.item.palette_item)  # type: ignore

    def _handle_selection(self, item: PaletteItem) -> None:
        """Handle item selection based on category."""
        if item.category == "command":
            # Return the command ID
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
                # "No adapter" selected
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
            # Return the theme ID
            self.dismiss(("theme", item.id))

        elif item.category == "npc":
            # Return the NPC ID
            self.dismiss(("npc", item.id))

        else:
            # Generic action
            self.dismiss(("action", item.id))

    def _download_model(self, model, model_path: Path) -> None:
        """Download the model."""
        self._downloading = True

        status = self.query_one("#status-line", Static)
        status.update(Text(f"Downloading {model.name}...", style=CYAN))

        progress = self.query_one("#download-progress", ProgressBar)
        progress.display = True

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
                            status.update(Text(f"Downloading {model.name}... {pct}% ({mb}/{total_mb} MB)", style=CYAN))

            status.update(Text(f"Downloaded {model.name}", style=f"bold {GREEN}"))
            progress.display = False
            self._downloading = False

            import asyncio
            await asyncio.sleep(0.5)
            self.dismiss(("model", model_path))

        except Exception as e:
            status.update(Text(f"Download failed: {e}", style=f"bold {RED}"))
            progress.display = False
            self._downloading = False
            if model_path.exists():
                model_path.unlink()

    def _download_adapter(self, adapter, adapter_path: Path) -> None:
        """Download an adapter."""
        self._downloading = True

        status = self.query_one("#status-line", Static)
        status.update(Text(f"Downloading {adapter.name}...", style=CYAN))

        progress = self.query_one("#download-progress", ProgressBar)
        progress.display = True

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
                            status.update(Text(f"Downloading {adapter.name}... {pct}% ({mb}/{total_mb} MB)", style=CYAN))

            status.update(Text(f"Downloaded {adapter.name}", style=f"bold {GREEN}"))
            progress.display = False
            self._downloading = False

            import asyncio
            await asyncio.sleep(0.5)
            self.dismiss(("adapter", adapter_path))

        except Exception as e:
            status.update(Text(f"Download failed: {e}", style=f"bold {RED}"))
            progress.display = False
            self._downloading = False
            if adapter_path.exists():
                adapter_path.unlink()
