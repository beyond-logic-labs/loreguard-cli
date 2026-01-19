"""Custom footer with model status."""

from textual.reactive import reactive
from textual.widgets import Static
from rich.text import Text
from rich.table import Table

from ..styles import BG

# Gray tones for minimal look
KEY_FG = "#AAAAAA"       # Keys (/, ^N, etc.)
LABEL_FG = "#666677"     # Labels (Cmd, Chat, LLM, NLI, etc.)
MODEL_FG = "#888899"     # Model names
VERSION_FG = "#55AA77"   # API version (subtle green)


class LoreguardFooter(Static):
    """Footer showing model status."""

    DEFAULT_CSS = f"""
    LoreguardFooter {{
        height: 5;
        dock: bottom;
        padding: 0 2;
        border-top: solid #333344;
        background: {BG};
    }}
    """

    llm_model: reactive[str] = reactive("", layout=True)
    llm_adapter: reactive[str] = reactive("", layout=True)
    nli_model: reactive[str] = reactive("", layout=True)
    intent_model: reactive[str] = reactive("", layout=True)
    api_version: reactive[str] = reactive("", layout=True)

    def on_mount(self) -> None:
        """Start syncing model labels from the app state."""
        self._sync_with_app()
        self.set_interval(0.5, self._sync_with_app)

    def _sync_with_app(self) -> None:
        app = self.app
        tunnel = getattr(self.screen, "_tunnel", None) or getattr(app, "_tunnel", None)

        # LLM model
        model_path = getattr(app, "model_path", None)
        llm_name = model_path.stem if model_path else ""

        # Adapter
        adapter_path = getattr(app, "adapter_path", None)
        adapter_name = adapter_path.stem if adapter_path else ""

        # NLI model (citation check)
        nli_name = ""
        nli_service = getattr(tunnel, "nli_service", None) if tunnel else None
        if nli_service and getattr(nli_service, "is_loaded", False):
            nli_name = getattr(nli_service, "model_name", "")
            if "/" in nli_name:
                nli_name = nli_name.split("/")[-1]

        # Intent classifier model
        intent_name = ""
        intent_classifier = getattr(tunnel, "intent_classifier", None) if tunnel else None
        if intent_classifier and getattr(intent_classifier, "is_loaded", False):
            intent_name = getattr(intent_classifier, "model_name", "")
            if "/" in intent_name:
                intent_name = intent_name.split("/")[-1]

        # Backend API version
        api_ver = getattr(tunnel, "backend_version", "") if tunnel else ""

        # Update reactives if changed
        if llm_name != self.llm_model:
            self.llm_model = llm_name
        if adapter_name != self.llm_adapter:
            self.llm_adapter = adapter_name
        if nli_name != self.nli_model:
            self.nli_model = nli_name
        if intent_name != self.intent_model:
            self.intent_model = intent_name
        if api_ver != self.api_version:
            self.api_version = api_ver

    def _shortcut(self, key: str, label: str) -> Text:
        """Create a formatted shortcut text."""
        t = Text()
        t.append(key, style=f"bold {KEY_FG}")
        t.append(f" {label}", style=LABEL_FG)
        return t

    def render(self) -> Table:
        """Render footer with shortcuts row and model info row."""
        # Row 1: Shortcuts + API version
        shortcuts = Text()
        shortcuts.append_text(self._shortcut("/", "commands"))
        shortcuts.append("  ")
        shortcuts.append_text(self._shortcut("^N", "chat"))
        shortcuts.append("  ")
        shortcuts.append_text(self._shortcut("^L", "model"))
        shortcuts.append("  ")
        shortcuts.append_text(self._shortcut("^T", "theme"))
        shortcuts.append("  ")
        shortcuts.append_text(self._shortcut("^C", "quit"))

        api_info = Text()
        if self.api_version:
            api_info.append("API ", style=LABEL_FG)
            api_info.append(self.api_version, style=VERSION_FG)

        # Row 2: Model info
        model_info = Text()

        if self.llm_model:
            model_info.append("LLM ", style=LABEL_FG)
            model_info.append(self.llm_model, style=MODEL_FG)
            if self.llm_adapter:
                model_info.append("+", style=LABEL_FG)
                model_info.append(self.llm_adapter, style=MODEL_FG)

        if self.nli_model:
            if model_info:
                model_info.append("  ")
            model_info.append("NLI ", style=LABEL_FG)
            model_info.append(self.nli_model, style=MODEL_FG)

        if self.intent_model:
            if model_info:
                model_info.append("  ")
            model_info.append("Intent ", style=LABEL_FG)
            model_info.append(self.intent_model, style=MODEL_FG)

        table = Table.grid(expand=True)
        table.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
        table.add_column(justify="right", no_wrap=True)
        table.add_row(shortcuts, api_info)
        table.add_row("", "")
        table.add_row(model_info, "")
        return table
