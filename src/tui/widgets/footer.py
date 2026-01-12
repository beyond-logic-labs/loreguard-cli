"""Custom footer with model status."""

from textual.reactive import reactive
from textual.widgets import Static
from rich.text import Text
from rich.table import Table

from ..styles import FG, FG_DIM, CYAN, GREEN


class LoreguardFooter(Static):
    """Footer showing model status."""

    DEFAULT_CSS = """
    LoreguardFooter {
        height: 2;
        dock: bottom;
        padding: 0 1;
        margin-top: 1;
        background: $surface;
    }
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
        llm_name = model_path.stem if model_path else ""  # Use stem (no extension)

        # Adapter
        adapter_path = getattr(app, "adapter_path", None)
        adapter_name = adapter_path.stem if adapter_path else ""

        # NLI model (citation check)
        nli_name = ""
        nli_service = getattr(tunnel, "nli_service", None) if tunnel else None
        if nli_service and getattr(nli_service, "is_loaded", False):
            nli_name = getattr(nli_service, "model_name", "")

        # Intent classifier model
        intent_name = ""
        intent_classifier = getattr(tunnel, "intent_classifier", None) if tunnel else None
        if intent_classifier and getattr(intent_classifier, "is_loaded", False):
            intent_name = getattr(intent_classifier, "model_name", "")
            # Shorten model name for display
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

    def render(self) -> Table:
        """Render footer with LLM + NLI/Intent on left, server version on right."""
        # Left side: LLM + adapter + NLI + Intent
        left = Text()
        if self.llm_model:
            left.append("LLM:", style=FG_DIM)
            left.append(self.llm_model, style=FG)
            if self.llm_adapter:
                left.append(" + ", style=FG_DIM)
                left.append(self.llm_adapter, style=GREEN)

        if self.nli_model:
            if left:
                left.append(" ", style=FG_DIM)
            left.append("NLI:", style=FG_DIM)
            left.append(self.nli_model, style=FG)

        if self.intent_model:
            if left:
                left.append(" ", style=FG_DIM)
            left.append("Intent:", style=FG_DIM)
            left.append(self.intent_model, style=FG)

        # Right side: Server version
        right = Text()
        if self.api_version:
            right.append("Server Version: ", style=FG_DIM)
            right.append(self.api_version, style=CYAN)

        table = Table.grid(expand=True)
        table.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
        table.add_column(justify="right", no_wrap=True)
        table.add_row(left, right)
        return table
