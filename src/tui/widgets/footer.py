"""Custom footer with model status."""

from textual.reactive import reactive
from textual.widgets import Footer
from rich.protocol import is_renderable
from rich.table import Table
from rich.text import Text

from ..styles import FG, FG_DIM, CYAN


class LoreguardFooter(Footer):
    """Footer showing key bindings plus model status."""

    llm_model: reactive[str] = reactive("", layout=True)
    nli_model: reactive[str] = reactive("", layout=True)
    intent_model: reactive[str] = reactive("", layout=True)
    backend_version: reactive[str] = reactive("", layout=True)

    def on_mount(self) -> None:
        """Start syncing model labels from the app state."""
        self._sync_with_app()
        self.set_interval(0.5, self._sync_with_app)

    def _sync_with_app(self) -> None:
        app = self.app
        tunnel = getattr(self.screen, "_tunnel", None) or getattr(app, "_tunnel", None)

        # LLM model
        model_path = getattr(app, "model_path", None)
        llm_name = model_path.name if model_path else ""

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

        # Backend version (from tunnel if available)
        backend_ver = getattr(tunnel, "backend_version", "") if tunnel else ""

        # Update reactives if changed
        if llm_name != self.llm_model:
            self.llm_model = llm_name
        if nli_name != self.nli_model:
            self.nli_model = nli_name
        if intent_name != self.intent_model:
            self.intent_model = intent_name
        if backend_ver != self.backend_version:
            self.backend_version = backend_ver

    def render(self):
        key_text = super().render()
        if not is_renderable(key_text):
            key_text = Text("")

        info_text = Text()

        # Show loaded models
        models_shown = 0

        if self.llm_model:
            info_text.append("LLM:", style=FG_DIM)
            info_text.append(self.llm_model, style=FG)
            models_shown += 1

        if self.nli_model:
            if models_shown > 0:
                info_text.append(" ", style=FG_DIM)
            info_text.append("NLI:", style=FG_DIM)
            info_text.append(self.nli_model, style=FG)
            models_shown += 1

        if self.intent_model:
            if models_shown > 0:
                info_text.append(" ", style=FG_DIM)
            info_text.append("Intent:", style=FG_DIM)
            info_text.append(self.intent_model, style=FG)
            models_shown += 1

        if self.backend_version:
            if models_shown > 0:
                info_text.append(" ", style=FG_DIM)
            info_text.append("API:", style=FG_DIM)
            info_text.append(self.backend_version, style=CYAN)

        table = Table.grid(expand=True)
        table.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
        table.add_column(justify="right", no_wrap=True, overflow="ellipsis")
        table.add_row(key_text, info_text)
        return table
