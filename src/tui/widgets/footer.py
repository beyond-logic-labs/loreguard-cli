"""Custom footer with model status."""

from textual.reactive import reactive
from textual.widgets import Footer
from rich.protocol import is_renderable
from rich.table import Table
from rich.text import Text

from ..styles import FG, FG_DIM


class LoreguardFooter(Footer):
    """Footer showing key bindings plus model status."""

    llm_model: reactive[str] = reactive("none", layout=True)
    nli_model: reactive[str] = reactive("disabled", layout=True)

    def on_mount(self) -> None:
        """Start syncing model labels from the app state."""
        self._sync_with_app()
        self.set_interval(0.5, self._sync_with_app)

    def _sync_with_app(self) -> None:
        app = self.app

        model_path = getattr(app, "model_path", None)
        llm_name = model_path.name if model_path else "none"

        nli_name = "disabled"
        tunnel = getattr(self.screen, "_tunnel", None) or getattr(app, "_tunnel", None)
        nli_service = getattr(tunnel, "nli_service", None) if tunnel else None
        if nli_service:
            nli_name = getattr(nli_service, "model_name", nli_name)

        if llm_name != self.llm_model or nli_name != self.nli_model:
            self.llm_model = llm_name
            self.nli_model = nli_name

    def render(self):
        key_text = super().render()
        if not is_renderable(key_text):
            key_text = Text("")

        info_text = Text()
        info_text.append("LLM: ", style=FG_DIM)
        info_text.append(self.llm_model, style=FG)
        info_text.append("  ", style=FG_DIM)
        info_text.append("NLI: ", style=FG_DIM)
        info_text.append(self.nli_model, style=FG)

        table = Table.grid(expand=True)
        table.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
        table.add_column(justify="right", no_wrap=True, overflow="ellipsis")
        table.add_row(key_text, info_text)
        return table
