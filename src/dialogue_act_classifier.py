"""Dialogue Act Classification service for filler selection.

Uses DialogTag (DistilBERT) to classify user messages into dialogue act categories
such as Wh-Question, Yes-No-Question, Action-Directive, Statement, etc.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# Default model base used by DialogTag (can be overridden)
DEFAULT_DIALOGUE_ACT_MODEL = "distilbert-base-uncased"


# SWBD-DAMSL shorthand → canonical label mapping (best-effort)
_CODE_TO_LABEL = {
    "qw": "Wh-Question",
    "qy": "Yes-No-Question",
    "qyd": "Yes-No-Question",
    "qyr": "Yes-No-Question",
    "qy^d": "Yes-No-Question",
    "qo": "Open-Question",
    "qr": "Yes-No-Question",
    "ad": "Action-Directive",
    "sd": "Statement-Non-Opinion",
    "sv": "Statement-Opinion",
    "aa": "Acknowledge",
    "b": "Acknowledge",
    "bk": "Acknowledge",
    "bh": "Acknowledge",
}


@dataclass
class DialogueActResult:
    """Result of dialogue act classification."""
    dialogue_act: str
    confidence: float  # 0-1 confidence score
    latency_ms: int    # Classification latency in milliseconds


class DialogueActClassifier:
    """Service for dialogue act classification using DialogTag."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        """Initialize the dialogue act classifier.

        Args:
            model_name: Optional model name for DialogTag. If None, uses default.
        """
        self._model = None
        self._model_name = model_name or DEFAULT_DIALOGUE_ACT_MODEL
        self._device = None
        self._load_lock = threading.Lock()

    @property
    def model_name(self) -> str:
        """Return the configured model identifier."""
        return self._model_name

    @property
    def device(self) -> Optional[str]:
        """Get the device being used (if known)."""
        return self._device

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None

    def _resolve_device(self) -> str:
        """Resolve the best available device."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        except ImportError:
            return "cpu"

    def load_model(self) -> bool:
        """Load the dialogue act model.

        Thread-safe: uses lock to prevent concurrent model loading.
        Returns True if model loaded successfully, False otherwise.
        """
        if self._model is not None:
            return True

        with self._load_lock:
            if self._model is not None:
                return True

            try:
                from dialog_tag import DialogTag

                self._device = self._resolve_device()
                logger.info(
                    "Loading dialogue act classifier: %s (device=%s)",
                    self._model_name,
                    self._device,
                )

                # DialogTag API may vary by version; try a few signatures.
                try:
                    self._model = DialogTag(self._model_name, device=self._device)
                except TypeError:
                    try:
                        self._model = DialogTag(self._model_name)
                    except TypeError:
                        self._model = DialogTag()

                logger.info("Dialogue act classifier loaded successfully")
                return True

            except Exception as e:
                logger.error("Failed to load dialogue act classifier: %s", e)
                return False

    def classify(self, query: str) -> DialogueActResult:
        """Classify the dialogue act of a user query.

        Args:
            query: The user's message to classify

        Returns:
            DialogueActResult with act label and confidence
        """
        import time

        if self._model is None:
            if not self.load_model():
                raise RuntimeError("Dialogue act classifier not loaded")

        start_time = time.time()
        tag, confidence = self._predict(query)
        latency_ms = int((time.time() - start_time) * 1000)

        normalized = normalize_dialogue_act(tag)
        if normalized:
            tag = normalized

        logger.info(
            "Dialogue act classification: %s (confidence=%.2f, latency=%dms)",
            tag,
            confidence,
            latency_ms,
        )

        return DialogueActResult(
            dialogue_act=tag,
            confidence=confidence,
            latency_ms=latency_ms,
        )

    def _predict(self, query: str) -> Tuple[str, float]:
        """Run the underlying model prediction with best-effort confidence."""
        if self._model is None:
            raise RuntimeError("Dialogue act classifier not loaded")

        # Primary API: predict_tag
        if hasattr(self._model, "predict_tag"):
            result = self._model.predict_tag(query)
        elif hasattr(self._model, "predict"):
            result = self._model.predict(query)
        else:
            raise RuntimeError("DialogTag model does not support prediction API")

        # Parse result (string / tuple / dict)
        tag = ""
        confidence = 0.0

        if isinstance(result, tuple) and len(result) >= 1:
            tag = str(result[0])
            if len(result) >= 2 and isinstance(result[1], (float, int)):
                confidence = float(result[1])
        elif isinstance(result, dict):
            tag = str(result.get("tag") or result.get("label") or result.get("dialogue_act") or "")
            score = result.get("confidence") or result.get("score")
            if isinstance(score, (float, int)):
                confidence = float(score)
        else:
            tag = str(result)

        # Fallback: try predict_proba if available
        if confidence == 0.0 and hasattr(self._model, "predict_proba"):
            try:
                probs = self._model.predict_proba(query)
                if isinstance(probs, dict):
                    confidence = float(probs.get(tag, 0.0))
            except Exception:
                pass

        return tag, confidence


def normalize_dialogue_act(tag: str) -> str:
    """Normalize a dialogue act label to a canonical form.

    Handles SWBD-DAMSL shorthand tags and common string variants.
    """
    if not tag:
        return ""

    raw = tag.strip()
    if not raw:
        return ""

    lower = raw.lower()
    if lower in _CODE_TO_LABEL:
        return _CODE_TO_LABEL[lower]

    # Normalize common phrase variants
    if "wh" in lower and "question" in lower:
        return "Wh-Question"
    if "yes" in lower and "question" in lower:
        return "Yes-No-Question"
    if "action" in lower and ("directive" in lower or "command" in lower):
        return "Action-Directive"
    if "open" in lower and "question" in lower:
        return "Open-Question"
    if "statement" in lower and "opinion" in lower:
        return "Statement-Opinion"
    if "statement" in lower:
        return "Statement-Non-Opinion"
    if "ack" in lower or "backchannel" in lower:
        return "Acknowledge"

    return raw


def is_dialogue_act_model_available() -> bool:
    """Check if the dialogue act model is available in HuggingFace cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        cached_path = try_to_load_from_cache(DEFAULT_DIALOGUE_ACT_MODEL, "config.json")
        return cached_path is not None
    except ImportError:
        return False
    except Exception:
        return False


def download_dialogue_act_model(progress_callback=None, error_callback=None) -> bool:
    """Pre-download the dialogue act model to HuggingFace cache.

    Args:
        progress_callback: Optional callback(downloaded_mb, total_mb, filename) for progress updates.
        error_callback: Optional callback(error_message) to report errors.

    Returns:
        True if download successful, False otherwise.
    """
    try:
        from huggingface_hub import snapshot_download
        from tqdm import tqdm

        logger.info("Downloading dialogue act model: %s", DEFAULT_DIALOGUE_ACT_MODEL)
        logger.info("Source: https://huggingface.co/%s", DEFAULT_DIALOGUE_ACT_MODEL)

        if progress_callback:
            class TqdmCallback(tqdm):
                def __init__(self, *args, **kwargs):
                    self._filename = kwargs.get("desc", "file")
                    super().__init__(*args, **kwargs)

                def update(self, n=1):
                    super().update(n)
                    if self.total and progress_callback:
                        downloaded_mb = self.n / (1024 * 1024)
                        total_mb = self.total / (1024 * 1024)
                        progress_callback(downloaded_mb, total_mb, self._filename)

            snapshot_download(
                DEFAULT_DIALOGUE_ACT_MODEL,
                local_files_only=False,
                tqdm_class=TqdmCallback,
            )
        else:
            snapshot_download(DEFAULT_DIALOGUE_ACT_MODEL, local_files_only=False)

        logger.info("Dialogue act model downloaded successfully")
        return True
    except Exception as e:
        error_msg = str(e)
        logger.error("Failed to download dialogue act model: %s", error_msg)
        if error_callback:
            error_callback(error_msg)
        return False


def get_dialogue_act_model_info() -> dict:
    """Get information about the dialogue act model."""
    return {
        "name": "DialogTag (DistilBERT)",
        "model_id": DEFAULT_DIALOGUE_ACT_MODEL,
        "url": f"https://huggingface.co/{DEFAULT_DIALOGUE_ACT_MODEL}",
        "size_mb": 250,  # Approximate size
        "description": "Dialogue act classification model for filler selection",
    }
