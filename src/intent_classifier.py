"""Intent Classification service for adaptive retrieval (ADR-0010).

This module provides zero-shot intent classification for the NPC dialogue pipeline.
It uses BART-large-MNLI to classify user messages into retrieval strategy categories:
- A_NO_RETRIEVAL: Greetings, chitchat, farewells (skip retrieval)
- B_WORKING_MEMORY: Simple identity/state questions (working memory only)
- C_LIGHT_RETRIEVAL: Direct factual questions (top 3 sources)
- D_FULL_RETRIEVAL: Complex multi-hop questions (full RAG pipeline)

This is the client-side implementation that runs locally on the user's machine,
providing fast inference (50-100ms).
"""

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class IntentLabel(str, Enum):
    """Intent classification labels for adaptive retrieval."""
    NO_RETRIEVAL = "A_NO_RETRIEVAL"
    WORKING_MEMORY = "B_WORKING_MEMORY"
    LIGHT_RETRIEVAL = "C_LIGHT_RETRIEVAL"
    FULL_RETRIEVAL = "D_FULL_RETRIEVAL"


@dataclass
class IntentResult:
    """Result of intent classification."""
    intent: IntentLabel
    confidence: float  # 0-1 confidence score
    latency_ms: int    # Classification latency in milliseconds


# Default model for intent classification
# DeBERTa-v3-large is state-of-the-art for zero-shot classification
DEFAULT_INTENT_MODEL = "MoritzLaurer/DeBERTa-v3-large-zeroshot-v2.0"

# Intent hypothesis templates for zero-shot classification
# Each intent maps to a hypothesis that BART will evaluate
INTENT_HYPOTHESES = {
    IntentLabel.NO_RETRIEVAL: "This is a greeting, chitchat, or farewell that does not require any information retrieval.",
    IntentLabel.WORKING_MEMORY: "This is a simple question about identity, name, or basic state that only requires basic memory.",
    IntentLabel.LIGHT_RETRIEVAL: "This is a direct factual question that requires looking up specific information.",
    IntentLabel.FULL_RETRIEVAL: "This is a complex question that requires comprehensive information retrieval and analysis.",
}

# Promise detection hypothesis for follow-up triggers (ADR-0020)
# Single hypothesis - confidence score determines if it's a promise
PROMISE_HYPOTHESIS = "This statement contains a promise to check, investigate, look into, find out, or get back to someone later."


@dataclass
class PromiseResult:
    """Result of promise classification."""
    has_promise: bool   # True if confidence > threshold
    confidence: float   # Raw score from model (0-1)
    latency_ms: int     # Classification latency in milliseconds


class IntentClassifier:
    """Service for zero-shot intent classification using BART-large-MNLI.

    Uses zero-shot classification to categorize user messages into one of four
    retrieval strategies without any fine-tuning required.
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the intent classifier.

        Args:
            model_path: Path to local model directory. If None, uses HuggingFace hub.
        """
        self._classifier = None
        self._model_path = model_path or DEFAULT_INTENT_MODEL
        self._device = None
        self._load_lock = threading.Lock()  # Protect lazy loading from race conditions

    @property
    def model_name(self) -> str:
        """Return the configured model identifier."""
        return self._model_path

    def _resolve_device(self) -> str:
        """Resolve the best available device."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
            return "cpu"
        except ImportError:
            return "cpu"

    def load_model(self) -> bool:
        """Load the classification model.

        Thread-safe: uses lock to prevent concurrent model loading which could
        cause memory issues or race conditions.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        # Fast path: already loaded
        if self._classifier is not None:
            return True

        with self._load_lock:
            # Double-check after acquiring lock
            if self._classifier is not None:
                return True

            try:
                from transformers import pipeline

                self._device = self._resolve_device()
                logger.info(f"Loading intent classifier: {self._model_path} (device={self._device})")

                # Use zero-shot-classification pipeline
                device_idx = 0 if self._device == "cuda" else -1 if self._device == "cpu" else 0
                self._classifier = pipeline(
                    "zero-shot-classification",
                    model=self._model_path,
                    device=device_idx if self._device != "mps" else "mps",
                )

                logger.info("Intent classifier loaded successfully")
                return True

            except Exception as e:
                logger.error(f"Failed to load intent classifier: {e}")
                return False

    def classify(self, query: str) -> IntentResult:
        """Classify the intent of a user query.

        Args:
            query: The user's message to classify

        Returns:
            IntentResult with intent classification and confidence
        """
        import time

        if self._classifier is None:
            if not self.load_model():
                raise RuntimeError("Intent classifier not loaded")

        start_time = time.time()

        # Get candidate labels and hypotheses
        labels = list(INTENT_HYPOTHESES.keys())
        hypotheses = list(INTENT_HYPOTHESES.values())

        # Run zero-shot classification
        # The pipeline will evaluate each hypothesis against the query
        result = self._classifier(
            query,
            candidate_labels=hypotheses,
            hypothesis_template="{}",  # Use hypotheses directly
            multi_label=False,
        )

        latency_ms = int((time.time() - start_time) * 1000)

        # Map the winning hypothesis back to intent label
        winning_hypothesis = result["labels"][0]
        confidence = result["scores"][0]

        # Find the intent that corresponds to the winning hypothesis
        intent = IntentLabel.FULL_RETRIEVAL  # Default
        for label, hypothesis in INTENT_HYPOTHESES.items():
            if hypothesis == winning_hypothesis:
                intent = label
                break

        logger.info(f"Intent classification: {intent.value} (confidence={confidence:.2f}, latency={latency_ms}ms)")

        return IntentResult(
            intent=intent,
            confidence=confidence,
            latency_ms=latency_ms,
        )

    def classify_with_fallback(self, query: str) -> IntentResult:
        """Classify intent with fallback to full retrieval on error.

        Args:
            query: The user's message to classify

        Returns:
            IntentResult (defaults to FULL_RETRIEVAL on error)
        """
        try:
            return self.classify(query)
        except Exception as e:
            logger.warning(f"Intent classification failed, falling back to full retrieval: {e}")
            return IntentResult(
                intent=IntentLabel.FULL_RETRIEVAL,
                confidence=0.0,  # 0 confidence indicates fallback
                latency_ms=0,
            )

    def classify_promise(self, text: str, threshold: float = 0.5) -> PromiseResult:
        """Classify whether NPC response contains a follow-up promise (ADR-0020).

        Uses single-label zero-shot classification: evaluates how well the text
        matches the promise hypothesis. High confidence = promise detected.

        Args:
            text: The NPC's response text to classify
            threshold: Confidence threshold for promise detection (default 0.5)

        Returns:
            PromiseResult with has_promise, confidence, and latency_ms
        """
        import time

        if self._classifier is None:
            if not self.load_model():
                raise RuntimeError("Intent classifier not loaded")

        start_time = time.time()

        # Run zero-shot classification with single hypothesis
        result = self._classifier(
            text,
            candidate_labels=[PROMISE_HYPOTHESIS],
            hypothesis_template="{}",
        )

        latency_ms = int((time.time() - start_time) * 1000)
        confidence = result["scores"][0]
        has_promise = confidence > threshold

        logger.info(f"Promise classification: has_promise={has_promise} (confidence={confidence:.2f}, threshold={threshold}, latency={latency_ms}ms)")

        return PromiseResult(
            has_promise=has_promise,
            confidence=confidence,
            latency_ms=latency_ms,
        )

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._classifier is not None

    @property
    def device(self) -> Optional[str]:
        """Get the device being used."""
        return self._device


def is_intent_model_available() -> bool:
    """Check if the intent model is available in HuggingFace cache.

    The transformers library caches models in ~/.cache/huggingface/hub/.
    This function checks if the BART model has been downloaded.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
        # Check if the model config file is cached (indicates model is downloaded)
        cached_path = try_to_load_from_cache(DEFAULT_INTENT_MODEL, "config.json")
        return cached_path is not None
    except ImportError:
        # huggingface_hub not installed, can't check cache
        return False
    except Exception:
        return False


def download_intent_model(progress_callback=None, error_callback=None) -> bool:
    """Pre-download the intent model to HuggingFace cache.

    This is optional - the model will be downloaded automatically on first use.
    But calling this explicitly gives better user feedback during setup.

    Args:
        progress_callback: Optional callback(downloaded_mb, total_mb, filename) for progress updates.
        error_callback: Optional callback(error_message) to report errors.

    Returns:
        True if download successful, False otherwise.
    """
    try:
        from huggingface_hub import snapshot_download
        from tqdm import tqdm

        logger.info(f"Downloading intent model: {DEFAULT_INTENT_MODEL}")
        logger.info(f"Source: https://huggingface.co/{DEFAULT_INTENT_MODEL}")

        if progress_callback:
            # Create custom tqdm class to report progress
            class TqdmCallback(tqdm):
                def __init__(self, *args, **kwargs):
                    # Extract filename from desc if available
                    self._filename = kwargs.get('desc', 'file')
                    super().__init__(*args, **kwargs)

                def update(self, n=1):
                    super().update(n)
                    if self.total and progress_callback:
                        downloaded_mb = self.n / (1024 * 1024)
                        total_mb = self.total / (1024 * 1024)
                        progress_callback(downloaded_mb, total_mb, self._filename)

            snapshot_download(
                DEFAULT_INTENT_MODEL,
                local_files_only=False,
                tqdm_class=TqdmCallback,
            )
        else:
            snapshot_download(DEFAULT_INTENT_MODEL, local_files_only=False)

        logger.info("Intent model downloaded successfully")
        return True
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to download intent model: {error_msg}")
        if error_callback:
            error_callback(error_msg)
        return False


def get_intent_model_info() -> dict:
    """Get information about the intent model.

    Returns:
        Dict with model info: name, url, size_mb
    """
    return {
        "name": "DeBERTa-v3-large-zeroshot",
        "model_id": DEFAULT_INTENT_MODEL,
        "url": f"https://huggingface.co/{DEFAULT_INTENT_MODEL}",
        "size_mb": 800,  # Approximate size for DeBERTa-v3-large
        "description": "State-of-the-art zero-shot classification model",
    }
