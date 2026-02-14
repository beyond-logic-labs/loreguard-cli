"""Chunk Detection service for natural conversation breaks (ADR-0023).

This module provides zero-shot classification to detect natural break points
in NPC responses. It splits a response into multiple chunks that can be
delivered sequentially for more human-like conversation flow.

Uses DeBERTa-v3-large-zeroshot to classify sentence boundaries:
- "continues same thought" → merge with previous chunk
- "starts new thought" → create new chunk

This is the client-side implementation that runs locally on the user's machine,
leveraging the same DeBERTa model used for intent classification.
"""

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A single chunk of text in a response."""
    text: str
    index: int  # Position in sequence (0-based)


@dataclass
class ChunkResult:
    """Result of chunk detection."""
    chunks: List[TextChunk]
    latency_ms: int  # Detection latency in milliseconds


# Hypotheses for zero-shot classification of sentence boundaries
CHUNK_HYPOTHESES = {
    "continues": "This text continues the same thought or topic as the previous sentence.",
    "starts_new": "This text starts a new thought, topic, or conversational turn.",
}

# Threshold for "starts new thought" classification
# If confidence > threshold, we create a new chunk
NEW_THOUGHT_THRESHOLD = 0.55


class ChunkDetector:
    """Service for detecting natural conversation breaks using DeBERTa.

    Uses zero-shot classification to determine where to split a response
    into natural chunks for more human-like delivery.
    """

    def __init__(self, classifier=None, model_path: Optional[str] = None):
        """Initialize the chunk detector.

        Args:
            classifier: Optional pre-loaded zero-shot classifier to reuse.
                       If None, will use IntentClassifier's model.
            model_path: Path to local model directory. If None, uses HuggingFace hub.
        """
        self._classifier = classifier
        self._model_path = model_path or "MoritzLaurer/DeBERTa-v3-large-zeroshot-v2.0"
        self._device = None
        self._load_lock = threading.Lock()

    @property
    def model_name(self) -> str:
        """Return the configured model identifier."""
        return self._model_path

    def set_classifier(self, classifier):
        """Set a pre-loaded classifier to reuse.

        This allows sharing the DeBERTa model with IntentClassifier
        to avoid loading it twice.
        """
        self._classifier = classifier

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

        Thread-safe: uses lock to prevent concurrent model loading.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        if self._classifier is not None:
            return True

        with self._load_lock:
            if self._classifier is not None:
                return True

            try:
                from transformers import pipeline

                self._device = self._resolve_device()
                logger.info(f"Loading chunk detector: {self._model_path} (device={self._device})")

                device_idx = 0 if self._device == "cuda" else -1 if self._device == "cpu" else 0
                self._classifier = pipeline(
                    "zero-shot-classification",
                    model=self._model_path,
                    device=device_idx if self._device != "mps" else "mps",
                )

                logger.info("Chunk detector loaded successfully")
                return True

            except Exception as e:
                logger.error(f"Failed to load chunk detector: {e}")
                return False

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences at natural break points.

        Uses a simple regex-based approach that handles common cases:
        - Period, exclamation, question mark followed by space/end
        - Ellipsis (...)
        - Preserves quotes and parentheses
        """
        # Split on sentence-ending punctuation followed by space or end
        # Handles: . ! ? ... followed by space or end
        pattern = r'(?<=[.!?])\s+|(?<=\.\.\.)\s*'
        sentences = re.split(pattern, text.strip())

        # Filter out empty strings and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def detect(self, text: str) -> ChunkResult:
        """Detect natural break points in text and split into chunks.

        Args:
            text: The NPC response text to analyze

        Returns:
            ChunkResult with list of TextChunk objects
        """
        start_time = time.time()

        # Handle empty or very short text
        if not text or len(text.strip()) < 10:
            return ChunkResult(
                chunks=[TextChunk(text=text, index=0)] if text else [],
                latency_ms=0,
            )

        # Split into sentences
        sentences = self._split_into_sentences(text)

        # If only one sentence, return as single chunk
        if len(sentences) <= 1:
            latency_ms = int((time.time() - start_time) * 1000)
            return ChunkResult(
                chunks=[TextChunk(text=text, index=0)],
                latency_ms=latency_ms,
            )

        # Ensure model is loaded
        if self._classifier is None:
            if not self.load_model():
                # Fallback: return full text as single chunk
                return ChunkResult(
                    chunks=[TextChunk(text=text, index=0)],
                    latency_ms=0,
                )

        # Classify each sentence boundary
        chunks: List[str] = [sentences[0]]
        hypotheses = list(CHUNK_HYPOTHESES.values())

        for i in range(1, len(sentences)):
            prev_sentence = sentences[i - 1]
            curr_sentence = sentences[i]

            # Create context for classification
            # We ask: does curr_sentence continue prev_sentence's thought?
            context = f"{prev_sentence} {curr_sentence}"

            try:
                result = self._classifier(
                    context,
                    candidate_labels=hypotheses,
                    hypothesis_template="{}",
                    multi_label=False,
                )

                # Check if "starts new thought" won
                starts_new_idx = hypotheses.index(CHUNK_HYPOTHESES["starts_new"])
                starts_new_score = 0.0

                for j, label in enumerate(result["labels"]):
                    if label == CHUNK_HYPOTHESES["starts_new"]:
                        starts_new_score = result["scores"][j]
                        break

                if starts_new_score > NEW_THOUGHT_THRESHOLD:
                    # Start new chunk
                    chunks.append(curr_sentence)
                    logger.debug(f"New chunk at sentence {i}: score={starts_new_score:.2f}")
                else:
                    # Merge with previous chunk
                    chunks[-1] = f"{chunks[-1]} {curr_sentence}"
                    logger.debug(f"Merged sentence {i}: score={starts_new_score:.2f}")

            except Exception as e:
                logger.warning(f"Classification failed for sentence {i}, merging: {e}")
                chunks[-1] = f"{chunks[-1]} {curr_sentence}"

        latency_ms = int((time.time() - start_time) * 1000)

        # Convert to TextChunk objects
        text_chunks = [
            TextChunk(text=chunk.strip(), index=i)
            for i, chunk in enumerate(chunks)
            if chunk.strip()
        ]

        logger.info(f"Chunk detection: {len(sentences)} sentences -> {len(text_chunks)} chunks (latency={latency_ms}ms)")

        return ChunkResult(
            chunks=text_chunks,
            latency_ms=latency_ms,
        )

    def detect_with_fallback(self, text: str) -> ChunkResult:
        """Detect chunks with fallback to single chunk on error.

        Args:
            text: The NPC response text to analyze

        Returns:
            ChunkResult (defaults to single chunk on error)
        """
        try:
            return self.detect(text)
        except Exception as e:
            logger.warning(f"Chunk detection failed, returning single chunk: {e}")
            return ChunkResult(
                chunks=[TextChunk(text=text, index=0)] if text else [],
                latency_ms=0,
            )

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._classifier is not None

    @property
    def device(self) -> Optional[str]:
        """Get the device being used."""
        return self._device
