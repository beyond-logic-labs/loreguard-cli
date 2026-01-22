"""NLI/grounding service for fact verification.

This module provides entailment verification for the NPC dialogue pipeline.
It supports both traditional NLI models and HHEM-style grounding models.

This is the client-side implementation that runs locally on the user's machine,
providing fast inference (50-100ms vs 1500ms on Fargate).
"""

import logging
from dataclasses import dataclass
from enum import Enum
import os
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class EntailmentLabel(str, Enum):
    """Entailment classification labels."""
    ENTAILED = "entailed"
    NEUTRAL = "neutral"
    CONTRADICTS = "contradicts"


@dataclass
class NLIResult:
    """Result of NLI verification."""
    entailment: EntailmentLabel
    confidence: float  # 0-1 confidence score
    prob_entailment: float = 0.0
    prob_neutral: float = 0.0
    prob_contradiction: float = 0.0


# Default model for verification (HHEM grounding)
DEFAULT_NLI_MODEL = "vectara/hallucination_evaluation_model"

# HHEM grounding threshold (entailed if score >= threshold)
DEFAULT_HHEM_THRESHOLD = 0.6


class NLIService:
    """Service for NLI-based entailment verification.

    Uses an NLI or grounding model to classify the relationship between
    evidence (premise) and claim (hypothesis) as:
    - ENTAILMENT: Evidence supports the claim
    - NEUTRAL: Evidence neither supports nor contradicts
    - CONTRADICTION: Evidence contradicts the claim
    """

    def __init__(self, model_path: Optional[str] = None, hhem_threshold: Optional[float] = None):
        """Initialize the NLI service.

        Args:
            model_path: Path to local model directory. If None, uses HuggingFace hub.
        """
        self._model = None
        self._tokenizer = None
        self._model_path = model_path or DEFAULT_NLI_MODEL
        self._use_hhem = self._is_hhem_model(self._model_path)
        self._hhem_threshold = hhem_threshold or float(os.getenv("NLI_HHEM_THRESHOLD", DEFAULT_HHEM_THRESHOLD))
        self._device = None
        self._max_length = 512
        self._label_order = None

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

    def _is_hhem_model(self, model_name: str) -> bool:
        """Detect if the selected model is an HHEM-style grounding scorer."""
        name = model_name.lower()
        return "hallucination_evaluation_model" in name or "hhem" in name

    def load_model(self) -> bool:
        """Load the NLI model and tokenizer.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        if self._model is not None:
            return True

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            self._device = self._resolve_device()
            logger.info(f"Loading NLI model: {self._model_path} (device={self._device})")

            if self._use_hhem:
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self._model_path,
                    trust_remote_code=True,
                )
                self._model.to(self._device)
                self._model.eval()

                if not hasattr(self._model, "predict"):
                    raise RuntimeError("HHEM model missing predict() helper")

                self._label_order = ["entailment", "neutral"]
                logger.info("HHEM model loaded successfully")
                return True

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(self._model_path)
            self._model.to(self._device)
            self._model.eval()

            # Determine label order from model config
            id2label = self._model.config.id2label
            logger.info(f"Model label mapping: {id2label}")

            # Build label order list
            self._label_order = []
            for i in range(len(id2label)):
                label = id2label[i].lower()
                if "entail" in label or "support" in label or "ground" in label:
                    self._label_order.append("entailment")
                elif "contra" in label or "refute" in label:
                    self._label_order.append("contradiction")
                elif "neutral" in label or "unknown" in label:
                    self._label_order.append("neutral")
                else:
                    self._label_order.append("neutral")

            logger.info(f"NLI model loaded successfully (label_order={self._label_order})")
            return True

        except Exception as e:
            logger.error(f"Failed to load NLI model: {e}")
            return False

    def verify(self, claim: str, evidence: str) -> NLIResult:
        """Verify if evidence entails the claim.

        Args:
            claim: The statement to verify (hypothesis)
            evidence: The supporting text (premise)

        Returns:
            NLIResult with entailment classification and confidence
        """
        if self._model is None:
            if not self.load_model():
                raise RuntimeError("NLI model not loaded")

        import torch

        if self._use_hhem:
            scores = self._predict_hhem([(evidence, claim)])
            score = scores[0]
            entailment = EntailmentLabel.ENTAILED if score >= self._hhem_threshold else EntailmentLabel.NEUTRAL
            return NLIResult(
                entailment=entailment,
                confidence=score,
                prob_entailment=score,
                prob_neutral=1.0 - score,
                prob_contradiction=0.0,
            )

        # Tokenize with proper sentence-pair format
        inputs = self._tokenizer(
            evidence,  # Premise first
            claim,     # Hypothesis second
            max_length=self._max_length,
            truncation=True,
            return_tensors="pt",
            padding=True,
        )

        # Move to device
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0].cpu().tolist()

        # Map probabilities to labels
        prob_map = {}
        for i, label in enumerate(self._label_order):
            prob_map[label] = probs[i]

        # Find predicted label
        predicted_idx = probs.index(max(probs))
        predicted_label = self._label_order[predicted_idx]
        confidence = max(probs)

        # Map to our enum
        label_to_enum = {
            "entailment": EntailmentLabel.ENTAILED,
            "neutral": EntailmentLabel.NEUTRAL,
            "contradiction": EntailmentLabel.CONTRADICTS,
        }
        entailment = label_to_enum[predicted_label]

        return NLIResult(
            entailment=entailment,
            confidence=confidence,
            prob_entailment=prob_map.get("entailment", 0.0),
            prob_neutral=prob_map.get("neutral", 0.0),
            prob_contradiction=prob_map.get("contradiction", 0.0),
        )

    def verify_batch(self, claims_evidence: List[Tuple[str, str]]) -> List[NLIResult]:
        """Verify multiple claim-evidence pairs in batch.

        Args:
            claims_evidence: List of (claim, evidence) tuples

        Returns:
            List of NLIResult for each pair
        """
        if self._model is None:
            if not self.load_model():
                raise RuntimeError("NLI model not loaded")

        if not claims_evidence:
            return []

        import torch

        if self._use_hhem:
            scores = self._predict_hhem([(evidence, claim) for claim, evidence in claims_evidence])
            results = []
            for score in scores:
                entailment = EntailmentLabel.ENTAILED if score >= self._hhem_threshold else EntailmentLabel.NEUTRAL
                results.append(NLIResult(
                    entailment=entailment,
                    confidence=score,
                    prob_entailment=score,
                    prob_neutral=1.0 - score,
                    prob_contradiction=0.0,
                ))
            return results

        # Prepare batch inputs - evidence (premise) first!
        premises = [evidence for claim, evidence in claims_evidence]
        hypotheses = [claim for claim, evidence in claims_evidence]

        inputs = self._tokenizer(
            premises,
            hypotheses,
            max_length=self._max_length,
            truncation=True,
            return_tensors="pt",
            padding=True,
        )

        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            all_probs = torch.softmax(logits, dim=1).cpu().tolist()

        results = []
        label_to_enum = {
            "entailment": EntailmentLabel.ENTAILED,
            "neutral": EntailmentLabel.NEUTRAL,
            "contradiction": EntailmentLabel.CONTRADICTS,
        }

        for probs in all_probs:
            prob_map = {}
            for i, label in enumerate(self._label_order):
                prob_map[label] = probs[i]

            predicted_idx = probs.index(max(probs))
            predicted_label = self._label_order[predicted_idx]
            confidence = max(probs)
            entailment = label_to_enum[predicted_label]

            results.append(NLIResult(
                entailment=entailment,
                confidence=confidence,
                prob_entailment=prob_map.get("entailment", 0.0),
                prob_neutral=prob_map.get("neutral", 0.0),
                prob_contradiction=prob_map.get("contradiction", 0.0),
            ))

        return results

    def _predict_hhem(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Run HHEM prediction and normalize output to list of floats."""
        import torch
        scores = self._model.predict(pairs)
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().tolist()
        elif hasattr(scores, "tolist"):
            scores = scores.tolist()
        if isinstance(scores, float):
            return [scores]
        return [float(s) for s in scores]

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None

    @property
    def device(self) -> Optional[str]:
        """Get the device being used."""
        return self._device


def is_nli_model_available() -> bool:
    """Check if the NLI model is available in HuggingFace cache.

    The transformers library caches models in ~/.cache/huggingface/hub/.
    This function checks if the configured NLI/HHEM model has been downloaded.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
        # Check if the model config file is cached (indicates model is downloaded)
        cached_path = try_to_load_from_cache(DEFAULT_NLI_MODEL, "config.json")
        return cached_path is not None
    except ImportError:
        # huggingface_hub not installed, can't check cache
        return False
    except Exception:
        return False


def download_nli_model(progress_callback=None, error_callback=None) -> bool:
    """Pre-download the NLI model to HuggingFace cache.

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

        logger.info(f"Downloading NLI model: {DEFAULT_NLI_MODEL}")
        logger.info(f"Source: https://huggingface.co/{DEFAULT_NLI_MODEL}")

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
                DEFAULT_NLI_MODEL,
                local_files_only=False,
                tqdm_class=TqdmCallback,
            )
        else:
            snapshot_download(DEFAULT_NLI_MODEL, local_files_only=False)

        logger.info("NLI model downloaded successfully")
        return True
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to download NLI model: {error_msg}")
        if error_callback:
            error_callback(error_msg)
        return False


def get_nli_model_info() -> dict:
    """Get information about the NLI model.

    Returns:
        Dict with model info: name, url, size_mb
    """
    return {
        "name": "HHEM Grounding Model",
        "model_id": DEFAULT_NLI_MODEL,
        "url": f"https://huggingface.co/{DEFAULT_NLI_MODEL}",
        "size_mb": 600,  # Approximate size
        "description": "Grounding model for citation verification (HHEM)",
    }
