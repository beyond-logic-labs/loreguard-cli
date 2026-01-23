"""
Tests for NLI/HHEM grounding service.

These tests verify that the HHEM implementation correctly:
1. Detects HHEM vs RoBERTa models
2. Applies threshold correctly (score >= 0.6 = entailed)
3. Handles thread safety for concurrent requests
4. Handles edge cases (empty strings, batch processing)
"""

import pytest
import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock
from typing import List, Tuple

from src.nli import (
    NLIService,
    NLIResult,
    EntailmentLabel,
    DEFAULT_HHEM_THRESHOLD,
    DEFAULT_NLI_MODEL,
)


class TestHHEMDetection:
    """Tests for HHEM model detection logic."""

    def test_detects_hhem_by_full_name(self):
        """Should detect HHEM when full model name contains 'hallucination_evaluation_model'."""
        service = NLIService(model_path="vectara/hallucination_evaluation_model")
        assert service._use_hhem is True

    def test_detects_hhem_by_short_name(self):
        """Should detect HHEM when model path contains 'hhem'."""
        service = NLIService(model_path="/path/to/hhem")
        assert service._use_hhem is True

    def test_detects_hhem_case_insensitive(self):
        """Should detect HHEM regardless of case."""
        service = NLIService(model_path="VECTARA/HALLUCINATION_EVALUATION_MODEL")
        assert service._use_hhem is True

    def test_detects_roberta_not_hhem(self):
        """Should not detect RoBERTa models as HHEM."""
        service = NLIService(model_path="roberta-large-mnli")
        assert service._use_hhem is False

    def test_detects_ynie_not_hhem(self):
        """Should not detect ynie model as HHEM."""
        service = NLIService(model_path="ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
        assert service._use_hhem is False


class TestHHEMThreshold:
    """Tests for HHEM threshold logic."""

    def test_default_threshold(self):
        """Should use default threshold of 0.6."""
        service = NLIService()
        assert service._hhem_threshold == DEFAULT_HHEM_THRESHOLD
        assert service._hhem_threshold == 0.6

    def test_custom_threshold(self):
        """Should allow custom threshold."""
        service = NLIService(hhem_threshold=0.7)
        assert service._hhem_threshold == 0.7

    def test_threshold_from_env(self):
        """Should read threshold from environment variable."""
        with patch.dict("os.environ", {"NLI_HHEM_THRESHOLD": "0.8"}):
            service = NLIService()
            assert service._hhem_threshold == 0.8

    def test_explicit_threshold_overrides_env(self):
        """Explicit threshold should override environment variable."""
        with patch.dict("os.environ", {"NLI_HHEM_THRESHOLD": "0.8"}):
            service = NLIService(hhem_threshold=0.5)
            assert service._hhem_threshold == 0.5


class TestHHEMVerification:
    """Tests for HHEM verification logic with mocked model."""

    @pytest.fixture
    def mock_hhem_service(self):
        """Create a service with mocked HHEM model."""
        service = NLIService(model_path="vectara/hallucination_evaluation_model")

        # Mock the model's predict method
        mock_model = MagicMock()
        mock_model.predict = MagicMock()
        service._model = mock_model
        service._device = "cpu"

        return service

    def test_score_above_threshold_is_entailed(self, mock_hhem_service):
        """Score >= threshold should return ENTAILED."""
        mock_hhem_service._model.predict.return_value = [0.75]  # Above 0.6

        result = mock_hhem_service.verify(
            claim="The sky is blue",
            evidence="The sky appears blue during daytime."
        )

        assert result.entailment == EntailmentLabel.ENTAILED
        assert result.confidence == 0.75
        assert result.prob_entailment == 0.75
        assert result.prob_neutral == 0.25

    def test_score_below_threshold_is_neutral(self, mock_hhem_service):
        """Score < threshold should return NEUTRAL."""
        mock_hhem_service._model.predict.return_value = [0.4]  # Below 0.6

        result = mock_hhem_service.verify(
            claim="The sky is green",
            evidence="The sky appears blue during daytime."
        )

        assert result.entailment == EntailmentLabel.NEUTRAL
        assert result.confidence == 0.4
        assert result.prob_entailment == 0.4
        assert result.prob_neutral == 0.6

    def test_score_exactly_at_threshold_is_entailed(self, mock_hhem_service):
        """Score exactly at threshold should return ENTAILED (>= not >)."""
        mock_hhem_service._model.predict.return_value = [0.6]  # Exactly at threshold

        result = mock_hhem_service.verify(
            claim="Test claim",
            evidence="Test evidence"
        )

        assert result.entailment == EntailmentLabel.ENTAILED

    def test_verify_passes_evidence_claim_order_correctly(self, mock_hhem_service):
        """HHEM expects (evidence, claim) order - verify we pass it correctly."""
        mock_hhem_service._model.predict.return_value = [0.8]

        mock_hhem_service.verify(
            claim="My claim",
            evidence="My evidence"
        )

        # Check the order: (evidence, claim)
        call_args = mock_hhem_service._model.predict.call_args[0][0]
        assert call_args == [("My evidence", "My claim")]

    def test_custom_threshold_applied(self, mock_hhem_service):
        """Custom threshold should be applied correctly."""
        mock_hhem_service._hhem_threshold = 0.8
        mock_hhem_service._model.predict.return_value = [0.75]  # Above 0.6 but below 0.8

        result = mock_hhem_service.verify(claim="test", evidence="test")

        # Should be NEUTRAL because 0.75 < 0.8
        assert result.entailment == EntailmentLabel.NEUTRAL


class TestHHEMBatchVerification:
    """Tests for HHEM batch verification."""

    @pytest.fixture
    def mock_hhem_service(self):
        """Create a service with mocked HHEM model."""
        service = NLIService(model_path="vectara/hallucination_evaluation_model")

        mock_model = MagicMock()
        mock_model.predict = MagicMock()
        service._model = mock_model
        service._device = "cpu"

        return service

    def test_batch_returns_correct_count(self, mock_hhem_service):
        """Batch should return same number of results as inputs."""
        mock_hhem_service._model.predict.return_value = [0.8, 0.5, 0.7]

        pairs = [
            ("claim1", "evidence1"),
            ("claim2", "evidence2"),
            ("claim3", "evidence3"),
        ]

        results = mock_hhem_service.verify_batch(pairs)

        assert len(results) == 3

    def test_batch_empty_input(self, mock_hhem_service):
        """Empty batch should return empty results."""
        results = mock_hhem_service.verify_batch([])
        assert results == []

    def test_batch_applies_threshold_to_each(self, mock_hhem_service):
        """Each result in batch should have threshold applied."""
        mock_hhem_service._model.predict.return_value = [0.8, 0.4, 0.6]  # above, below, at

        pairs = [
            ("claim1", "evidence1"),
            ("claim2", "evidence2"),
            ("claim3", "evidence3"),
        ]

        results = mock_hhem_service.verify_batch(pairs)

        assert results[0].entailment == EntailmentLabel.ENTAILED  # 0.8 >= 0.6
        assert results[1].entailment == EntailmentLabel.NEUTRAL   # 0.4 < 0.6
        assert results[2].entailment == EntailmentLabel.ENTAILED  # 0.6 >= 0.6

    def test_batch_passes_correct_order(self, mock_hhem_service):
        """Batch should pass (evidence, claim) pairs to model."""
        mock_hhem_service._model.predict.return_value = [0.8, 0.7]

        pairs = [
            ("claim1", "evidence1"),
            ("claim2", "evidence2"),
        ]

        mock_hhem_service.verify_batch(pairs)

        # Check the pairs passed to model: (evidence, claim) order
        call_args = mock_hhem_service._model.predict.call_args[0][0]
        assert call_args == [("evidence1", "claim1"), ("evidence2", "claim2")]


class TestThreadSafety:
    """Tests for thread safety of NLI service."""

    @pytest.fixture
    def mock_hhem_service(self):
        """Create a service with mocked HHEM model that tracks concurrent access."""
        service = NLIService(model_path="vectara/hallucination_evaluation_model")

        # Track concurrent access
        service._concurrent_calls = 0
        service._max_concurrent = 0
        service._call_lock = threading.Lock()

        def mock_predict(pairs):
            with service._call_lock:
                service._concurrent_calls += 1
                service._max_concurrent = max(service._max_concurrent, service._concurrent_calls)

            # Simulate some processing time
            time.sleep(0.05)

            with service._call_lock:
                service._concurrent_calls -= 1

            return [0.8] * len(pairs)

        mock_model = MagicMock()
        mock_model.predict = mock_predict
        service._model = mock_model
        service._device = "cpu"

        return service

    def test_concurrent_calls_are_serialized(self, mock_hhem_service):
        """Concurrent verify() calls should be serialized (max 1 at a time)."""
        results = []
        errors = []

        def worker():
            try:
                result = mock_hhem_service.verify("claim", "evidence")
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed
        assert len(errors) == 0
        assert len(results) == 5

        # Max concurrent should be 1 due to lock
        assert mock_hhem_service._max_concurrent == 1

    def test_inference_lock_exists(self):
        """Service should have an inference lock."""
        service = NLIService()
        assert hasattr(service, "_inference_lock")
        assert isinstance(service._inference_lock, type(threading.Lock()))


class TestPredictHHEMOutputHandling:
    """Tests for _predict_hhem output normalization."""

    @pytest.fixture
    def mock_hhem_service(self):
        """Create a service with mocked HHEM model."""
        service = NLIService(model_path="vectara/hallucination_evaluation_model")
        service._model = MagicMock()
        service._device = "cpu"
        return service

    def test_handles_tensor_output(self, mock_hhem_service):
        """Should handle PyTorch tensor output."""
        import torch
        mock_hhem_service._model.predict.return_value = torch.tensor([0.75, 0.8])

        scores = mock_hhem_service._predict_hhem([("e1", "c1"), ("e2", "c2")])

        # Use approximate comparison due to float32 precision
        assert len(scores) == 2
        assert abs(scores[0] - 0.75) < 1e-6
        assert abs(scores[1] - 0.8) < 1e-6

    def test_handles_list_output(self, mock_hhem_service):
        """Should handle list output."""
        mock_hhem_service._model.predict.return_value = [0.75, 0.8]

        scores = mock_hhem_service._predict_hhem([("e1", "c1"), ("e2", "c2")])

        assert scores == [0.75, 0.8]

    def test_handles_single_float_output(self, mock_hhem_service):
        """Should handle single float output (wrap in list)."""
        mock_hhem_service._model.predict.return_value = 0.75

        scores = mock_hhem_service._predict_hhem([("e1", "c1")])

        assert scores == [0.75]

    def test_handles_numpy_array_output(self, mock_hhem_service):
        """Should handle numpy array output."""
        import numpy as np
        mock_hhem_service._model.predict.return_value = np.array([0.75, 0.8])

        scores = mock_hhem_service._predict_hhem([("e1", "c1"), ("e2", "c2")])

        assert scores == [0.75, 0.8]


class TestModelProperties:
    """Tests for model property accessors."""

    def test_is_loaded_false_initially(self):
        """is_loaded should be False before loading."""
        service = NLIService()
        assert service.is_loaded is False

    def test_device_none_initially(self):
        """device should be None before loading."""
        service = NLIService()
        assert service.device is None

    def test_model_name_returns_path(self):
        """model_name property should return the configured path."""
        service = NLIService(model_path="test/model")
        assert service.model_name == "test/model"

    def test_default_model_is_hhem(self):
        """Default model should be HHEM."""
        service = NLIService()
        assert service.model_name == DEFAULT_NLI_MODEL
        assert "hallucination_evaluation_model" in service.model_name


class TestNLIResultDataclass:
    """Tests for NLIResult dataclass."""

    def test_result_fields(self):
        """NLIResult should have all expected fields."""
        result = NLIResult(
            entailment=EntailmentLabel.ENTAILED,
            confidence=0.9,
            prob_entailment=0.9,
            prob_neutral=0.05,
            prob_contradiction=0.05,
        )

        assert result.entailment == EntailmentLabel.ENTAILED
        assert result.confidence == 0.9
        assert result.prob_entailment == 0.9
        assert result.prob_neutral == 0.05
        assert result.prob_contradiction == 0.05

    def test_result_defaults(self):
        """NLIResult should have sensible defaults."""
        result = NLIResult(
            entailment=EntailmentLabel.NEUTRAL,
            confidence=0.5,
        )

        assert result.prob_entailment == 0.0
        assert result.prob_neutral == 0.0
        assert result.prob_contradiction == 0.0


class TestEntailmentLabel:
    """Tests for EntailmentLabel enum."""

    def test_label_values(self):
        """EntailmentLabel should have expected values."""
        assert EntailmentLabel.ENTAILED.value == "entailed"
        assert EntailmentLabel.NEUTRAL.value == "neutral"
        assert EntailmentLabel.CONTRADICTS.value == "contradicts"

    def test_label_is_string(self):
        """EntailmentLabel should be usable as string."""
        assert str(EntailmentLabel.ENTAILED) == "EntailmentLabel.ENTAILED"
        assert EntailmentLabel.ENTAILED == "entailed"  # Due to str base class
