from src.intent_classifier import (
    INTENT_LABEL_DESCRIPTIONS,
    IntentClassifier,
    IntentLabel,
)


class _FakeClassifier:
    def __init__(self, labels, scores):
        self.labels = labels
        self.scores = scores
        self.calls = []

    def __call__(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return {"labels": self.labels, "scores": self.scores}


def test_classify_uses_descriptive_labels_and_maps_winner():
    winner = INTENT_LABEL_DESCRIPTIONS[IntentLabel.NO_RETRIEVAL]
    second = INTENT_LABEL_DESCRIPTIONS[IntentLabel.LIGHT_RETRIEVAL]
    fake = _FakeClassifier(labels=[winner, second], scores=[0.88, 0.12])

    classifier = IntentClassifier(model_path="dummy-model")
    classifier._classifier = fake

    result = classifier.classify("hi")

    assert result.intent == IntentLabel.NO_RETRIEVAL
    assert result.confidence == 0.88
    assert result.latency_ms >= 0

    assert len(fake.calls) == 1
    _, kwargs = fake.calls[0]
    assert kwargs["candidate_labels"] == list(INTENT_LABEL_DESCRIPTIONS.values())
    assert kwargs["hypothesis_template"] == "This user message is {}."
    assert kwargs["multi_label"] is False


def test_classify_with_fallback_returns_full_retrieval_on_error():
    classifier = IntentClassifier(model_path="dummy-model")
    classifier._classifier = None

    result = classifier.classify_with_fallback("hi")

    assert result.intent == IntentLabel.FULL_RETRIEVAL
    assert result.confidence == 0.0
    assert result.latency_ms == 0
