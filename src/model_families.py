"""Model family profiles for chat template and stop sequence configuration.

Different model families (Llama, Qwen, Gemma, etc.) use different chat template
formats and stop tokens. This module provides preconfigured profiles so users
can switch models without manually adjusting server flags.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelFamilyProfile:
    """Preconfigured settings for a model family.

    Attributes:
        id: Unique identifier (used in config.json).
        name: Human-readable display name.
        chat_template_file: Jinja template filename (relative to templates/).
            None means use the model's GGUF-embedded template via --jinja.
        stop_sequences: Model-family-specific stop tokens for generation.
        description: Short description for UI display.
    """
    id: str
    name: str
    chat_template_file: Optional[str]
    stop_sequences: tuple[str, ...]
    description: str


# Registry of known model family profiles.
# Key = profile ID (stored in config.json as model_family).
MODEL_FAMILIES: dict[str, ModelFamilyProfile] = {
    "llama3": ModelFamilyProfile(
        id="llama3",
        name="Llama 3 / 3.1",
        chat_template_file="llama31-no-tools.jinja",
        stop_sequences=(
            "<|eot_id|>",
            "<|end_of_text|>",
        ),
        description="Meta Llama 3.x series. Uses custom template to disable tool-calling.",
    ),
    "qwen3": ModelFamilyProfile(
        id="qwen3",
        name="Qwen 3 / 3.5",
        chat_template_file=None,
        stop_sequences=(
            "<|im_end|>",
            "<|im_start|>",
            "<|endoftext|>",
        ),
        description="Alibaba Qwen 3.x series. ChatML format with thinking support.",
    ),
    "gemma": ModelFamilyProfile(
        id="gemma",
        name="Google Gemma",
        chat_template_file=None,
        stop_sequences=(
            "<end_of_turn>",
            "<start_of_turn>",
        ),
        description="Google Gemma models. Uses model-embedded template.",
    ),
    "chatml": ModelFamilyProfile(
        id="chatml",
        name="ChatML (Generic)",
        chat_template_file=None,
        stop_sequences=(
            "<|im_end|>",
            "<|im_start|>",
            "<|endoftext|>",
            "</s>",
        ),
        description="Generic ChatML-compatible models (Nous Hermes, OpenChat, etc.).",
    ),
}

DEFAULT_MODEL_FAMILY = "llama3"


def get_model_family(family_id: str) -> ModelFamilyProfile:
    """Get a model family profile by ID.

    Falls back to DEFAULT_MODEL_FAMILY if the ID is unknown.
    """
    profile = MODEL_FAMILIES.get(family_id)
    if profile is None:
        logger.warning(
            "Unknown model family '%s', falling back to '%s'. Valid: %s",
            family_id, DEFAULT_MODEL_FAMILY, ", ".join(MODEL_FAMILIES.keys()),
        )
        profile = MODEL_FAMILIES[DEFAULT_MODEL_FAMILY]
    return profile


# Superset of all stop markers across all families.
# Used for _strip_chat_markers() safety net — catches markers from ANY model family.
ALL_STOP_MARKERS: tuple[str, ...] = tuple(sorted(set(
    marker
    for profile in MODEL_FAMILIES.values()
    for marker in profile.stop_sequences
) | {
    "</s>", "<|end|>", "<|user|>", "<|assistant|>",
}))
