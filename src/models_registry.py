"""Registry of supported models for Loreguard NPC inference.

This module defines the models that are officially supported and tested
for NPC inference. Users can also specify custom model folders.
"""

import platform
from dataclasses import dataclass, field
from typing import Optional


# HuggingFace organization for Beyond Logic Labs assets
HF_ORG = "beyond-logic-labs"


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


@dataclass
class ModelInfo:
    """Information about a supported model."""
    id: str                     # Unique identifier
    name: str                   # Display name
    filename: str               # GGUF filename (or folder name for MLX)
    size_gb: float              # Approximate size in GB
    size_bytes: int             # Exact size in bytes (for download progress)
    context_length: int         # Context window size
    url: str                    # Download URL (HuggingFace)
    description: str            # Short description
    hardware: str               # Hardware requirement hint
    recommended: bool = False   # Show as recommended
    experimental: bool = False  # Mark as experimental/lower quality
    is_mlx: bool = False        # True for MLX format (Apple Silicon only)
    requires_apple_silicon: bool = False  # Only show on Apple Silicon


@dataclass
class AdapterInfo:
    """Information about a LoRA adapter."""
    id: str                     # Unique identifier
    name: str                   # Display name
    filename: str               # GGUF filename
    size_mb: float              # Approximate size in MB
    size_bytes: int             # Exact size in bytes (for download progress)
    url: str                    # Download URL (HuggingFace)
    description: str            # Short description
    base_models: list[str]      # Compatible base model IDs
    recommended: bool = False   # Show as recommended


# Supported models for NPC inference
# Fine-tuned Loreguard models with multi-pass pipeline training
# Based on unsloth/Llama-3.1-8B-Instruct with Unsloth Dynamic (UD) quantization
# UD = per-layer optimized quantization for better accuracy at same VRAM
# Ordered by recommendation (best first)
SUPPORTED_MODELS: list[ModelInfo] = [
    # GGUF models with Unsloth Dynamic quantization (cross-platform, uses llama-server)
    ModelInfo(
        id="loreguard-vanilla-ud-q6k",
        name="Loreguard Vanilla UD Q6_K",
        filename="loreguard-vanilla-UD-Q6_K.gguf",
        size_gb=6.6,
        size_bytes=7_085_559_072,
        context_length=8192,
        url=f"https://huggingface.co/{HF_ORG}/loreguard-vanilla-gguf/resolve/main/loreguard-vanilla-UD-Q6_K.gguf",
        description="Recommended. Best quality/size balance.",
        hardware="12GB RAM • 8GB VRAM",
        recommended=True,
    ),
    ModelInfo(
        id="loreguard-vanilla-ud-q5km",
        name="Loreguard Vanilla UD Q5_K_M",
        filename="loreguard-vanilla-UD-Q5_K_M.gguf",
        size_gb=5.7,
        size_bytes=6_145_035_552,
        context_length=8192,
        url=f"https://huggingface.co/{HF_ORG}/loreguard-vanilla-gguf/resolve/main/loreguard-vanilla-UD-Q5_K_M.gguf",
        description="Good quality, smaller size.",
        hardware="10GB RAM • 6-8GB VRAM",
        recommended=False,
    ),
    ModelInfo(
        id="loreguard-vanilla-ud-q4km",
        name="Loreguard Vanilla UD Q4_K_M",
        filename="loreguard-vanilla-UD-Q4_K_M.gguf",
        size_gb=4.9,
        size_bytes=5_282_912_544,
        context_length=8192,
        url=f"https://huggingface.co/{HF_ORG}/loreguard-vanilla-gguf/resolve/main/loreguard-vanilla-UD-Q4_K_M.gguf",
        description="Best for 6GB VRAM. Smallest size.",
        hardware="8GB RAM • 6GB VRAM",
        recommended=False,
    ),
    ModelInfo(
        id="loreguard-vanilla-ud-q8",
        name="Loreguard Vanilla UD Q8_0",
        filename="loreguard-vanilla-UD-Q8_0.gguf",
        size_gb=8.5,
        size_bytes=9_177_550_624,
        context_length=8192,
        url=f"https://huggingface.co/{HF_ORG}/loreguard-vanilla-gguf/resolve/main/loreguard-vanilla-UD-Q8_0.gguf",
        description="Maximum quality. Requires more VRAM.",
        hardware="16GB RAM • 12GB VRAM",
        recommended=False,
    ),
]


def get_model_by_id(model_id: str) -> Optional[ModelInfo]:
    """Get a model by its ID."""
    for model in SUPPORTED_MODELS:
        if model.id == model_id:
            return model
    return None


def get_recommended_model() -> ModelInfo:
    """Get the recommended model."""
    for model in SUPPORTED_MODELS:
        if model.recommended:
            return model
    return SUPPORTED_MODELS[0]


def get_available_models(use_discovery: bool = True) -> list[ModelInfo]:
    """Get models available for the current platform.

    Args:
        use_discovery: If True, also fetch models from HuggingFace dynamically.
                      Falls back to static list on network errors.

    Filters out MLX models on non-Apple Silicon platforms.
    """
    available = []

    # Start with static models
    for model in SUPPORTED_MODELS:
        # Skip Apple Silicon-only models on other platforms
        if model.requires_apple_silicon and not is_apple_silicon():
            continue
        available.append(model)

    # Try dynamic discovery
    if use_discovery:
        try:
            from .hf_discovery import discover_models
            discovered = discover_models(use_cache=True)
            # Add discovered models that aren't already in the list
            existing_ids = {m.id for m in available}
            for model in discovered:
                if model.id not in existing_ids:
                    available.append(model)
        except Exception as e:
            # Silently fall back to static list on any error
            pass

    return available


def get_gguf_models() -> list[ModelInfo]:
    """Get only GGUF models (cross-platform)."""
    return [m for m in SUPPORTED_MODELS if not m.is_mlx]


def get_mlx_models() -> list[ModelInfo]:
    """Get only MLX models (Apple Silicon only)."""
    if not is_apple_silicon():
        return []
    return [m for m in SUPPORTED_MODELS if m.is_mlx]


# Supported LoRA adapters for pipeline-enhanced NPC inference
# Note: Merged models (SUPPORTED_MODELS) are recommended over base + adapter
# These adapters are for advanced users who want to use their own base model
SUPPORTED_ADAPTERS: list[AdapterInfo] = [
    AdapterInfo(
        id="loreguard-vanilla-lora",
        name="Loreguard Vanilla LoRA",
        filename="loreguard-vanilla.gguf",
        size_mb=160.0,
        size_bytes=167_787_680,
        url=f"https://huggingface.co/{HF_ORG}/loreguard-vanilla-lora-gguf/resolve/main/loreguard-vanilla.gguf",
        description="LoRA adapter for custom base models. Use with Llama 3.1 8B.",
        base_models=["llama-3.1-8b-instruct"],
        recommended=False,
    ),
]


def get_adapter_by_id(adapter_id: str) -> Optional[AdapterInfo]:
    """Get an adapter by its ID."""
    for adapter in SUPPORTED_ADAPTERS:
        if adapter.id == adapter_id:
            return adapter
    return None


def get_recommended_adapter() -> Optional[AdapterInfo]:
    """Get the recommended adapter."""
    for adapter in SUPPORTED_ADAPTERS:
        if adapter.recommended:
            return adapter
    return SUPPORTED_ADAPTERS[0] if SUPPORTED_ADAPTERS else None


def get_adapters_for_model(model_id: str) -> list[AdapterInfo]:
    """Get all adapters compatible with a given base model."""
    compatible = []
    for adapter in SUPPORTED_ADAPTERS:
        if model_id in adapter.base_models:
            compatible.append(adapter)
    return compatible


# NLI (Natural Language Inference) model for fact verification
# This is a HuggingFace transformer model, not a GGUF file
# Downloaded automatically via the transformers library to ~/.cache/huggingface/hub/
@dataclass
class NLIModelInfo:
    """Information about the NLI model for fact verification."""
    id: str                  # HuggingFace model ID
    name: str                # Display name
    size_gb: float           # Approximate size in GB
    description: str         # Short description


NLI_MODEL = NLIModelInfo(
    id="roberta-large-mnli",
    name="RoBERTa Large MNLI",
    size_gb=1.4,
    description="RoBERTa model for Natural Language Inference. Verifies NPC claims against knowledge base.",
)
