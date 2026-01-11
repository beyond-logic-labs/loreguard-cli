"""Registry of supported models for Loreguard NPC inference.

This module defines the models that are officially supported and tested
for NPC inference. Users can also specify custom model folders.
"""

from dataclasses import dataclass
from typing import Optional


# HuggingFace organization for Beyond Logic Labs assets
HF_ORG = "beyond-logic-labs"


@dataclass
class ModelInfo:
    """Information about a supported model."""
    id: str                     # Unique identifier
    name: str                   # Display name
    filename: str               # GGUF filename
    size_gb: float              # Approximate size in GB
    size_bytes: int             # Exact size in bytes (for download progress)
    context_length: int         # Context window size
    url: str                    # Download URL (HuggingFace)
    description: str            # Short description
    hardware: str               # Hardware requirement hint
    recommended: bool = False   # Show as recommended
    experimental: bool = False  # Mark as experimental/lower quality


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
# Ordered by recommendation (best first)
SUPPORTED_MODELS: list[ModelInfo] = [
    # Loreguard Vanilla - Generic NPC pipeline (citations, multi-pass)
    ModelInfo(
        id="loreguard-vanilla-q4k",
        name="Loreguard Vanilla Q4_K",
        filename="Llama-3.1-8B-loreguard-vanilla-Q4_K.gguf",
        size_gb=4.6,
        size_bytes=4_920_739_200,
        context_length=8192,
        url=f"https://huggingface.co/{HF_ORG}/loreguard-vanilla-gguf/resolve/main/Llama-3.1-8B-loreguard-vanilla-Q4_K.gguf",
        description="Best for 6GB VRAM. Trained on Loreguard NPC pipeline.",
        hardware="8GB RAM • 6GB VRAM",
        recommended=True,
    ),
    ModelInfo(
        id="loreguard-vanilla-q5km",
        name="Loreguard Vanilla Q5_K_M",
        filename="Llama-3.1-8B-loreguard-vanilla-Q5_K_M.gguf",
        size_gb=5.3,
        size_bytes=5_732_992_384,
        context_length=8192,
        url=f"https://huggingface.co/{HF_ORG}/loreguard-vanilla-gguf/resolve/main/Llama-3.1-8B-loreguard-vanilla-Q5_K_M.gguf",
        description="Better quality. Fits 6GB with Q4 KV cache (-ctk q4_0 -ctv q4_0).",
        hardware="10GB RAM • 6-8GB VRAM",
        recommended=False,
    ),
    ModelInfo(
        id="loreguard-vanilla-q6k",
        name="Loreguard Vanilla Q6_K",
        filename="Llama-3.1-8B-loreguard-vanilla-Q6_K.gguf",
        size_gb=6.1,
        size_bytes=6_596_011_392,
        context_length=8192,
        url=f"https://huggingface.co/{HF_ORG}/loreguard-vanilla-gguf/resolve/main/Llama-3.1-8B-loreguard-vanilla-Q6_K.gguf",
        description="Highest quality. Requires 8GB+ VRAM.",
        hardware="12GB RAM • 8GB VRAM",
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
