"""Registry of supported models for Loreguard NPC inference.

This module defines the models that are officially supported and tested
for NPC inference. Users can also specify custom model folders.
"""

from dataclasses import dataclass
from typing import Optional


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


# Supported models for NPC inference
# Ordered by recommendation (best first)
SUPPORTED_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="qwen3-4b-instruct",
        name="Qwen3 4B Instruct",
        filename="Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        size_gb=2.8,
        size_bytes=2_936_012_800,
        context_length=32768,
        url="https://huggingface.co/lmstudio-community/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        description="Best balance of speed and quality. Recommended for most users.",
        hardware="8GB RAM • Any GPU",
        recommended=True,
    ),
    ModelInfo(
        id="llama-3.2-3b-instruct",
        name="Llama 3.2 3B Instruct",
        filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        size_gb=2.0,
        size_bytes=2_019_377_120,
        context_length=131072,
        url="https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        description="Fast and capable. Great for quick responses.",
        hardware="8GB RAM • Any GPU",
        recommended=False,
    ),
    ModelInfo(
        id="rnj-1-instruct",
        name="RNJ-1 Instruct Q6_K",
        filename="rnj-1-instruct-Q6_K.gguf",
        size_gb=6.1,
        size_bytes=6_538_379_264,
        context_length=32768,
        url="https://huggingface.co/lmstudio-community/rnj-1-instruct-GGUF/resolve/main/rnj-1-instruct-Q6_K.gguf",
        description="Roleplay-focused model. Excellent for immersive NPCs.",
        hardware="12GB RAM • 8GB VRAM",
        recommended=False,
    ),
    ModelInfo(
        id="qwen3-8b",
        name="Qwen3 8B",
        filename="Qwen3-8B-Q4_K_M.gguf",
        size_gb=5.0,
        size_bytes=5_400_682_496,
        context_length=32768,
        url="https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
        description="Higher quality responses. Requires more VRAM.",
        hardware="12GB RAM • 6GB VRAM",
        recommended=False,
    ),
    ModelInfo(
        id="meta-llama-3-8b-instruct",
        name="Meta Llama 3 8B Instruct",
        filename="Meta-Llama-3-8B-Instruct.Q4_K_S.gguf",
        size_gb=4.9,
        size_bytes=5_268_701_184,
        context_length=8192,
        url="https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_S.gguf",
        description="Strong general-purpose model from Meta.",
        hardware="12GB RAM • 6GB VRAM",
        recommended=False,
    ),
    ModelInfo(
        id="gpt-oss-20b",
        name="GPT-OSS 20B",
        filename="gpt-oss-20b-MXFP4.gguf",
        size_gb=11.5,
        size_bytes=12_348_907_520,
        context_length=32768,
        url="https://huggingface.co/lmstudio-community/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-MXFP4.gguf",
        description="Large model for high-quality responses. Needs powerful hardware.",
        hardware="16GB RAM • 12GB VRAM",
        recommended=False,
    ),
    ModelInfo(
        id="qwen3-1.7b",
        name="Qwen3 1.7B",
        filename="Qwen3-1.7B-Q4_K_M.gguf",
        size_gb=1.1,
        size_bytes=1_191_362_560,
        context_length=32768,
        url="https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf",
        description="Very fast but lower quality. For testing only.",
        hardware="4GB RAM • CPU OK",
        experimental=True,
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
