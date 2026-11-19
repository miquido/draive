try:
    import google.genai  # pyright: ignore[reportUnusedImport]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "draive.gemini requires the 'gemini' extra. Install via `pip install draive[gemini]`."
    ) from exc

from draive.gemini.client import Gemini
from draive.gemini.config import (
    GeminiConfig,
    GeminiDecisionConfig,
    GeminiEmbeddingConfig,
    GeminiSafetyConfig,
    GeminiSafetyThreshold,
)

__all__ = (
    "Gemini",
    "GeminiConfig",
    "GeminiDecisionConfig",
    "GeminiEmbeddingConfig",
    "GeminiSafetyConfig",
    "GeminiSafetyThreshold",
)
