try:
    import ollama  # pyright: ignore[reportUnusedImport]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "draive.ollama requires the 'ollama' extra. Install via `pip install draive[ollama]`."
    ) from exc

from draive.ollama.client import Ollama
from draive.ollama.config import OllamaChatConfig, OllamaEmbeddingConfig

__all__ = (
    "Ollama",
    "OllamaChatConfig",
    "OllamaEmbeddingConfig",
)
