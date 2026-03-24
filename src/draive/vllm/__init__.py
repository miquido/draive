try:
    import openai  # pyright: ignore[reportUnusedImport]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "draive.vllm requires the 'vllm' extra. Install via `pip install draive[vllm]`."
    ) from exc

from draive.vllm.client import VLLM
from draive.vllm.config import (
    VLLMChatConfig,
    VLLMEmbeddingConfig,
)

__all__ = (
    "VLLM",
    "VLLMChatConfig",
    "VLLMEmbeddingConfig",
)
