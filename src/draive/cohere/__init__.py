try:
    import cohere  # pyright: ignore[reportUnusedImport]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "draive.cohere requires the 'cohere' extra. Install via `pip install draive[cohere]`."
    ) from exc

from draive.cohere.client import Cohere
from draive.cohere.config import CohereImageEmbeddingConfig, CohereTextEmbeddingConfig

__all__ = (
    "Cohere",
    "CohereImageEmbeddingConfig",
    "CohereTextEmbeddingConfig",
)
