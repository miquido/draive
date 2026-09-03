try:
    # mistralai is a namespace package, bare `import mistralai` succeeds without the client
    import mistralai.client  # pyright: ignore[reportUnusedImport]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "draive.mistral requires the 'mistral' extra. Install via `pip install draive[mistral]`."
    ) from exc

from draive.mistral.client import Mistral
from draive.mistral.config import MistralChatConfig, MistralEmbeddingConfig, MistralModerationConfig

__all__ = (
    "Mistral",
    "MistralChatConfig",
    "MistralEmbeddingConfig",
    "MistralModerationConfig",
)
