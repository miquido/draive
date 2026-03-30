try:
    import anthropic  # pyright: ignore[reportUnusedImport]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "draive.anthropic requires the 'anthropic' extra."
        " Install via `pip install draive[anthropic]`."
    ) from exc


from draive.anthropic.client import Anthropic
from draive.anthropic.config import AnthropicConfig

__all__ = (
    "Anthropic",
    "AnthropicConfig",
)
