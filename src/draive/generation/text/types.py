from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from draive.models import ModelInstructions, Toolbox
from draive.multimodal import MultimodalContent

__all__ = ("TextGenerating",)


@runtime_checkable
class TextGenerating(Protocol):
    async def __call__(
        self,
        *,
        instructions: ModelInstructions,
        input: MultimodalContent,  # noqa: A002
        toolbox: Toolbox,
        examples: Iterable[tuple[MultimodalContent, str]],
        **extra: Any,
    ) -> str: ...
