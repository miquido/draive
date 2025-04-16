from typing import Any, Protocol, runtime_checkable

from draive.instructions import Instruction
from draive.multimodal import MediaContent, Multimodal

__all__ = ("ImageGenerating",)


@runtime_checkable
class ImageGenerating(Protocol):
    async def __call__(
        self,
        *,
        instruction: Instruction | str,
        input: Multimodal | None,  # noqa: A002
        **extra: Any,
    ) -> MediaContent: ...
