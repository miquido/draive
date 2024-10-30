from typing import Any, Protocol, runtime_checkable

from draive.instructions import Instruction
from draive.types import ImageContent, Multimodal

__all__ = [
    "ImageGenerator",
]


@runtime_checkable
class ImageGenerator(Protocol):
    async def __call__(
        self,
        *,
        instruction: Instruction | str,
        input: Multimodal | None,  # noqa: A002
        **extra: Any,
    ) -> ImageContent: ...
