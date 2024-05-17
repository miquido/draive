from typing import Any, Protocol, runtime_checkable

from draive.types import ImageContent, Instruction, MultimodalContent, MultimodalContentElement

__all__ = [
    "ImageGenerator",
]


@runtime_checkable
class ImageGenerator(Protocol):
    async def __call__(
        self,
        *,
        instruction: Instruction | str,
        input: MultimodalContent | MultimodalContentElement | None = None,  # noqa: A002
        **extra: Any,
    ) -> ImageContent: ...
