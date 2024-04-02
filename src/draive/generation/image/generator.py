from typing import Protocol, runtime_checkable

from draive.types import ImageContent

__all__ = [
    "ImageGenerator",
]


@runtime_checkable
class ImageGenerator(Protocol):
    async def __call__(
        self,
        *,
        instruction: str,
    ) -> ImageContent:
        ...
