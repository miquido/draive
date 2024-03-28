from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from draive.tools import Toolbox
from draive.types import MultimodalContent

__all__ = [
    "TextGenerator",
]


@runtime_checkable
class TextGenerator(Protocol):
    async def __call__(
        self,
        *,
        instruction: str,
        input: MultimodalContent,  # noqa: A002
        tools: Toolbox | None = None,
        examples: Iterable[tuple[MultimodalContent, str]] | None = None,
    ) -> str:
        ...
