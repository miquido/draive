from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from draive.lmm import Toolbox
from draive.types import Instruction, MultimodalContent, MultimodalContentElement

__all__ = [
    "TextGenerator",
]


@runtime_checkable
class TextGenerator(Protocol):
    async def __call__(
        self,
        *,
        instruction: Instruction | str,
        input: MultimodalContent | MultimodalContentElement,  # noqa: A002
        tools: Toolbox | None = None,
        examples: Iterable[tuple[MultimodalContent | MultimodalContentElement, str]] | None = None,
        **extra: Any,
    ) -> str: ...
