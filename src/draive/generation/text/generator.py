from collections.abc import Iterable, Sequence
from typing import Any, Protocol, runtime_checkable

from draive.lmm import AnyTool, Toolbox
from draive.types import Instruction, MultimodalContent, MultimodalContentConvertible

__all__ = [
    "TextGenerator",
]


@runtime_checkable
class TextGenerator(Protocol):
    async def __call__(
        self,
        *,
        instruction: Instruction | str,
        input: MultimodalContent | MultimodalContentConvertible,  # noqa: A002
        tools: Toolbox | Sequence[AnyTool] | None = None,
        examples: Iterable[tuple[MultimodalContent | MultimodalContentConvertible, str]]
        | None = None,
        **extra: Any,
    ) -> str: ...
