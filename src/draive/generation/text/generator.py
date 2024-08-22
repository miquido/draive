from collections.abc import Iterable, Sequence
from typing import Any, Protocol, runtime_checkable

from draive.instructions import Instruction
from draive.lmm import AnyTool, Toolbox
from draive.types import MultimodalContent, MultimodalContentConvertible

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
        prefill: str | None = None,
        tools: Toolbox | Sequence[AnyTool] | None = None,
        examples: Iterable[tuple[MultimodalContent | MultimodalContentConvertible, str]]
        | None = None,
        **extra: Any,
    ) -> str: ...
