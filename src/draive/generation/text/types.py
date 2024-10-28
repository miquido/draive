from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from draive.instructions import Instruction
from draive.lmm import AnyTool, Toolbox
from draive.types import Multimodal

__all__ = [
    "TextGenerator",
]


@runtime_checkable
class TextGenerator(Protocol):
    async def __call__(
        self,
        *,
        instruction: Instruction | str | None,
        input: Multimodal,  # noqa: A002
        prefill: str | None,
        tools: Toolbox | Iterable[AnyTool] | None,
        examples: Iterable[tuple[Multimodal, str]] | None,
        **extra: Any,
    ) -> str: ...
