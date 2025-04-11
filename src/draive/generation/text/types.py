from collections.abc import AsyncIterable, Iterable
from typing import Any, Protocol, runtime_checkable

from draive.instructions import Instruction
from draive.multimodal import Multimodal
from draive.prompts import Prompt
from draive.tools import Tool, Toolbox

__all__ = ("TextGenerating",)


@runtime_checkable
class TextGenerating(Protocol):
    async def __call__(
        self,
        *,
        instruction: Instruction | str | None,
        input: Prompt | Multimodal,  # noqa: A002
        tools: Toolbox | Iterable[Tool] | None,
        examples: Iterable[tuple[Multimodal, str]] | None,
        stream: bool,
        **extra: Any,
    ) -> AsyncIterable[str] | str: ...
