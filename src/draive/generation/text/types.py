from collections.abc import AsyncIterable, Iterable
from typing import Any, Protocol, runtime_checkable

from draive.instructions import Instruction
from draive.multimodal import MultimodalContent
from draive.prompts import Prompt
from draive.tools import Toolbox

__all__ = ("TextGenerating",)


@runtime_checkable
class TextGenerating(Protocol):
    async def __call__(
        self,
        *,
        instruction: Instruction | None,
        input: Prompt | MultimodalContent,  # noqa: A002
        toolbox: Toolbox,
        examples: Iterable[tuple[MultimodalContent, str]],
        stream: bool,
        **extra: Any,
    ) -> AsyncIterable[str] | str: ...
