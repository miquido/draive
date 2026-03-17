from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from draive.models import ModelInstructions
from draive.multimodal import Multimodal
from draive.tools import Toolbox

__all__ = ("TextGenerating",)


@runtime_checkable
class TextGenerating(Protocol):
    async def __call__(
        self,
        *,
        instructions: ModelInstructions,
        input: Multimodal,  # noqa: A002
        toolbox: Toolbox,
        examples: Iterable[tuple[Multimodal, str]],
        **extra: Any,
    ) -> str: ...
