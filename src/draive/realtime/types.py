from collections.abc import AsyncIterator, Sequence
from typing import (
    Any,
    Literal,
    Protocol,
    runtime_checkable,
)

from draive.instructions import Instruction
from draive.multimodal import MultimodalContent
from draive.tools import Toolbox

__all__ = (
    "RealtimeOutputSelection",
    "RealtimeProcessing",
)


type RealtimeOutputSelection = Sequence[Literal["text", "audio"]] | Literal["auto", "text", "audio"]


@runtime_checkable
class RealtimeProcessing(Protocol):
    async def __call__(
        self,
        *,
        instruction: Instruction | None,
        input_stream: AsyncIterator[MultimodalContent],
        toolbox: Toolbox,
        output: RealtimeOutputSelection,
        **extra: Any,
    ) -> AsyncIterator[MultimodalContent]: ...
