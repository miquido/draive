from collections.abc import AsyncIterator, Sequence
from typing import (
    Any,
    Final,
    Literal,
    Protocol,
    runtime_checkable,
)

from draive.instructions import Instruction
from draive.multimodal import MultimodalContent
from draive.multimodal.meta import MetaContent
from draive.tools import Toolbox

__all__ = (
    "RealtimeOutputSelection",
    "RealtimeProcessing",
)


type RealtimeOutputSelection = (
    Sequence[Literal["text", "image", "audio"]] | Literal["auto", "text", "image", "audio"]
)


REALTIME_TURN_COMPLETE: Final[MetaContent] = MetaContent.of("REALTIME_TURN_COMPLETE")


@runtime_checkable
class RealtimeProcessing(Protocol):
    async def __call__(
        self,
        *,
        instruction: Instruction | None,
        input: AsyncIterator[MultimodalContent],  # noqa: A002
        tools: Toolbox,
        output: RealtimeOutputSelection,
        **extra: Any,
    ) -> AsyncIterator[MultimodalContent]: ...
