from collections.abc import MutableSequence
from typing import Protocol, runtime_checkable

from draive.lmm import LMMContext, LMMContextElement
from draive.multimodal import MultimodalContent

__all__ = [
    "StageCondition",
    "StageContextProcessing",
    "StageProcessing",
    "StageResultProcessing",
]


@runtime_checkable
class StageResultProcessing(Protocol):
    async def __call__(
        self,
        content: MultimodalContent,
    ) -> MultimodalContent: ...


@runtime_checkable
class StageProcessing(Protocol):
    async def __call__(
        self,
        *,
        context: MutableSequence[LMMContextElement],
        result: MultimodalContent,
    ) -> MultimodalContent: ...


@runtime_checkable
class StageCondition(Protocol):
    async def __call__(
        self,
        context: LMMContext,
        result: MultimodalContent,
    ) -> bool: ...


@runtime_checkable
class StageContextProcessing(Protocol):
    async def __call__(
        self,
        context: MutableSequence[LMMContextElement],
    ) -> None: ...
