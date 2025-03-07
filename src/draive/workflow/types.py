from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from draive.lmm import LMMContext
from draive.multimodal import MultimodalContent

__all__ = [
    "StageCondition",
    "StageContextProcessing",
    "StageMerging",
    "StageProcessing",
    "StageResultProcessing",
]


@runtime_checkable
class StageProcessing(Protocol):
    async def __call__(
        self,
        *,
        context: LMMContext,
        result: MultimodalContent,
    ) -> tuple[LMMContext, MultimodalContent]: ...


@runtime_checkable
class StageMerging(Protocol):
    async def __call__(
        self,
        *,
        branches: Sequence[tuple[LMMContext, MultimodalContent] | BaseException],
    ) -> tuple[LMMContext, MultimodalContent]: ...


@runtime_checkable
class StageCondition(Protocol):
    async def __call__(
        self,
        context: LMMContext,
        result: MultimodalContent,
    ) -> bool: ...


@runtime_checkable
class StageResultProcessing(Protocol):
    async def __call__(
        self,
        content: MultimodalContent,
    ) -> MultimodalContent: ...


@runtime_checkable
class StageContextProcessing(Protocol):
    async def __call__(
        self,
        context: LMMContext,
    ) -> LMMContext: ...
