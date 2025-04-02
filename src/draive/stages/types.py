from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from draive.lmm import LMMContext
from draive.multimodal import MultimodalContent

__all__ = (
    "StageCondition",
    "StageContextTransforming",
    "StageException",
    "StageExecution",
    "StageMerging",
    "StageResultTransforming",
    "StageStateAccessing",
)


@runtime_checkable
class StageExecution(Protocol):
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
class StageResultTransforming(Protocol):
    async def __call__(
        self,
        content: MultimodalContent,
    ) -> MultimodalContent: ...


@runtime_checkable
class StageStateAccessing(Protocol):
    async def __call__(
        self,
        context: LMMContext,
        result: MultimodalContent,
    ) -> None: ...


@runtime_checkable
class StageContextTransforming(Protocol):
    async def __call__(
        self,
        context: LMMContext,
    ) -> LMMContext: ...


class StageException(Exception):
    def __init__(
        self,
        *args: object,
        execution_result: MultimodalContent | None = None,
    ) -> None:
        super().__init__(*args)
        self.execution_result: MultimodalContent | None = execution_result
