from collections.abc import AsyncIterable, Iterable
from typing import Protocol, runtime_checkable

from haiway import Meta, MetaValues

from draive.models import (
    ModelContext,
    ModelReasoningChunk,
    ModelToolRequest,
    ModelToolResponse,
)
from draive.multimodal import MultimodalContentPart
from draive.steps.state import StepState
from draive.tools import ToolEvent

__all__ = (
    "StepConditionVerifying",
    "StepContextMutating",
    "StepException",
    "StepExecuting",
    "StepLoopConditionVerifying",
    "StepMerging",
    "StepOutputChunk",
    "StepProcessing",
    "StepStatePreserving",
    "StepStateRestoring",
    "StepStream",
)


class StepException(Exception):
    """Base exception raised by step execution.

    This exception carries the originating step state and optional metadata so
    callers can recover structured context without parsing error messages.

    Parameters
    ----------
    *args : object
        Positional exception message arguments forwarded to ``Exception``.
    state : StepState
        Step state captured at the failure point.
    meta : Meta | MetaValues | None, optional
        Additional metadata associated with the failure. ``None`` produces an
        empty :class:`haiway.Meta` value.

    Attributes
    ----------
    state : StepState
        Step state captured at the failure point.
    meta : Meta
        Normalized metadata attached to the failure context.
    """

    __slots__ = (
        "meta",
        "state",
    )

    def __init__(
        self,
        *args: object,
        state: StepState,
        meta: Meta | MetaValues | None = None,
    ) -> None:
        """Initialize a step exception with structured failure context.

        Parameters
        ----------
        *args : object
            Positional exception message arguments forwarded to ``Exception``.
        state : StepState
            Step state captured at the failure point.
        meta : Meta | MetaValues | None, optional
            Additional metadata associated with the failure.
        """
        super().__init__(*args)
        self.state: StepState = state
        self.meta: Meta = Meta.of(meta)


StepOutputChunk = (
    MultimodalContentPart | ModelReasoningChunk | ModelToolRequest | ModelToolResponse | ToolEvent
)
StepStream = AsyncIterable[StepOutputChunk | StepState]


@runtime_checkable
class StepExecuting(Protocol):
    def __call__(
        self,
        state: StepState,
    ) -> StepStream: ...


@runtime_checkable
class StepProcessing(Protocol):
    async def __call__(
        self,
        state: StepState,
    ) -> StepState: ...


@runtime_checkable
class StepConditionVerifying(Protocol):
    async def __call__(
        self,
        state: StepState,
    ) -> bool: ...


@runtime_checkable
class StepLoopConditionVerifying(Protocol):
    async def __call__(
        self,
        state: StepState,
        iteration: int,
    ) -> bool: ...


@runtime_checkable
class StepContextMutating(Protocol):
    async def __call__(
        self,
        context: ModelContext,
    ) -> ModelContext: ...


@runtime_checkable
class StepMerging(Protocol):
    async def __call__(
        self,
        branches: Iterable[StepState],
    ) -> StepState: ...


@runtime_checkable
class StepStatePreserving(Protocol):
    async def __call__(
        self,
        state: StepState,
    ) -> None: ...


@runtime_checkable
class StepStateRestoring(Protocol):
    async def __call__(
        self,
    ) -> StepState: ...
