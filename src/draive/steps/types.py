from collections.abc import AsyncGenerator, Iterable
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
from draive.utils import ProcessingEvent

__all__ = (
    "StepConditionVerifying",
    "StepContextMutating",
    "StepException",
    "StepExecuting",
    "StepLoopConditionVerifying",
    "StepMerging",
    "StepOutputChunk",
    "StepProcessing",
    "StepRetriesExhausted",
    "StepStatePreserving",
    "StepStateRestoring",
    "StepStream",
    "StepTimeout",
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


class StepRetriesExhausted(StepException):
    """Raised when a retried step keeps failing after all permitted retries.

    The last failure is available as ``__cause__``.

    Parameters
    ----------
    *args : object
        Positional exception message arguments forwarded to ``Exception``.
    state : StepState
        Step state the failed attempts started from.
    attempts : int
        Total number of executions, including the initial one.
    meta : Meta | MetaValues | None, optional
        Additional metadata associated with the failure.
    """

    __slots__ = ("attempts",)

    def __init__(
        self,
        *args: object,
        state: StepState,
        attempts: int,
        meta: Meta | MetaValues | None = None,
    ) -> None:
        super().__init__(*args, state=state, meta=meta)
        self.attempts: int = attempts


class StepTimeout(StepException):
    """Raised when a step exceeds its execution time limit.

    Parameters
    ----------
    *args : object
        Positional exception message arguments forwarded to ``Exception``.
    state : StepState
        Latest step state observed before the timeout.
    timeout : float
        Configured time limit in seconds.
    meta : Meta | MetaValues | None, optional
        Additional metadata associated with the failure.
    """

    __slots__ = ("timeout",)

    def __init__(
        self,
        *args: object,
        state: StepState,
        timeout: float,
        meta: Meta | MetaValues | None = None,
    ) -> None:
        super().__init__(*args, state=state, meta=meta)
        self.timeout: float = timeout


StepOutputChunk = (
    MultimodalContentPart
    | ModelReasoningChunk
    | ModelToolRequest
    | ModelToolResponse
    | ProcessingEvent
)
StepStream = AsyncGenerator[StepOutputChunk | StepState]


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
