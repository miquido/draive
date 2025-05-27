from collections.abc import Mapping, Sequence
from typing import NamedTuple, Protocol, runtime_checkable

from draive.commons import Meta
from draive.lmm import LMMContext
from draive.multimodal import MultimodalContent

__all__ = (
    "StageCondition",
    "StageContextTransforming",
    "StageException",
    "StageExecution",
    "StageMerging",
    "StageResultTransforming",
    "StageRouting",
    "StageState",
    "StageStateAccessing",
)


class StageState(NamedTuple):
    """Named tuple representing the state of a stage execution.

    This type encapsulates both the LMM context and result content that flows
    through stage processing pipelines. It provides a structured way to handle
    the dual nature of stage transformations where both context and result
    can be modified.

    Attributes
    ----------
    context : LMMContext
        The LMM context containing conversation history and elements.
    result : MultimodalContent
        The current result content from stage processing.
    """

    context: LMMContext
    result: MultimodalContent


@runtime_checkable
class StageExecution(Protocol):
    """Protocol for stage execution functions.

    Defines the signature for functions that execute a stage's processing logic.
    These functions transform both the LMM context and result value.

    Parameters
    ----------
    context : LMMContext
        The current LMM context containing conversation history.
    result : MultimodalContent
        The current result value from previous stages.

    Returns
    -------
    StageState
        A named tuple containing the updated context and result.
    """

    async def __call__(
        self,
        *,
        context: LMMContext,
        result: MultimodalContent,
    ) -> StageState: ...


@runtime_checkable
class StageMerging(Protocol):
    """Protocol for merging results from concurrent stage executions.

    Defines the signature for functions that combine results from multiple
    concurrent stage executions into a single context and result.

    Parameters
    ----------
    branches : Sequence[StageState | BaseException]
        Results from concurrent stage executions. Each element is either
        a successful result tuple or an exception from a failed execution.

    Returns
    -------
    StageState
        A single merged state containing context and result from all branch results.
    """

    async def __call__(
        self,
        *,
        branches: Sequence[StageState | BaseException],
    ) -> StageState: ...


@runtime_checkable
class StageCondition(Protocol):
    """Protocol for stage condition evaluation functions.

    Defines the signature for functions that evaluate conditions based on
    stage metadata, context, and result to determine control flow.

    Parameters
    ----------
    meta : Meta
        Metadata associated with the stage being evaluated.
    context : LMMContext
        The current LMM context.
    result : MultimodalContent
        The current result value.

    Returns
    -------
    bool
        True if the condition is met, False otherwise.
    """

    async def __call__(
        self,
        meta: Meta,
        context: LMMContext,
        result: MultimodalContent,
    ) -> bool: ...


@runtime_checkable
class StageResultTransforming(Protocol):
    """Protocol for result transformation functions.

    Defines the signature for functions that transform the result content
    without modifying the LMM context.

    Parameters
    ----------
    content : MultimodalContent
        The current result content to be transformed.

    Returns
    -------
    MultimodalContent
        The transformed result content.
    """

    async def __call__(
        self,
        content: MultimodalContent,
    ) -> MultimodalContent: ...


@runtime_checkable
class StageStateAccessing(Protocol):
    """Protocol for state accessing functions.

    Defines the signature for functions that access or modify state
    based on the current context and result without changing them.

    Parameters
    ----------
    context : LMMContext
        The current LMM context for state access.
    result : MultimodalContent
        The current result for state access.

    Returns
    -------
    None
        These functions perform side effects but don't return values.
    """

    async def __call__(
        self,
        context: LMMContext,
        result: MultimodalContent,
    ) -> None: ...


@runtime_checkable
class StageContextTransforming(Protocol):
    """Protocol for context transformation functions.

    Defines the signature for functions that transform the LMM context
    without modifying the result content.

    Parameters
    ----------
    context : LMMContext
        The current LMM context to be transformed.

    Returns
    -------
    LMMContext
        The transformed LMM context.
    """

    async def __call__(
        self,
        context: LMMContext,
    ) -> LMMContext: ...


@runtime_checkable
class StageRouting(Protocol):
    """Protocol for stage routing functions.

    Defines the signature for functions that select which stage to execute
    from a set of available options based on context and result.

    Parameters
    ----------
    context : LMMContext
        The current LMM context for routing decision.
    result : MultimodalContent
        The current result for routing decision.
    options : Mapping[str, Meta]
        Available routing options with their metadata.

    Returns
    -------
    str
        The key of the selected routing option.
    """

    async def __call__(
        self,
        context: LMMContext,
        result: MultimodalContent,
        options: Mapping[str, Meta],
    ) -> str: ...


class StageException(Exception):
    """Exception that can carry an execution result.

    This exception allows stage execution to fail while still providing
    a partial result that can be used by calling code.

    Parameters
    ----------
    *args : object
        Standard exception arguments.
    execution_result : MultimodalContent | None
        Optional partial result from the failed execution.

    Attributes
    ----------
    execution_result : MultimodalContent | None
        The partial result from execution, if any.
    """

    def __init__(
        self,
        *args: object,
        execution_result: MultimodalContent | None = None,
    ) -> None:
        super().__init__(*args)
        self.execution_result: MultimodalContent | None = execution_result
