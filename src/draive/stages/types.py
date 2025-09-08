from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, Self, cast, final, overload, runtime_checkable

from haiway import Meta, MetaValues, MissingState, State

from draive.models import ModelContext, ModelOutput
from draive.multimodal import Multimodal, MultimodalContent
from draive.parameters import DataModel
from draive.utils import Memory

__all__ = (
    "StageCacheKeyMaking",
    "StageCacheReading",
    "StageCacheWriting",
    "StageConditioning",
    "StageContextTransforming",
    "StageException",
    "StageExecution",
    "StageLoopConditioning",
    "StageMemory",
    "StageMerging",
    "StageResultTransforming",
    "StageRouting",
    "StageState",
    "StageStateAccessing",
)


@final
class StageState:
    """
    Immutable state container for stage execution.

    StageState encapsulates the complete execution state of a stage, including
    the model context, current result, and any associated state models. It provides
    methods for creating new states with updated values while maintaining immutability.

    The state includes:
    - context: The model conversation context (input/output pairs, tool calls, etc.)
    - result: The current multimodal result content
    - state: Dictionary of state models indexed by their types
    """

    @classmethod
    def of(
        cls,
        *state: DataModel | State,
        context: ModelContext,
        result: Multimodal | None = None,
    ) -> Self:
        """
        Create a new StageState from individual components.

        Parameters
        ----------
        *state : DataModel | State
            State models to include in the stage state.
        context : ModelContext
            The model context.
        result : Multimodal | None
            The result content, or None for empty content.

        Returns
        -------
        Self
            A new StageState instance.
        """
        return cls(
            context=context,
            result=MultimodalContent.of(result) if result is not None else MultimodalContent.empty,
            state={type(element): element for element in state},
        )

    __slots__ = (
        "_state",
        "context",
        "result",
    )

    def __init__(
        self,
        *,
        context: ModelContext,
        result: MultimodalContent,
        state: Mapping[type[DataModel] | type[State], DataModel | State],
    ) -> None:
        """
        Initialize a new StageState instance.

        Parameters
        ----------
        context : ModelContext
            The model context.
        result : MultimodalContent
            The current result content.
        state : Mapping[type[DataModel] | type[State], DataModel | State]
            Dictionary mapping state types to their instances.
        """

        assert not context or isinstance(context[-1], ModelOutput)  # nosec: B101
        self.context: ModelContext
        object.__setattr__(
            self,
            "context",
            context if isinstance(context, tuple) else tuple(context),
        )
        self.result: MultimodalContent
        object.__setattr__(
            self,
            "result",
            result,
        )
        self._state: Mapping[type[DataModel] | type[State], DataModel | State]
        object.__setattr__(
            self,
            "_state",
            state,
        )

    def __setattr__(
        self,
        name: str,
        value: Any,
    ) -> Any:
        raise AttributeError(
            f"Can't modify immutable {self.__class__.__qualname__},"
            f" attribute - '{name}' cannot be modified"
        )

    def __delattr__(
        self,
        name: str,
    ) -> None:
        raise AttributeError(
            f"Can't modify immutable {self.__class__.__qualname__},"
            f" attribute - '{name}' cannot be deleted"
        )

    @overload
    def get[StateType: DataModel | State](
        self,
        state: type[StateType],
        /,
    ) -> StateType | None: ...

    @overload
    def get[StateType: DataModel | State](
        self,
        state: type[StateType],
        /,
        *,
        required: Literal[True],
    ) -> StateType: ...

    @overload
    def get[StateType: DataModel | State](
        self,
        state: type[StateType],
        /,
        *,
        default: StateType,
    ) -> StateType: ...

    def get[StateType: DataModel | State](
        self,
        state: type[StateType],
        /,
        *,
        default: StateType | None = None,
        required: bool = False,
    ) -> StateType | None:
        """
        Retrieve a state model by its type.

        Parameters
        ----------
        state : type[StateType]
            The type of state model to retrieve.
        default : StateType | None
            Default value to return if the state is not found.
        required : bool
            If True, raises MissingState when the state is not found.

        Returns
        -------
        StateType | None
            The state model if found, the default value, or None.

        Raises
        ------
        MissingState
            When required=True and the state is not found.
        """

        current: StateType | None = cast(StateType | None, self._state.get(state))
        if current is not None:
            return current

        elif default is not None:
            return default

        elif required:
            raise MissingState(f"{state.__qualname__} is not available within stage state")

        else:
            return None

    def updated(
        self,
        *state: DataModel | State,
        context: ModelContext | None = None,
        result: Multimodal | None = None,
    ) -> Self:
        """
        Create a new StageState with updated values.

        Returns a new StageState instance with the specified values updated
        while preserving existing values for unspecified parameters.

        Parameters
        ----------
        *state : DataModel | State
            State models to add or update.
        context : ModelContext | None
            New context to use, or None to preserve current context.
        result : Multimodal | None
            New result to use, or None to preserve current result.

        Returns
        -------
        Self
            A new StageState instance with the updated values.
        """

        return self.__class__(
            state={**self._state, **{type(element): element for element in state}}
            if state
            else self._state,
            context=context if context is not None else self.context,
            result=MultimodalContent.of(result) if result is not None else self.result,
        )

    def merged(
        self,
        other: Self,
        /,
    ) -> Self:
        """
        Create a new StageState by merging with another StageState.

        Combines the state models, concatenates the contexts, and merges
        the results from both StageState instances.

        Parameters
        ----------
        other : Self
            The other StageState to merge with.

        Returns
        -------
        Self
            A new StageState containing the merged data.
        """

        return self.__class__(
            state={**self._state, **other._state},
            context=(*self.context, *other.context),
            result=MultimodalContent.of(self.result, other.result),
        )


class StageException(Exception):
    """
    Exception that preserves stage state when an error occurs.

    This exception type allows stage execution errors to carry the current
    stage state, enabling recovery or inspection of the state at the time
    of failure.
    """

    def __init__(
        self,
        *args: object,
        state: StageState,
        meta: Meta | MetaValues | None = None,
    ) -> None:
        """
        Initialize a new StageException.

        Parameters
        ----------
        *args : object
            Exception arguments passed to the parent Exception class.
        state : StageState
            The stage state at the time the exception occurred.

        meta : Meta | MetaValues | None
            Additional exception metadata.

        """
        super().__init__(*args)
        self.state: StageState = state
        self.meta: Meta = Meta.of(meta)


@runtime_checkable
class StageExecution(Protocol):
    """
    Protocol defining the interface for stage execution functions.

    Any callable that accepts a StageState and returns a StageState
    (potentially modified) conforms to this protocol. This is the core
    execution interface for all stages.
    """

    async def __call__(
        self,
        *,
        state: StageState,
    ) -> StageState:
        """Execute the stage with the given state and return the updated state."""
        ...


@runtime_checkable
class StageMerging(Protocol):
    """
    Protocol for merging results from concurrent stage executions.

    Defines functions that can combine multiple stage execution results
    (which may include exceptions) into a single consolidated result.
    Used by Stage.concurrent() to merge parallel execution results.
    """

    async def __call__(
        self,
        *,
        branches: Sequence[StageState | StageException],
    ) -> StageState:
        """Merge multiple stage execution results into a single state."""
        ...


@runtime_checkable
class StageConditioning(Protocol):
    """
    Protocol for stage condition evaluation functions.

    Defines functions that evaluate whether a condition is met based on
    the current stage state. Used by Stage.when() for conditional execution.
    """

    async def __call__(
        self,
        *,
        state: StageState,
    ) -> bool:
        """Evaluate whether a condition is met based on the stage state."""
        ...


@runtime_checkable
class StageLoopConditioning(Protocol):
    """
    Protocol for stage loop condition evaluation functions.

    Defines functions that determine whether to continue looping based on
    the current stage state and iteration number. Used by Stage.loop() to
    control loop termination.
    """

    async def __call__(
        self,
        *,
        state: StageState,
        iteration: int,
    ) -> bool:
        """Evaluate whether to continue looping based on state and iteration."""
        ...


@runtime_checkable
class StageStateAccessing(Protocol):
    """
    Protocol for functions that access stage state without modification.

    Defines functions that can read or inspect stage state but do not
    return any value. Useful for logging, monitoring, or side-effect
    operations that need access to the current state.
    """

    async def __call__(
        self,
        *,
        state: StageState,
    ) -> None:
        """Access or inspect the stage state without modification."""
        ...


@runtime_checkable
class StageContextTransforming(Protocol):
    """
    Protocol for model context transformation functions.

    Defines functions that can transform a `ModelContext` into a new context.
    Used by `Stage.transform_context()` to modify conversation context while
    preserving the result.
    """

    async def __call__(
        self,
        context: ModelContext,
    ) -> ModelContext:
        """Transform a `ModelContext` into a new context."""
        ...


@runtime_checkable
class StageResultTransforming(Protocol):
    """
    Protocol for result transformation functions.

    Defines functions that can transform multimodal content into new content.
    Used by Stage.transform_result() to modify stage results while preserving
    the context.
    """

    async def __call__(
        self,
        result: MultimodalContent,
    ) -> MultimodalContent:
        """Transform multimodal content into new content."""
        ...


@runtime_checkable
class StageRouting(Protocol):
    """
    Protocol for stage routing functions.

    Defines functions that can select one stage from multiple options based on
    the current state and available stage metadata. Used by Stage.router() to
    dynamically choose which stage to execute.
    """

    async def __call__(
        self,
        *,
        state: StageState,
        options: Mapping[str, Meta],
    ) -> str:
        """Select a stage option based on state and available metadata."""
        ...


type StageMemory = Memory[StageState, StageState]


class StageCacheKeyMaking[Key](Protocol):
    """Callable that constructs a cache key for a given stage state."""

    def __call__(
        self,
        *,
        state: StageState,
    ) -> Key: ...


class StageCacheReading[Key](Protocol):
    """Async callable that retrieves a cached state for the given key."""

    async def __call__(
        self,
        key: Key,
    ) -> StageState | None: ...


class StageCacheWriting[Key](Protocol):
    """Async callable that stores a stage state for the given key."""

    async def __call__(
        self,
        key: Key,
        value: StageState,
    ) -> None: ...
