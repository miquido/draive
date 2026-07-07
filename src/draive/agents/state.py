from collections.abc import Collection, MutableMapping
from typing import Any, ClassVar, Self, final
from uuid import UUID

from haiway import Disposable, Meta, MetaValues, State, ctx

from draive.agents.types import (
    AgentMemoryRecalling,
    AgentMemoryRemembering,
    AgentThread,
)
from draive.models import ModelContext
from draive.models.types import ModelInput
from draive.steps.state import StepState
from draive.steps.step import Step, step

__all__ = ("AgentMemory",)


@final
class AgentMemory(State):
    """Pluggable recall/remember behavior for agent execution.

    An ``AgentMemory`` instance wraps a pair of async callables - one
    resolving the model context to use for a new turn (``recall``), and one
    persisting the context produced after a turn completes (``remember``).
    Both are scoped by ``AgentThread`` so a single memory instance can serve
    multiple concurrent conversation threads.

    Attributes
    ----------
    disabled : ClassVar[Self]
        No-op memory that passes the incoming input straight through on
        recall and discards context on remember. Used as the default
        ``memory`` for ``Agent`` factory methods.
    meta : Meta
        Additional metadata attached to the memory instance.
    """

    disabled: ClassVar[Self]  # defined after the class

    @classmethod
    def volatile(
        cls,
        *,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create an in-process memory keyed by conversation thread.

        Parameters
        ----------
        meta : Meta | MetaValues | None, default=None
            Additional metadata attached to the resulting memory instance.

        Returns
        -------
        Self
            A memory instance accumulating context per ``AgentThread`` in a
            plain in-memory mapping. State is lost when the process exits and
            is not shared across processes; intended for local development,
            tests, and single-process deployments.
        """
        memory: MutableMapping[UUID, ModelContext] = {}

        async def recall(
            thread: AgentThread,
            input: ModelInput,  # noqa: A002
            **extra: Any,
        ) -> ModelContext:
            return (*memory.get(thread.identifier, ()), input)

        async def remember(
            thread: AgentThread,
            context: ModelContext,
            **extra: Any,
        ) -> None:
            memory[thread.identifier] = context

        return cls(
            recalling=recall,
            remembering=remember,
            meta=Meta.of(meta),
        )

    _recalling: AgentMemoryRecalling
    _remembering: AgentMemoryRemembering
    meta: Meta

    def __init__(
        self,
        recalling: AgentMemoryRecalling,
        remembering: AgentMemoryRemembering,
        meta: Meta = Meta.empty,
    ) -> None:
        super().__init__(
            _recalling=recalling,
            _remembering=remembering,
            meta=meta,
        )

    def with_ctx(
        self,
        *ctx_state: State,
        disposables: Collection[Disposable] = (),
    ) -> Self:
        """Bind additional scoped context state and disposables to recall/remember.

        Parameters
        ----------
        *ctx_state : State
            State instances injected via ``ctx.updating`` for every recall and
            remember call, including when invoked through ``recall_step`` and
            ``remember_step``.
        disposables : Collection[Disposable], default=()
            Disposable resources entered for the duration of each recall and
            remember call; any state instances they produce are also made
            available in context.

        Returns
        -------
        Self
            A wrapped memory with additional scoped runtime context, or ``self``
            when no context state and disposables are provided.

        Notes
        -----
        Rationale: allows local dependency injection (e.g. a dedicated database
        session or model configuration) scoped to memory operations, without
        introducing global mutable state. Mirrors ``Step.with_ctx``.
        """
        recalling: AgentMemoryRecalling = self._recalling
        remembering: AgentMemoryRemembering = self._remembering

        if not ctx_state and not disposables:
            return self  # nothing to change...

        async def recall_with_ctx(
            thread: AgentThread,
            input: ModelInput,  # noqa: A002
            **extra: Any,
        ) -> ModelContext:
            async with ctx.disposables(*disposables):
                with ctx.updating(*ctx_state):
                    return await recalling(
                        thread=thread,
                        input=input,
                        **extra,
                    )

        async def remember_with_ctx(
            thread: AgentThread,
            context: ModelContext,
            **extra: Any,
        ) -> None:
            async with ctx.disposables(*disposables):
                with ctx.updating(*ctx_state):
                    await remembering(
                        thread=thread,
                        context=context,
                        **extra,
                    )

        return self.__class__(
            recalling=recall_with_ctx,
            remembering=remember_with_ctx,
            meta=self.meta,
        )

    async def recall(
        self,
        thread: AgentThread,
        input: ModelInput,  # noqa: A002
        **extra: Any,
    ) -> ModelContext:
        """Resolve the model context to use for a new turn.

        Parameters
        ----------
        thread : AgentThread
            Conversation thread the recalled context is scoped to.
        input : ModelInput
            Newly arrived input for the current turn, not yet part of any
            stored context.
        **extra : Any
            Additional implementation-specific arguments forwarded to the
            underlying recall callable.

        Returns
        -------
        ModelContext
            Context to use for the upcoming completion, typically prior
            history with ``input`` appended.
        """
        return await self._recalling(
            thread=thread,
            input=input,
            **extra,
        )

    def recall_step(
        self,
        input: ModelInput,  # noqa: A002
    ) -> Step:
        """Create a ``Step`` that replaces state context with recalled context.

        Parameters
        ----------
        input : ModelInput
            Newly arrived input for the current turn, forwarded to
            ``recall``.

        Returns
        -------
        Step
            A step reading the active ``AgentThread`` from context and
            replacing ``StepState.context`` with the recalled context.
        """

        @step
        async def recall(
            state: StepState,
        ) -> StepState:
            return state.replacing_context(
                await self._recalling(
                    thread=ctx.state(AgentThread),
                    input=input,
                )
            )

        return recall

    async def remember(
        self,
        thread: AgentThread,
        context: ModelContext,
        **extra: Any,
    ) -> None:
        """Persist context produced after a turn completes.

        Parameters
        ----------
        thread : AgentThread
            Conversation thread the persisted context is scoped to.
        context : ModelContext
            Full context accumulated for the turn, to be stored for future
            recall calls.
        **extra : Any
            Additional implementation-specific arguments forwarded to the
            underlying remember callable.

        Returns
        -------
        None
            Completes once the context has been persisted.
        """
        await self._remembering(
            thread=thread,
            context=context,
            **extra,
        )

    @property
    def remember_step(self) -> Step:
        """Create a ``Step`` that persists the current state context.

        Returns
        -------
        Step
            A step reading the active ``AgentThread`` from context, passing
            the current ``StepState.context`` to ``remember``, and leaving
            state unchanged.
        """

        @step
        async def remember(
            state: StepState,
        ) -> StepState:
            await self._remembering(
                thread=ctx.state(AgentThread),
                context=state.context,
            )

            return state

        return remember


async def _recall(
    thread: AgentThread,
    input: ModelInput,  # noqa: A002
    **extra: Any,
) -> ModelContext:
    return (input,)


async def _remember(
    thread: AgentThread,
    context: ModelContext,
    **extra: Any,
) -> None:
    pass


AgentMemory.disabled = AgentMemory(
    recalling=_recall,
    remembering=_remember,
)
