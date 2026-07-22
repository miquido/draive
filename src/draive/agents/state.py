from collections import OrderedDict
from collections.abc import Collection
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

    Each instance is intended to serve exactly one agent. State is scoped
    per conversation thread only - not per agent - so sharing one instance
    between multiple agents would mix their contexts within a thread. Create
    a separate memory instance for each agent.

    Attributes
    ----------
    meta : Meta
        Additional metadata attached to the memory instance.
    """

    disabled: ClassVar[Self]  # defined after the class

    @classmethod
    def volatile(
        cls,
        initial: ModelContext = (),
        *,
        threads_limit: int | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create an in-process memory keyed by conversation thread.

        Context is stored as the latest snapshot per thread: every remember
        replaces the thread's context with the provided one and recall reads
        it back whole. Agents may transform their context arbitrarily between
        recall and remember (compaction, summarization, replacement) -
        whatever is remembered becomes the next recalled context, exactly as
        provided. Concurrent remembers within the same thread overwrite each
        other; the last writer wins.

        Parameters
        ----------
        initial: ModelContext, default=()
            Initial context for threads.
        threads_limit : int | None, default=None
            Maximum number of threads retained at once. When exceeded, the
            least recently used thread is evicted together with its context.
            ``None`` disables eviction - memory then grows unboundedly with
            the number of threads for the lifetime of the process.
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
        assert threads_limit is None or threads_limit > 0  # nosec: B101
        memory: OrderedDict[UUID, ModelContext] = OrderedDict()

        async def recall(
            thread: AgentThread,
            input: ModelInput,  # noqa: A002
            **extra: Any,
        ) -> ModelContext:
            recalled: ModelContext = memory.get(thread.identifier, initial)
            if thread.identifier in memory:
                memory.move_to_end(thread.identifier)
            return (*recalled, input)

        async def remember(
            thread: AgentThread,
            context: ModelContext,
            **extra: Any,
        ) -> None:
            # the provided context replaces the thread snapshot as-is
            memory[thread.identifier] = tuple(context)
            memory.move_to_end(thread.identifier)
            while threads_limit is not None and len(memory) > threads_limit:
                memory.popitem(last=False)  # evict least recently used thread

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
                        thread,
                        input,
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
                        thread,
                        context,
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
            thread,
            input,
            **extra,
        )

    def recall_step(
        self,
        input: ModelInput,  # noqa: A002
        **extra: Any,
    ) -> Step:
        """Create a ``Step`` that replaces state context with recalled context.

        Parameters
        ----------
        input : ModelInput
            Newly arrived input for the current turn, forwarded to
            ``recall``.
        **extra : Any
            Additional implementation-specific arguments forwarded to the
            underlying recall callable.

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
                    ctx.state(AgentThread),
                    input,
                    **extra,
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
            thread,
            context,
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
                ctx.state(AgentThread),
                state.context,
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
