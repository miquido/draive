from collections import OrderedDict
from collections.abc import Collection
from typing import Any, ClassVar, Self, final
from uuid import UUID

from haiway import Disposable, Meta, MetaValues, State, ctx

from draive.agents.types import (
    AgentException,
    AgentMemoryPreparing,
    AgentMemoryRecalling,
    AgentMemoryRemembering,
    AgentThread,
)
from draive.models import ModelContext
from draive.models.types import ModelInstructions
from draive.multimodal import Template, TemplatesRepository
from draive.steps.state import StepState
from draive.steps.step import Step, step

__all__ = ("AgentMemory",)


@final
class AgentMemory(State):
    """Pluggable prepare/recall/remember behavior for agent execution.

    An ``AgentMemory`` instance wraps async callables - one preparing the
    memory for a turn based on agent instructions (``prepare``), one
    resolving the complete model context to use for a new turn (``recall``),
    and one persisting the context produced after a turn completes
    (``remember``). All are scoped by ``AgentThread`` - the executing agent
    URI together with the conversation thread identifier - so a single memory
    instance can serve multiple agents and concurrent conversation threads,
    as long as the underlying implementation keys stored context by both.
    Built-in implementations do; custom callables ignoring the executing
    agent would mix contexts of different agents within a thread.

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
        """Create an in-process memory keyed by executing agent URI and thread.

        Context is stored as the latest snapshot per agent and thread: every
        remember replaces the entry's context with the provided one and
        recall reads it back whole. Agents may transform their context
        arbitrarily between recall and remember (compaction, summarization,
        replacement) - whatever is remembered becomes the next recalled
        context, exactly as provided. Concurrent remembers within the same
        agent and thread overwrite each other; the last writer wins.

        Parameters
        ----------
        initial: ModelContext, default=()
            Initial context for new agent-thread entries.
        threads_limit : int | None, default=None
            Maximum number of agent-thread entries retained at once. When
            exceeded, the least recently prepared or remembered entry is
            evicted together with its context. Preparing or remembering an
            entry refreshes its recency; recalling it does not. ``None``
            disables eviction - memory then grows unboundedly with the number
            of entries for the lifetime of the process.
        meta : Meta | MetaValues | None, default=None
            Additional metadata attached to the resulting memory instance.

        Returns
        -------
        Self
            A memory instance accumulating context per executing agent URI
            and conversation thread in a plain in-memory mapping. State is lost
            when the process exits and is not shared across processes;
            intended for local development, tests, and single-process
            deployments.
        """
        assert threads_limit is None or threads_limit > 0  # nosec: B101
        initial_context: ModelContext = tuple(initial)
        memory: OrderedDict[tuple[str, UUID], ModelContext] = OrderedDict()

        async def prepare(
            thread: AgentThread,
            instructions: ModelInstructions,
            **extra: Any,
        ) -> None:
            key: tuple[str, UUID] = (thread.agent_uri, thread.identifier)
            if key in memory:
                memory.move_to_end(key)

            else:
                memory[key] = initial_context

            while threads_limit is not None and len(memory) > threads_limit:
                memory.popitem(last=False)

        async def recall(
            thread: AgentThread,
            context: ModelContext,
            **extra: Any,
        ) -> ModelContext:
            return (*memory.get((thread.agent_uri, thread.identifier), initial_context), *context)

        async def remember(
            thread: AgentThread,
            context: ModelContext,
            **extra: Any,
        ) -> None:
            key: tuple[str, UUID] = (thread.agent_uri, thread.identifier)
            memory[key] = tuple(context)
            memory.move_to_end(key)

            while threads_limit is not None and len(memory) > threads_limit:
                memory.popitem(last=False)

        return cls(
            preparing=prepare,
            recalling=recall,
            remembering=remember,
            meta=Meta.of(meta),
        )

    _preparing: AgentMemoryPreparing
    _recalling: AgentMemoryRecalling
    _remembering: AgentMemoryRemembering
    meta: Meta

    def __init__(
        self,
        recalling: AgentMemoryRecalling,
        remembering: AgentMemoryRemembering,
        preparing: AgentMemoryPreparing | None = None,
        meta: Meta = Meta.empty,
    ) -> None:
        super().__init__(
            _preparing=preparing or _prepare,
            _recalling=recalling,
            _remembering=remembering,
            meta=meta,
        )

    def with_ctx(
        self,
        *ctx_state: State,
        disposables: Collection[Disposable] = (),
    ) -> Self:
        """Bind additional scoped context state and disposables to memory operations.

        Parameters
        ----------
        *ctx_state : State
            State instances injected via ``ctx.updating`` for every prepare,
            recall, and remember call, including when invoked through
            ``prepare_step``, ``recall_step``, and ``remember_step``.
        disposables : Collection[Disposable], default=()
            Disposable resources entered for the duration of each prepare,
            recall, and remember call; any state instances they produce are
            also made available in context. The same instances are re-entered
            for every operation, so they must support repeated use - one-shot
            disposables would fail on the second memory operation.

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
        preparing: AgentMemoryPreparing = self._preparing
        recalling: AgentMemoryRecalling = self._recalling
        remembering: AgentMemoryRemembering = self._remembering

        if not ctx_state and not disposables:
            return self  # nothing to change...

        async def prepare_with_ctx(
            thread: AgentThread,
            instructions: ModelInstructions,
            **extra: Any,
        ) -> None:
            async with ctx.disposables(*disposables):
                with ctx.updating(*ctx_state):
                    await preparing(
                        thread,
                        instructions,
                        **extra,
                    )

        async def recall_with_ctx(
            thread: AgentThread,
            context: ModelContext,
            **extra: Any,
        ) -> ModelContext:
            async with ctx.disposables(*disposables):
                with ctx.updating(*ctx_state):
                    return await recalling(
                        thread,
                        context,
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
            preparing=prepare_with_ctx,
            recalling=recall_with_ctx,
            remembering=remember_with_ctx,
            meta=self.meta,
        )

    async def prepare(
        self,
        thread: AgentThread,
        instructions: ModelInstructions,
        **extra: Any,
    ) -> None:
        """Prepare memory for a turn based on agent instructions.

        Called lazily before each recall. Implementations must be idempotent
        per agent and thread - only the memory implementation knows whether
        an agent-thread pair is already prepared.

        Parameters
        ----------
        thread : AgentThread
            Executing agent thread scope the preparation is applied to.
        instructions : ModelInstructions
            Resolved instructions of the agent utilizing the memory.
        **extra : Any
            Additional implementation-specific arguments forwarded to the
            underlying prepare callable.

        Returns
        -------
        None
            Completes once the memory is prepared for the agent and thread.
        """
        return await self._preparing(
            thread,
            instructions,
            **extra,
        )

    def prepare_step(
        self,
        instructions: Template | ModelInstructions,
        **extra: Any,
    ) -> Step:
        """Create a ``Step`` that prepares memory for the current turn.

        Parameters
        ----------
        instructions : Template | ModelInstructions
            Instructions of the agent utilizing the memory. ``Template``
            values are resolved through ``TemplatesRepository`` when the step
            executes, independently of the completion using the same
            instructions.
        **extra : Any
            Additional implementation-specific arguments forwarded to the
            underlying prepare callable.

        Returns
        -------
        Step
            A step reading the active ``AgentThread`` from context, preparing
            memory with the resolved instructions, and leaving state
            unchanged.

        Raises
        ------
        AgentException
            Raised when the step executes without an ``AgentThread`` bound in
            the current context.
        """

        @step
        async def prepare(
            state: StepState,
        ) -> StepState:
            resolved_instructions: str
            if isinstance(instructions, Template):
                resolved_instructions = await TemplatesRepository.resolve_str(instructions)

            else:
                resolved_instructions = instructions

            await self._preparing(
                _current_thread(),
                resolved_instructions,
                **extra,
            )
            return state

        return prepare

    async def recall(
        self,
        thread: AgentThread,
        context: ModelContext,
        **extra: Any,
    ) -> ModelContext:
        """Resolve the complete model context to use for a new turn.

        Parameters
        ----------
        thread : AgentThread
            Executing agent thread scope the recalled context belongs to.
        context : ModelContext
            Context accumulated so far for the current turn, not yet part of
            any stored context.
        **extra : Any
            Additional implementation-specific arguments forwarded to the
            underlying recall callable.

        Returns
        -------
        ModelContext
            Complete context to use for the upcoming completion, typically
            stored history followed by the provided turn context.
        """
        return await self._recalling(
            thread,
            context,
            **extra,
        )

    def recall_step(
        self,
        **extra: Any,
    ) -> Step:
        """Create a ``Step`` that replaces state context with recalled context.

        The current ``StepState.context`` - the context accumulated so far
        for the turn - is passed to ``recall`` and replaced with the result.
        Running recall twice within one pipeline duplicates stored history;
        that is a composition error.

        Parameters
        ----------
        **extra : Any
            Additional implementation-specific arguments forwarded to the
            underlying recall callable.

        Returns
        -------
        Step
            A step reading the active ``AgentThread`` from context and
            replacing ``StepState.context`` with the recalled context.

        Raises
        ------
        AgentException
            Raised when the step executes without an ``AgentThread`` bound in
            the current context.
        """

        @step
        async def recall(
            state: StepState,
        ) -> StepState:
            return state.replacing_context(
                await self._recalling(
                    _current_thread(),
                    state.context,
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
            Executing agent thread scope the persisted context belongs to.
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

    def remember_step(
        self,
        **extra: Any,
    ) -> Step:
        """Create a ``Step`` that persists the current state context.

        Parameters
        ----------
        **extra : Any
            Additional implementation-specific arguments forwarded to the
            underlying remember callable.

        Returns
        -------
        Step
            A step reading the active ``AgentThread`` from context, passing
            the current ``StepState.context`` to ``remember``, and leaving
            state unchanged.

        Raises
        ------
        AgentException
            Raised when the step executes without an ``AgentThread`` bound in
            the current context.
        """

        @step
        async def remember(
            state: StepState,
        ) -> StepState:
            await self._remembering(
                _current_thread(),
                state.context,
                **extra,
            )

            return state

        return remember


def _current_thread() -> AgentThread:
    # absence is checked explicitly to raise a domain-specific error instead of
    # whatever ctx.state raises for a type without defaultable fields
    if not ctx.contains_state(AgentThread):
        raise AgentException(
            "AgentThread is not available in the current context - memory operations require"
            " an active agent thread bound in scope, e.g. by running within an Agent."
        )

    return ctx.state(AgentThread)


async def _prepare(
    thread: AgentThread,
    instructions: ModelInstructions,
    **extra: Any,
) -> None:
    pass


async def _recall(
    thread: AgentThread,
    context: ModelContext,
    **extra: Any,
) -> ModelContext:
    return context


async def _remember(
    thread: AgentThread,
    context: ModelContext,
    **extra: Any,
) -> None:
    pass


AgentMemory.disabled = AgentMemory(
    preparing=_prepare,
    recalling=_recall,
    remembering=_remember,
)
