from collections.abc import AsyncIterable, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol, Self, final, runtime_checkable
from uuid import UUID, uuid4

from haiway import Default, Meta, MetaValues, State

from draive.models.types import ModelContext, ModelInstructions
from draive.multimodal import Multimodal, MultimodalContent, MultimodalContentPart
from draive.tools import Tool
from draive.utils import ProcessingEvent

__all__ = (
    "AgentException",
    "AgentExecuting",
    "AgentIdentity",
    "AgentMemoryPreparing",
    "AgentMemoryRecalling",
    "AgentMemoryRemembering",
    "AgentMessage",
    "AgentThread",
    "AgentUnavailable",
)


class AgentException(Exception):
    """Base exception raised by agent helpers.

    Raises
    ------
    AgentException
        Raised when agent operations fail with a framework-level error.
    """


@final
class AgentUnavailable(AgentException):
    """Raised when a referenced agent cannot be accessed.

    Raises
    ------
    AgentUnavailable
        Raised when an agent reference points to an unavailable or unknown agent.
    """


@final
class AgentIdentity(State, serializable=True):
    """Identity and description of an agent instance.

    Parameters
    ----------
    uri : str
        Stable URI identifying the agent instance.
    name : str
        Human-readable agent name.
    description : str
        Short description of the agent's purpose.
    meta : Meta
        Additional metadata attached to the identity.
    """

    @classmethod
    def of(
        cls,
        uri: str | None = None,
        *,
        name: str,
        description: str = "",
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create an agent identity.

        Parameters
        ----------
        uri : str | None, default=None
            Explicit agent URI. When omitted, a unique ``agent://`` URI is generated.
        name : str
            Human-readable agent name.
        description : str, default=""
            Short description of the agent's purpose.
        meta : Meta | MetaValues | None, default=None
            Additional metadata attached to the identity.

        Returns
        -------
        Self
            New immutable identity instance.
        """
        return cls(
            uri=uri if uri else f"agent://{uuid4()}",
            name=name,
            description=description,
            meta=Meta.of(meta),
        )

    uri: str
    name: str
    description: str
    meta: Meta = Meta.empty


@final
class AgentMessage(State, serializable=True):
    """Single message delivered to an agent.

    Parameters
    ----------
    thread : UUID
        Conversation thread identifier associated with the message.
    created : datetime
        UTC timestamp recording when the message was created.
    content : MultimodalContent
        Normalized multimodal message payload.
    meta : Meta
        Additional message metadata.
    """

    @classmethod
    def of(
        cls,
        content: Multimodal,
        *,
        thread: UUID | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create an agent message.

        Parameters
        ----------
        content : Multimodal
            Message payload converted into ``MultimodalContent``.
        thread : UUID | None, default=None
            Conversation thread identifier. When omitted, a new thread is created.
        meta : Meta | MetaValues | None, default=None
            Additional message metadata.

        Returns
        -------
        Self
            New immutable message instance.
        """
        return cls(
            thread=thread if thread is not None else uuid4(),
            created=datetime.now(UTC),
            content=MultimodalContent.of(content),
            meta=Meta.of(meta),
        )

    thread: UUID = Default(default_factory=uuid4)
    created: datetime = Default(default_factory=lambda: datetime.now(UTC))
    content: MultimodalContent
    meta: Meta = Meta.empty


@final
class AgentThread(State):
    """Scoped runtime context of the executing agent within a conversation thread.

    The ``identifier`` names the conversation thread - a connected series of
    messages shared across nested agent calls - while ``agent_uri`` marks
    which agent is currently executing within it. ``Agent.respond`` binds a
    thread stamped with its own agent URI in scope for the duration of
    message handling; nested agent calls rebind it with the same identifier
    and their own URI. Memory operations are scoped by both.

    Parameters
    ----------
    identifier : UUID
        Thread identifier propagated through nested calls.
    agent_uri : str
        URI of the agent currently executing within the thread.
    created : datetime
        UTC timestamp recording when the thread context was created.
    meta : Meta
        Metadata shared within the active agent execution scope.
    """

    @classmethod
    def of(
        cls,
        identifier: UUID,
        *,
        agent_uri: str,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create an agent thread execution context.

        Parameters
        ----------
        identifier : UUID
            Conversation thread identifier.
        agent_uri : str
            URI of the agent executing within the thread. Agents stamp their
            own URI when binding the thread in scope.
        meta : Meta | MetaValues | None, default=None
            Metadata propagated through the active context scope.

        Returns
        -------
        Self
            New immutable agent context instance.
        """
        return cls(
            identifier=identifier,
            agent_uri=agent_uri,
            meta=Meta.of(meta),
        )

    identifier: UUID
    agent_uri: str
    created: datetime = Default(default_factory=lambda: datetime.now(UTC))
    meta: Meta = Meta.empty


@runtime_checkable
class AgentExecuting(Protocol):
    """Runtime contract implemented by agent executors.

    Returns
    -------
    AsyncIterable[MultimodalContentPart | ProcessingEvent]
        Stream produced by the executor for each processed message.
    """

    def __call__(
        self,
        message: AgentMessage,
    ) -> AsyncIterable[MultimodalContentPart | ProcessingEvent]:
        """Execute an agent for a single message.

        Parameters
        ----------
        message : AgentMessage
            Fully prepared agent message with thread and metadata.

        Returns
        -------
        AsyncIterable[MultimodalContentPart | ProcessingEvent]
            Stream of visible output chunks and processing events.
        """
        ...


@runtime_checkable
class AgentMemoryPreparing(Protocol):
    """Runtime contract implemented by ``AgentMemory`` prepare callables.

    Called lazily before each recall within a turn. Implementations must be
    idempotent per agent and thread - preparation may run once per turn and
    only the memory implementation knows whether an agent-thread pair is
    already prepared.

    Returns
    -------
    Sequence[Tool] | None
        Tools contributed by the memory for the prepared turn, if any.
    """

    async def __call__(
        self,
        thread: AgentThread,
        instructions: ModelInstructions,
        **extra: Any,
    ) -> Sequence[Tool] | None:
        """Prepare memory for the agent based on its instructions.

        Parameters
        ----------
        thread : AgentThread
            Executing agent thread scope the preparation is applied to -
            implementations key state by the executing agent (``agent_uri``)
            together with the thread ``identifier``.
        instructions : ModelInstructions
            Instructions of the agent utilizing the memory, resolved for the
            current turn. Resolution happens independently of the completion
            using the same instructions.
        **extra : Any
            Additional implementation-specific arguments.

        Returns
        -------
        Sequence[Tool] | None
            Tools contributed by the memory for the prepared turn, merged into
            the agent toolbox for its duration. ``None`` or empty when the
            memory contributes no tools.
        """
        ...


@runtime_checkable
class AgentMemoryRecalling(Protocol):
    """Runtime contract implemented by ``AgentMemory`` recall callables.

    Returns
    -------
    ModelContext
        Complete context to use for the upcoming completion, typically stored
        history followed by the provided turn context.
    """

    async def __call__(
        self,
        thread: AgentThread,
        context: ModelContext,
        **extra: Any,
    ) -> ModelContext:
        """Resolve the complete model context to use for a new turn.

        Parameters
        ----------
        thread : AgentThread
            Executing agent thread scope the recalled context belongs to -
            implementations key stored context by the executing agent
            (``agent_uri``) together with the thread ``identifier``.
        context : ModelContext
            Context accumulated so far for the current turn, not yet part of
            any stored context. Implementations must not assume it is a
            single input - preceding steps may have accumulated more.
        **extra : Any
            Additional implementation-specific arguments.

        Returns
        -------
        ModelContext
            Complete context to use for the upcoming completion, typically
            stored history followed by the provided turn context.
        """
        ...


@runtime_checkable
class AgentMemoryRemembering(Protocol):
    """Runtime contract implemented by ``AgentMemory`` remember callables.

    Returns
    -------
    None
        Completes once the context has been persisted.
    """

    async def __call__(
        self,
        thread: AgentThread,
        context: ModelContext,
        **extra: Any,
    ) -> None:
        """Persist context produced after a turn completes.

        Parameters
        ----------
        thread : AgentThread
            Executing agent thread scope the persisted context belongs to -
            implementations key stored context by the executing agent
            (``agent_uri``) together with the thread ``identifier``.
        context : ModelContext
            Full context accumulated for the turn.
        **extra : Any
            Additional implementation-specific arguments.

        Returns
        -------
        None
            Completes once the context has been persisted.
        """
        ...
