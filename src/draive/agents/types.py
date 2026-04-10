from collections.abc import AsyncIterable
from datetime import UTC, datetime
from typing import Protocol, Self, final, runtime_checkable
from uuid import UUID, uuid4

from haiway import Default, Meta, MetaValues, State

from draive.multimodal import Multimodal, MultimodalContent, MultimodalContentPart
from draive.utils import ProcessingEvent

__all__ = (
    "AgentException",
    "AgentExecuting",
    "AgentIdentity",
    "AgentMessage",
    "AgentUnavailable",
)


class AgentException(Exception):
    """Base exception raised by agent helpers.

    Raises
    ------
    AgentException
        Raised when agent operations fail with a framework-level error.
    """


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
class AgentContext(State):
    """Scoped runtime context shared across nested agent calls.

    Parameters
    ----------
    thread : UUID
        Conversation thread identifier propagated through nested calls.
    meta : Meta
        Metadata shared within the active agent execution scope.
    """

    @classmethod
    def of(
        cls,
        thread: UUID | None = None,
        *,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create an agent execution context.

        Parameters
        ----------
        thread : UUID | None, default=None
            Conversation thread identifier. When omitted, a new thread is created.
        meta : Meta | MetaValues | None, default=None
            Metadata propagated through the active context scope.

        Returns
        -------
        Self
            New immutable agent context instance.
        """
        return cls(
            thread=thread if thread is not None else uuid4(),
            meta=Meta.of(meta),
        )

    thread: UUID = Default(default_factory=uuid4)
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
