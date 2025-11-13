from collections.abc import AsyncGenerator, AsyncIterable
from types import TracebackType
from typing import Any, Protocol, Self, runtime_checkable

from haiway import State, ctx

from draive.conversation.types import (
    ConversationEvent,
    ConversationInputChunk,
    ConversationOutputChunk,
)
from draive.models import ModelInstructions, ModelMemory, Toolbox

__all__ = (
    "RealtimeConversationPreparing",
    "RealtimeConversationSession",
    "RealtimeConversationSessionClosing",
    "RealtimeConversationSessionOpening",
    "RealtimeConversationSessionReading",
    "RealtimeConversationSessionScope",
    "RealtimeConversationSessionWriting",
)


@runtime_checkable
class RealtimeConversationSessionReading(Protocol):
    """Callable that reads the next output chunk from the realtime session."""

    async def __call__(self) -> ConversationOutputChunk | ConversationEvent: ...


@runtime_checkable
class RealtimeConversationSessionWriting(Protocol):
    """Callable that writes an input chunk to the realtime session."""

    async def __call__(
        self,
        input: ConversationInputChunk | ConversationEvent,  # noqa: A002
    ) -> None: ...


class RealtimeConversationSession(State):
    """Static helpers for interacting with the active realtime conversation session."""

    @classmethod
    async def read(cls) -> ConversationOutputChunk | ConversationEvent:
        """Read a single ``ConversationOutputChunk`` from the session."""
        return await ctx.state(cls).reading()

    @classmethod
    async def reader(cls) -> AsyncGenerator[ConversationOutputChunk | ConversationEvent]:
        """Async iterator continuously yielding session output chunks until error."""
        session: Self = ctx.state(cls)
        while True:  # breaks on exception
            yield await session.reading()

    @classmethod
    async def write(
        cls,
        input: ConversationInputChunk | ConversationEvent,  # noqa: A002
    ) -> None:
        """Write a single input chunk into the session."""
        return await ctx.state(cls).writing(input=input)

    @classmethod
    async def writer(
        cls,
        input: AsyncIterable[ConversationInputChunk | ConversationEvent],  # noqa: A002
    ) -> None:
        """Consume an async iterator and write its chunks to the session."""
        session: Self = ctx.state(cls)
        async for chunk in input:
            await session.writing(input=chunk)

    reading: RealtimeConversationSessionReading
    writing: RealtimeConversationSessionWriting


@runtime_checkable
class RealtimeConversationSessionOpening(Protocol):
    """Callable that opens a realtime conversation session."""

    async def __call__(self) -> RealtimeConversationSession: ...


@runtime_checkable
class RealtimeConversationSessionClosing(Protocol):
    """Callable that closes a realtime conversation session."""

    async def __call__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...


class RealtimeConversationSessionScope(State):
    """Async context manager that opens and closes a realtime session."""

    opening: RealtimeConversationSessionOpening
    closing: RealtimeConversationSessionClosing

    async def __aenter__(self) -> RealtimeConversationSession:
        """Open and return the underlying ``RealtimeConversationSession``."""
        return await self.opening()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the session, forwarding exceptions to the closing hook."""
        await self.closing(
            exc_type=exc_type,
            exc_val=exc_val,
            exc_tb=exc_tb,
        )


@runtime_checkable
class RealtimeConversationPreparing(Protocol):
    """Callable that prepares and returns a realtime conversation session scope."""

    async def __call__(
        self,
        *,
        instructions: ModelInstructions,
        toolbox: Toolbox,
        memory: ModelMemory,
        **extra: Any,
    ) -> RealtimeConversationSessionScope: ...
