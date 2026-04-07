from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Any, Protocol, overload, runtime_checkable

from haiway import State, statemethod

from draive.conversation.state import ConversationMemory
from draive.conversation.types import (
    ConversationEvent,
    ConversationInputChunk,
    ConversationInputStream,
    ConversationOutputChunk,
    ConversationOutputStream,
)
from draive.models import ModelInstructions, ModelReasoningChunk
from draive.multimodal import MultimodalContentPart
from draive.tools import Toolbox

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

    async def __call__(
        self,
    ) -> MultimodalContentPart | ModelReasoningChunk | ConversationEvent: ...


@runtime_checkable
class RealtimeConversationSessionWriting(Protocol):
    """Callable that writes an input chunk to the realtime session."""

    async def __call__(
        self,
        input: MultimodalContentPart | ConversationEvent,  # noqa: A002
    ) -> None: ...


class RealtimeConversationSession(State):
    """Static helpers for interacting with the active realtime conversation session."""

    @classmethod
    @overload
    async def read(cls) -> ConversationOutputChunk: ...

    @overload
    async def read(self) -> ConversationOutputChunk: ...

    @statemethod
    async def read(self) -> ConversationOutputChunk:
        """Read a single ``ConversationOutputChunk`` from the session."""
        return await self._reading()

    @classmethod
    @overload
    def reader(cls) -> ConversationOutputStream:  # pyright: ignore[reportInconsistentOverload]
        # it seems to be pyright limitation and false positive
        ...

    @overload
    def reader(self) -> ConversationOutputStream: ...

    @statemethod
    async def reader(self) -> ConversationOutputStream:
        """Async iterator continuously yielding session output chunks until error."""
        while True:  # breaks on exception
            yield await self._reading()

    @classmethod
    @overload
    async def write(
        cls,
        input: ConversationInputChunk,
    ) -> None: ...

    @overload
    async def write(
        self,
        input: ConversationInputChunk,
    ) -> None: ...

    @statemethod
    async def write(
        self,
        input: ConversationInputChunk,  # noqa: A002
    ) -> None:
        """Write a single input chunk into the session."""
        return await self._writing(input=input)

    @classmethod
    @overload
    async def writer(
        cls,
        input: ConversationInputStream,
    ) -> None: ...

    @overload
    async def writer(
        self,
        input: ConversationInputStream,
    ) -> None: ...

    @statemethod
    async def writer(
        self,
        input: ConversationInputStream,  # noqa: A002
    ) -> None:
        """Consume an async iterator and write its chunks to the session."""
        async for chunk in input:
            await self._writing(input=chunk)

    _reading: RealtimeConversationSessionReading
    _writing: RealtimeConversationSessionWriting

    def __init__(
        self,
        reading: RealtimeConversationSessionReading,
        writing: RealtimeConversationSessionWriting,
    ) -> None:
        super().__init__(
            _reading=reading,
            _writing=writing,
        )


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

    _opening: RealtimeConversationSessionOpening
    _closing: RealtimeConversationSessionClosing

    def __init__(
        self,
        opening: RealtimeConversationSessionOpening,
        closing: RealtimeConversationSessionClosing,
    ) -> None:
        super().__init__(
            _opening=opening,
            _closing=closing,
        )

    async def __aenter__(self) -> RealtimeConversationSession:
        """Open and return the underlying ``RealtimeConversationSession``."""
        return await self._opening()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the session, forwarding exceptions to the closing hook."""
        await self._closing(
            exc_type=exc_type,
            exc_val=exc_val,
            exc_tb=exc_tb,
        )


@runtime_checkable
class RealtimeConversationPreparing(Protocol):
    """Callable that prepares and returns a realtime conversation session scope."""

    def __call__(
        self,
        *,
        instructions: ModelInstructions,
        toolbox: Toolbox,
        memory: ConversationMemory,
        **extra: Any,
    ) -> AbstractAsyncContextManager[RealtimeConversationSession]: ...
