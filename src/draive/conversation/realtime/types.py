from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any, Protocol, Self, runtime_checkable

from haiway import State, ctx

from draive.conversation.types import ConversationMemory, ConversationStreamElement
from draive.instructions import Instruction
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
    async def __call__(self) -> ConversationStreamElement: ...


@runtime_checkable
class RealtimeConversationSessionWriting(Protocol):
    async def __call__(
        self,
        input: ConversationStreamElement,  # noqa: A002
    ) -> None: ...


class RealtimeConversationSession(State):
    @classmethod
    async def read(cls) -> ConversationStreamElement:
        return await ctx.state(cls).reading()

    @classmethod
    async def reader(cls) -> AsyncIterator[ConversationStreamElement]:
        session: Self = ctx.state(cls)
        while True:  # breaks on exception
            yield await session.reading()

    @classmethod
    async def write(
        cls,
        input: ConversationStreamElement,  # noqa: A002
    ) -> None:
        return await ctx.state(cls).writing(input=input)

    @classmethod
    async def writer(
        cls,
        input: AsyncIterator[ConversationStreamElement],  # noqa: A002
    ) -> None:
        session: Self = ctx.state(cls)
        while True:  # breaks on exception
            await session.writing(input=await anext(input))

    reading: RealtimeConversationSessionReading
    writing: RealtimeConversationSessionWriting


@runtime_checkable
class RealtimeConversationSessionOpening(Protocol):
    async def __call__(self) -> RealtimeConversationSession: ...


@runtime_checkable
class RealtimeConversationSessionClosing(Protocol):
    async def __call__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...


class RealtimeConversationSessionScope(State):
    opening: RealtimeConversationSessionOpening
    closing: RealtimeConversationSessionClosing

    async def __aenter__(self) -> RealtimeConversationSession:
        return await self.opening()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.closing(
            exc_type=exc_type,
            exc_val=exc_val,
            exc_tb=exc_tb,
        )


@runtime_checkable
class RealtimeConversationPreparing(Protocol):
    async def __call__(
        self,
        *,
        instruction: Instruction | None,
        memory: ConversationMemory | None,
        toolbox: Toolbox,
        **extra: Any,
    ) -> RealtimeConversationSessionScope: ...
