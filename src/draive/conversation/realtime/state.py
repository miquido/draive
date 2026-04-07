from collections.abc import Iterable
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Any, final, overload

from haiway import State, ctx, statemethod

from draive.conversation.realtime.default import realtime_conversation_preparing
from draive.conversation.realtime.types import (
    RealtimeConversationPreparing,
    RealtimeConversationSession,
    RealtimeConversationSessionScope,
)
from draive.conversation.state import ConversationMemory
from draive.models import (
    ModelInstructions,
    ModelSessionOutputSelection,
)
from draive.multimodal import Template, TemplatesRepository
from draive.tools import Tool, Toolbox

__all__ = ("RealtimeConversation",)


@final
class RealtimeConversation(State):
    """Helper for preparing realtime conversation sessions.

    Normalizes tools and memory into model context and delegates to the configured
    realtime preparation implementation.
    """

    @overload
    @classmethod
    def prepare(  # pyright: ignore[reportInconsistentOverload]
        # it seems to be pyright limitation and false positive
        cls,
        *,
        instructions: Template | ModelInstructions = "",
        tools: Toolbox | Iterable[Tool] = Toolbox.empty,
        memory: ConversationMemory = ConversationMemory.disabled,
        output: ModelSessionOutputSelection = "auto",
        **extra: Any,
    ) -> AbstractAsyncContextManager[RealtimeConversationSession]: ...

    @overload
    def prepare(
        self,
        *,
        instructions: Template | ModelInstructions = "",
        tools: Toolbox | Iterable[Tool] = Toolbox.empty,
        memory: ConversationMemory = ConversationMemory.disabled,
        output: ModelSessionOutputSelection = "auto",
        **extra: Any,
    ) -> AbstractAsyncContextManager[RealtimeConversationSession]: ...

    @statemethod
    def prepare(
        self,
        *,
        instructions: Template | ModelInstructions = "",
        tools: Toolbox | Iterable[Tool] = Toolbox.empty,
        memory: ConversationMemory = ConversationMemory.disabled,
        output: ModelSessionOutputSelection = "auto",
        **extra: Any,
    ) -> AbstractAsyncContextManager[RealtimeConversationSession]:
        scope: AbstractAsyncContextManager[str]
        session_scope: AbstractAsyncContextManager[RealtimeConversationSession]

        async def opening(
            instructions: Template | ModelInstructions = instructions,
        ) -> RealtimeConversationSession:
            nonlocal scope
            nonlocal session_scope
            scope = ctx.scope("conversation.realtime")
            await scope.__aenter__()
            if isinstance(instructions, Template):
                ctx.record_info(
                    attributes={"instructions.template": instructions.identifier},
                )
                instructions = await TemplatesRepository.resolve_str(instructions)

            session_scope = self._preparing(
                instructions=instructions,
                toolbox=Toolbox.of(tools),
                memory=memory,
                output=output,
                **extra,
            )
            return await session_scope.__aenter__()

        async def closing(
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
        ) -> None:
            try:
                await session_scope.__aexit__(  # noqa: F821
                    exc_type,
                    exc_val,
                    exc_tb,
                )

            finally:
                await scope.__aexit__(  # noqa: F821
                    exc_type,
                    exc_val,
                    exc_tb,
                )

        return RealtimeConversationSessionScope(
            opening=opening,
            closing=closing,
        )

    _preparing: RealtimeConversationPreparing = realtime_conversation_preparing

    def __init__(
        self,
        preparing: RealtimeConversationPreparing = realtime_conversation_preparing,
    ) -> None:
        super().__init__(_preparing=preparing)
