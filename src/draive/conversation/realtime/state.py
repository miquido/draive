from collections.abc import Iterable
from typing import Any, final, overload

from haiway import State, ctx, statemethod

from draive.conversation.realtime.default import realtime_conversation_preparing
from draive.conversation.realtime.types import (
    RealtimeConversationPreparing,
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
    async def prepare(
        cls,
        *,
        instructions: Template | ModelInstructions = "",
        tools: Toolbox | Iterable[Tool] = Toolbox.empty,
        memory: ConversationMemory = ConversationMemory.disabled,
        output: ModelSessionOutputSelection = "auto",
        **extra: Any,
    ) -> RealtimeConversationSessionScope: ...

    @overload
    async def prepare(
        self,
        *,
        instructions: Template | ModelInstructions = "",
        tools: Toolbox | Iterable[Tool] = Toolbox.empty,
        memory: ConversationMemory = ConversationMemory.disabled,
        output: ModelSessionOutputSelection = "auto",
        **extra: Any,
    ) -> RealtimeConversationSessionScope: ...

    @statemethod
    async def prepare(
        self,
        *,
        instructions: Template | ModelInstructions = "",
        tools: Toolbox | Iterable[Tool] = Toolbox.empty,
        memory: ConversationMemory = ConversationMemory.disabled,
        output: ModelSessionOutputSelection = "auto",
        **extra: Any,
    ) -> RealtimeConversationSessionScope:
        async with ctx.scope("conversation.realtime"):
            if isinstance(instructions, Template):
                ctx.record_info(
                    attributes={"instructions.template": instructions.identifier},
                )
                instructions = await TemplatesRepository.resolve_str(instructions)

            return await self._preparing(
                instructions=instructions,
                toolbox=Toolbox.of(tools),
                memory=memory,
                output=output,
                **extra,
            )

    _preparing: RealtimeConversationPreparing = realtime_conversation_preparing

    def __init__(
        self,
        preparing: RealtimeConversationPreparing = realtime_conversation_preparing,
    ) -> None:
        super().__init__(_preparing=preparing)
