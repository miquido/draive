from collections.abc import Iterable, Sequence
from typing import Any, final, overload

from haiway import State, ctx, statemethod

from draive.conversation.completion.default import conversation_completion
from draive.conversation.completion.types import ConversationCompleting
from draive.conversation.state import ConversationMemory
from draive.conversation.types import ConversationOutputStream, ConversationTurn
from draive.models import (
    ModelInstructions,
)
from draive.multimodal import (
    Multimodal,
    Template,
    TemplatesRepository,
)
from draive.tools import Tool, Toolbox

__all__ = ("Conversation",)


@final
class Conversation(State):
    @overload
    @classmethod
    def completion(  # pyright: ignore[reportInconsistentOverload]
        # it seems to be pyright limitation and false positive
        cls,
        *,
        instructions: Template | ModelInstructions = "",
        tools: Toolbox | Iterable[Tool] = Toolbox.empty,
        memory: Sequence[ConversationTurn] | ConversationMemory = ConversationMemory.disabled,
        message: Template | Multimodal,
        **extra: Any,
    ) -> ConversationOutputStream: ...

    @overload
    def completion(
        self,
        *,
        instructions: Template | ModelInstructions = "",
        tools: Toolbox | Iterable[Tool] = Toolbox.empty,
        memory: Sequence[ConversationTurn] | ConversationMemory = ConversationMemory.disabled,
        message: Template | Multimodal,
        **extra: Any,
    ) -> ConversationOutputStream: ...

    @statemethod
    async def completion(
        self,
        *,
        instructions: Template | ModelInstructions = "",
        tools: Toolbox | Iterable[Tool] = Toolbox.empty,
        memory: Sequence[ConversationTurn] | ConversationMemory = ConversationMemory.disabled,
        message: Template | Multimodal,
        **extra: Any,
    ) -> ConversationOutputStream:
        async with ctx.scope("conversation"):
            if isinstance(instructions, Template):
                ctx.record_info(
                    attributes={"instructions.template": instructions.identifier},
                )
                instructions = await TemplatesRepository.resolve_str(instructions)

            if isinstance(message, Template):
                ctx.record_info(
                    attributes={"message.template": message.identifier},
                )
                message = await TemplatesRepository.resolve_str(message)

            if not isinstance(memory, ConversationMemory):
                memory = ConversationMemory.constant(memory)

            completion_stream: ConversationOutputStream = self._completing(
                instructions=instructions,
                toolbox=Toolbox.of(tools),
                memory=memory,
                message=message,
                **extra,
            )
            try:
                async for chunk in completion_stream:
                    yield chunk

            finally:
                # release the completion when the consumer stops before its end
                await completion_stream.aclose()

    _completing: ConversationCompleting

    def __init__(
        self,
        completing: ConversationCompleting = conversation_completion,
    ) -> None:
        super().__init__(_completing=completing)
