from collections.abc import Generator, Iterable, Mapping
from typing import Any, final

from haiway import BasicValue, State, ctx

from draive.conversation.realtime.default import realtime_conversation_preparing
from draive.conversation.realtime.types import (
    RealtimeConversationPreparing,
    RealtimeConversationSessionScope,
)
from draive.conversation.types import ConversationMessage
from draive.models import (
    ModelInput,
    ModelInstructions,
    ModelMemory,
    ModelMemoryRecall,
    ModelOutput,
    ModelSessionOutputSelection,
    Tool,
    Toolbox,
)
from draive.multimodal import Template, TemplatesRepository

__all__ = ("RealtimeConversation",)


@final
class RealtimeConversation(State):
    """Helper for preparing realtime conversation sessions.

    Normalizes tools and memory into model context and delegates to the configured
    realtime preparation implementation.
    """

    @classmethod
    async def prepare(
        cls,
        *,
        instructions: Template | ModelInstructions = "",
        tools: Toolbox | Iterable[Tool] = (),
        memory: ModelMemory | Iterable[ConversationMessage] = (),
        output: ModelSessionOutputSelection = "auto",
        **extra: Any,
    ) -> RealtimeConversationSessionScope:
        """Prepare a realtime conversation session scope.

        Parameters
        ----------
        instructions : Template | ModelInstructions, optional
            Instructions to steer the session.
        tools : Toolbox | Iterable[Tool], optional
            Tools to expose within the session.
        memory : ModelMemory | Iterable[ConversationMessage], optional
            Initial memory; an iterable is converted to model context.
        output : ModelSessionOutputSelection, optional
            Desired session output selection policy.
        **extra : Any
            Provider-specific kwargs forwarded to the preparation implementation.
        """

        async with ctx.scope("conversation_realtime"):
            conversation: RealtimeConversation = ctx.state(cls)

            conversation_memory: ModelMemory
            memory_variables: Mapping[str, BasicValue] | None
            if isinstance(memory, ModelMemory):
                conversation_memory = memory
                memory_recall: ModelMemoryRecall = await memory.recall()
                memory_variables = memory_recall.variables

            else:

                def model_context_elements() -> Generator[ModelInput | ModelOutput]:
                    for message in memory:
                        if message.role == "user":
                            yield ModelInput.of(message.content)

                        else:
                            yield ModelOutput.of(message.content)

                conversation_memory = ModelMemory.constant(*model_context_elements())
                memory_variables = None

            if isinstance(instructions, Template):
                ctx.record_info(
                    attributes={"instructions.template": instructions.identifier},
                )

            return await conversation.preparing(
                instructions=await TemplatesRepository.resolve_str(
                    instructions,
                    arguments={
                        key: value if isinstance(value, str) else str(value)
                        for key, value in memory_variables.items()
                    }
                    if memory_variables
                    else None,
                ),
                toolbox=Toolbox.of(tools),
                memory=conversation_memory,
                output=output,
                **extra,
            )

    preparing: RealtimeConversationPreparing = realtime_conversation_preparing
