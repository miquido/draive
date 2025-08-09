from collections.abc import Generator, Iterable
from typing import Any, final

from haiway import State, ctx

from draive.conversation.realtime.default import realtime_conversation_preparing
from draive.conversation.realtime.types import (
    RealtimeConversationPreparing,
    RealtimeConversationSessionScope,
)
from draive.conversation.types import ConversationMessage
from draive.models import (
    ModelInput,
    ModelMemory,
    ModelMemoryRecall,
    ModelOutput,
    ModelSessionOutputSelection,
    ResolveableInstructions,
    Tool,
    Toolbox,
)
from draive.utils import Memory

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
        instructions: ResolveableInstructions = "",
        tools: Toolbox | Iterable[Tool] = (),
        memory: ModelMemory | Iterable[ConversationMessage] = (),
        output: ModelSessionOutputSelection = "auto",
        **extra: Any,
    ) -> RealtimeConversationSessionScope:
        """Prepare a realtime conversation session scope.

        Parameters
        ----------
        instructions : ResolveableInstructions, optional
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
        conversation: RealtimeConversation = ctx.state(cls)

        conversation_memory: ModelMemory
        if isinstance(memory, Memory):
            conversation_memory = memory

        else:

            def model_context_elements() -> Generator[ModelInput | ModelOutput]:
                for message in memory:
                    if message.role == "user":
                        yield ModelInput.of(message.content)

                    else:
                        yield ModelOutput.of(message.content)

            conversation_memory = ModelMemory.constant(
                ModelMemoryRecall.of(*model_context_elements())
            )

        return await conversation.preparing(
            instructions=instructions,
            toolbox=Toolbox.of(tools),
            memory=conversation_memory,
            output=output,
            **extra,
        )

    preparing: RealtimeConversationPreparing = realtime_conversation_preparing
