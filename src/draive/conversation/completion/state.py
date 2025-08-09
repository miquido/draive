from collections.abc import AsyncIterator, Generator, Iterable
from typing import Any, Literal, final, overload

from haiway import State, ctx

from draive.conversation.completion.default import conversation_completion
from draive.conversation.completion.types import ConversationCompleting
from draive.conversation.types import ConversationMessage, ConversationOutputChunk
from draive.models import (
    ModelInput,
    ModelMemory,
    ModelMemoryRecall,
    ModelOutput,
    ResolveableInstructions,
    Tool,
    Toolbox,
)
from draive.multimodal import Multimodal
from draive.utils import Memory

__all__ = ("Conversation",)


@final
class Conversation(State):
    """High-level helper for conversational completion over generative models.

    Normalizes inputs/memory and delegates to the configured ``ConversationCompleting``
    implementation. Supports both full-response and streaming modes.
    """

    @overload
    @classmethod
    async def completion(
        cls,
        *,
        instructions: ResolveableInstructions = "",
        tools: Toolbox | Iterable[Tool] | None = None,
        memory: ModelMemory | Iterable[ConversationMessage] | None = None,
        input: ConversationMessage | Multimodal,
        stream: Literal[False] = False,
        **extra: Any,
    ) -> ConversationMessage: ...

    @overload
    @classmethod
    async def completion(
        cls,
        *,
        instructions: ResolveableInstructions = "",
        tools: Toolbox | Iterable[Tool] = (),
        memory: ModelMemory | Iterable[ConversationMessage] = (),
        input: ConversationMessage | Multimodal,
        stream: Literal[True],
        **extra: Any,
    ) -> AsyncIterator[ConversationOutputChunk]: ...

    @classmethod
    async def completion(
        cls,
        *,
        instructions: ResolveableInstructions = "",
        tools: Toolbox | Iterable[Tool] | None = None,
        memory: ModelMemory | Iterable[ConversationMessage] | None = None,
        input: ConversationMessage | Multimodal,  # noqa: A002
        stream: bool = False,
        **extra: Any,
    ) -> AsyncIterator[ConversationOutputChunk] | ConversationMessage:
        """Run a single conversational completion.

        Parameters
        ----------
        instructions : ResolveableInstructions, optional
            Instruction reference or content to steer the model.
        tools : Toolbox | Iterable[Tool] | None, optional
            Tools to expose for this turn.
        memory : ModelMemory | Iterable[ConversationMessage] | None, optional
            Conversation memory; if an iterable is provided it is converted to model context.
        input : ConversationMessage | Multimodal
            Input message or raw content (converted to a user message).
        stream : bool, optional
            When ``True``, return an async iterator of ``ConversationOutputChunk``.
        **extra : Any
            Provider-specific kwargs forwarded to the underlying implementation.

        Returns
        -------
        ConversationMessage or AsyncIterator[ConversationOutputChunk]
            Final response message or a stream of output chunks.
        """
        conversation_message: ConversationMessage
        if isinstance(input, ConversationMessage):
            conversation_message = input

        else:
            conversation_message = ConversationMessage.user(input)

        conversation_memory: ModelMemory
        if memory is None:
            conversation_memory = ModelMemory.constant(ModelMemoryRecall.empty)

        elif isinstance(memory, Memory):
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

        if stream:
            return await ctx.state(cls).completing(
                instructions=instructions,
                toolbox=Toolbox.of(tools),
                memory=conversation_memory,
                input=conversation_message,
                stream=True,
                **extra,
            )

        else:
            return await ctx.state(cls).completing(
                instructions=instructions,
                toolbox=Toolbox.of(tools),
                memory=conversation_memory,
                input=conversation_message,
                **extra,
            )

    completing: ConversationCompleting = conversation_completion
