from collections.abc import AsyncIterable, Generator, Iterable, Mapping
from typing import Any, Literal, final, overload

from haiway import BasicValue, State, ctx

from draive.conversation.completion.default import conversation_completion
from draive.conversation.completion.types import ConversationCompleting
from draive.conversation.types import ConversationMessage, ConversationOutputChunk
from draive.models import (
    ModelInput,
    ModelInstructions,
    ModelMemory,
    ModelMemoryRecall,
    ModelOutput,
    Tool,
    Toolbox,
)
from draive.multimodal import Multimodal, Template, TemplatesRepository

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
        instructions: Template | ModelInstructions = "",
        tools: Toolbox | Iterable[Tool] | None = None,
        memory: ModelMemory | Iterable[ConversationMessage] | None = None,
        input: ConversationMessage | Template | Multimodal,
        stream: Literal[False] = False,
        **extra: Any,
    ) -> ConversationMessage: ...

    @overload
    @classmethod
    async def completion(
        cls,
        *,
        instructions: Template | ModelInstructions = "",
        tools: Toolbox | Iterable[Tool] = (),
        memory: ModelMemory | Iterable[ConversationMessage] = (),
        input: ConversationMessage | Template | Multimodal,
        stream: Literal[True],
        **extra: Any,
    ) -> AsyncIterable[ConversationOutputChunk]: ...

    @classmethod
    async def completion(
        cls,
        *,
        instructions: Template | ModelInstructions = "",
        tools: Toolbox | Iterable[Tool] | None = None,
        memory: ModelMemory | Iterable[ConversationMessage] | None = None,
        input: ConversationMessage | Template | Multimodal,  # noqa: A002
        stream: bool = False,
        **extra: Any,
    ) -> AsyncIterable[ConversationOutputChunk] | ConversationMessage:
        """Run a single conversational completion.

        Parameters
        ----------
        instructions : Template | ModelInstructions, optional
            Instruction reference or content to steer the model.
        tools : Toolbox | Iterable[Tool] | None, optional
            Tools to expose for this turn.
        memory : ModelMemory | Iterable[ConversationMessage] | None, optional
            Conversation memory; if an iterable is provided it is converted to model context.
        input : ConversationMessage | Template |Multimodal
            Input message, message template or raw content (converted to a user message).
        stream : bool, optional
            When ``True``, return an async iterator of ``ConversationOutputChunk``.
        **extra : Any
            Provider-specific kwargs forwarded to the underlying implementation.

        Returns
        -------
        ConversationMessage or AsyncIterable[ConversationOutputChunk]
            Final response message or a stream of output chunks.
        """
        async with ctx.scope("conversation_completion"):
            conversation_message: ConversationMessage
            if isinstance(input, ConversationMessage):
                conversation_message = input

            else:
                conversation_message = ConversationMessage.user(
                    await TemplatesRepository.resolve(input)
                )

            conversation_memory: ModelMemory
            memory_variables: Mapping[str, BasicValue] | None
            if memory is None:
                conversation_memory = ModelMemory.constant()
                memory_variables = None

            elif isinstance(memory, ModelMemory):
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

            model_instructions: ModelInstructions = await TemplatesRepository.resolve_str(
                instructions,
                arguments={
                    key: value if isinstance(value, str) else str(value)
                    for key, value in memory_variables.items()
                }
                if memory_variables
                else None,
            )

            if stream:
                return await ctx.state(cls).completing(
                    instructions=model_instructions,
                    toolbox=Toolbox.of(tools),
                    memory=conversation_memory,
                    input=conversation_message,
                    stream=True,
                    **extra,
                )

            else:
                return await ctx.state(cls).completing(
                    instructions=model_instructions,
                    toolbox=Toolbox.of(tools),
                    memory=conversation_memory,
                    input=conversation_message,
                    stream=False,
                    **extra,
                )

    completing: ConversationCompleting = conversation_completion
