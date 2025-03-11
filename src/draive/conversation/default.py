from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from typing import Any, Literal, overload

from haiway import ArgumentsTrace, ResultTrace, ctx

from draive.conversation.stream import conversation_stream
from draive.conversation.types import ConversationMemory, ConversationMessage
from draive.instructions import Instruction
from draive.lmm import (
    LMMCompletion,
    LMMContextElement,
    LMMInput,
    LMMStreamChunk,
    LMMToolRequests,
    LMMToolResponses,
    lmm_invoke,
)
from draive.multimodal import Multimodal, MultimodalContent
from draive.prompts import Prompt
from draive.tools import Toolbox
from draive.utils import Memory, ProcessingEvent

__all__ = [
    "default_conversation_completion",
]


@overload
async def default_conversation_completion(
    *,
    instruction: Instruction | str | None,
    input: ConversationMessage | Prompt | Multimodal,
    memory: ConversationMemory,
    toolbox: Toolbox,
    stream: Literal[True],
    **extra: Any,
) -> AsyncIterator[LMMStreamChunk | ProcessingEvent]: ...


@overload
async def default_conversation_completion(
    *,
    instruction: Instruction | str | None,
    input: ConversationMessage | Prompt | Multimodal,
    memory: ConversationMemory,
    toolbox: Toolbox,
    stream: Literal[False] = False,
    **extra: Any,
) -> ConversationMessage: ...


@overload
async def default_conversation_completion(
    *,
    instruction: Instruction | str | None,
    input: ConversationMessage | Prompt | Multimodal,
    memory: ConversationMemory,
    toolbox: Toolbox,
    stream: bool,
    **extra: Any,
) -> AsyncIterator[LMMStreamChunk | ProcessingEvent] | ConversationMessage: ...


async def default_conversation_completion(
    *,
    instruction: Instruction | str | None,
    input: ConversationMessage | Prompt | Multimodal,  # noqa: A002
    memory: ConversationMemory,
    toolbox: Toolbox,
    stream: bool = False,
    **extra: Any,
) -> AsyncIterator[LMMStreamChunk | ProcessingEvent] | ConversationMessage:
    if stream:
        return ctx.stream(
            conversation_stream,
            instruction=instruction,
            input=input,
            memory=memory,
            toolbox=toolbox,
            **extra,
        )

    with ctx.scope("conversation_completion"):
        recalled_messages: Sequence[ConversationMessage] = await memory.recall()
        context: list[LMMContextElement]
        match input:
            case ConversationMessage() as message:
                await memory.remember(message)
                context = [
                    *(message.as_lmm_context_element() for message in recalled_messages),
                    message.as_lmm_context_element(),
                ]

            case Prompt() as prompt:
                prompt_messages: list[ConversationMessage] = []
                for element in prompt.content:
                    match element:
                        case LMMCompletion() as completion_element:
                            prompt_messages.append(
                                ConversationMessage(
                                    role="model",
                                    created=datetime.now(UTC),
                                    content=completion_element.content,
                                )
                            )

                        case LMMInput() as input_element:
                            prompt_messages.append(
                                ConversationMessage(
                                    role="user",
                                    created=datetime.now(UTC),
                                    content=input_element.content,
                                )
                            )

                        case _:
                            pass  # TODO add other content to memory when able

                await memory.remember(*prompt_messages)

                context = [
                    *(message.as_lmm_context_element() for message in recalled_messages),
                    *prompt.content,
                ]

            case content:
                message = ConversationMessage(
                    role="user",
                    created=datetime.now(UTC),
                    content=MultimodalContent.of(content),
                )
                await memory.remember(message)
                context = [
                    *(message.as_lmm_context_element() for message in recalled_messages),
                    message.as_lmm_context_element(),
                ]

        return await _conversation_completion(
            instruction=instruction,
            memory=memory,
            context=context,
            toolbox=toolbox,
            **extra,
        )


async def _conversation_completion(
    instruction: Instruction | str | None,
    memory: Memory[Sequence[ConversationMessage], ConversationMessage],
    context: list[LMMContextElement],
    toolbox: Toolbox,
    **extra: Any,
) -> ConversationMessage:
    ctx.record(
        ArgumentsTrace.of(
            instruction=instruction,
            context=context,
        )
    )
    recursion_level: int = 0
    while recursion_level <= toolbox.repeated_calls_limit:
        match await lmm_invoke(
            instruction=instruction,
            context=context,
            tools=toolbox.available_tools(),
            tool_selection=toolbox.tool_selection(repetition_level=recursion_level),
            output="text",
            **extra,
        ):
            case LMMCompletion() as completion:
                ctx.log_debug("Received conversation result")
                response_message: ConversationMessage = ConversationMessage(
                    role="model",
                    created=datetime.now(UTC),
                    content=completion.content,
                )

                await memory.remember(response_message)

                ctx.record(ResultTrace.of(response_message))

                return response_message

            case LMMToolRequests() as tool_requests:
                ctx.log_debug("Received conversation tool calls")

                tool_responses: LMMToolResponses = await toolbox.respond_all(tool_requests)

                if direct_content := [
                    response.content for response in tool_responses.responses if response.direct
                ]:
                    response_message: ConversationMessage = ConversationMessage(
                        role="model",
                        created=datetime.now(UTC),
                        content=MultimodalContent.of(*direct_content),
                    )
                    await memory.remember(response_message)

                    ctx.record(ResultTrace.of(response_message))

                    return response_message

                else:
                    context.extend(
                        [
                            tool_requests,
                            tool_responses,
                        ]
                    )

        recursion_level += 1  # continue with next recursion level

    raise RuntimeError("LMM exceeded limit of recursive calls")
