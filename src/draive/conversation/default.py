from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from datetime import UTC, datetime
from typing import Any, Literal, overload

from haiway import ArgumentsTrace, AsyncQueue, ResultTrace, ctx

from draive.conversation.types import ConversationMemory, ConversationMessage
from draive.instructions import Instruction
from draive.lmm import (
    LMMCompletion,
    LMMContextElement,
    LMMInput,
    LMMStreamChunk,
    LMMStreamInput,
    LMMStreamProperties,
    LMMToolRequest,
    LMMToolRequests,
    LMMToolResponse,
    LMMToolResponses,
    lmm_invoke,
    lmm_stream,
)
from draive.multimodal import Multimodal, MultimodalContent
from draive.prompts import Prompt
from draive.tools import Toolbox
from draive.utils import Memory

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
) -> AsyncIterator[LMMStreamChunk]: ...


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
) -> AsyncIterator[LMMStreamChunk] | ConversationMessage: ...


async def default_conversation_completion(
    *,
    instruction: Instruction | str | None,
    input: ConversationMessage | Prompt | Multimodal,  # noqa: A002
    memory: ConversationMemory,
    toolbox: Toolbox,
    stream: bool = False,
    **extra: Any,
) -> AsyncIterator[LMMStreamChunk] | ConversationMessage:
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

        if stream:
            return ctx.stream(
                _conversation_stream,
                instruction=instruction,
                memory=memory,
                context=context,
                toolbox=toolbox,
                **extra,
            )

        else:
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

                responses: Sequence[LMMToolResponse] = await toolbox.respond_all(tool_requests)

                if direct_content := [
                    response.content for response in responses if response.direct
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
                            LMMToolResponses(responses=responses),
                        ]
                    )

        recursion_level += 1  # continue with next recursion level

    raise RuntimeError("LMM exceeded limit of recursive calls")


async def _conversation_stream(
    instruction: Instruction | str | None,
    memory: Memory[Sequence[ConversationMessage], ConversationMessage],
    context: list[LMMContextElement],
    toolbox: Toolbox,
    **extra: Any,
) -> AsyncGenerator[LMMStreamChunk]:
    ctx.record(
        ArgumentsTrace.of(
            instruction=instruction,
            context=context,
        )
    )
    if not isinstance(context[-1], LMMInput):
        raise ValueError(f"Streaming input has to end with LMMInput, received {type(context[-1])}")

    input_stream = AsyncQueue[LMMStreamInput]()
    input_stream.enqueue(
        LMMStreamChunk.of(
            context[-1].content,
            eod=True,  # we provide single input chunk through this interface
        )
    )
    del context[-1]  # we are using last element as stream input, we have to remove it from context
    accumulated_content: MultimodalContent = MultimodalContent.of()
    async for element in await lmm_stream(
        properties=LMMStreamProperties(
            instruction=instruction,
            tools=toolbox.available_tools(),
        ),
        input=input_stream,
        context=context,
        **extra,
    ):
        match element:
            case LMMStreamChunk() as chunk:
                accumulated_content = accumulated_content.appending(
                    *chunk.content.parts,
                    merge_text=True,
                )

                yield chunk

                # on turn end finalize the stream - we are streaming only a single response here
                if chunk.eod:
                    input_stream.finish()
                    response_message: ConversationMessage = ConversationMessage(
                        role="model",
                        created=datetime.now(UTC),
                        content=accumulated_content,
                    )
                    await memory.remember(response_message)

                    ctx.record(ResultTrace.of(response_message))

                    return  # end of streaming for conversation completion

            case LMMToolRequest() as tool_request:
                ctx.spawn(
                    handle_tool_call,
                    tool_request,
                    toolbox=toolbox,
                    output=input_stream,
                )


async def handle_tool_call(
    request: LMMToolRequest,
    /,
    *,
    toolbox: Toolbox,
    output: AsyncQueue[LMMStreamInput],
) -> None:
    output.enqueue(await toolbox.respond(request))
