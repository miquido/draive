from collections.abc import AsyncGenerator, AsyncIterator, Iterable
from datetime import UTC, datetime
from typing import Any, Literal, overload

from haiway import AsyncQueue, Missing, ctx, not_missing

from draive.conversation.types import ConversationMessage
from draive.instructions import Instruction
from draive.lmm import (
    LMMCompletion,
    LMMContextElement,
    LMMStreamChunk,
    LMMStreamInput,
    LMMStreamProperties,
    LMMToolRequest,
    LMMToolRequests,
    LMMToolResponse,
    Toolbox,
    lmm_invoke,
    lmm_stream,
)
from draive.multimodal import MultimodalContent
from draive.utils import Memory

__all__: list[str] = [
    "default_conversation_completion",
]


@overload
async def default_conversation_completion(
    *,
    instruction: Instruction | str | None,
    message: ConversationMessage,
    memory: Memory[Iterable[ConversationMessage], ConversationMessage],
    toolbox: Toolbox,
    stream: Literal[True],
    **extra: Any,
) -> AsyncIterator[LMMStreamChunk]: ...


@overload
async def default_conversation_completion(
    *,
    instruction: Instruction | str | None,
    message: ConversationMessage,
    memory: Memory[Iterable[ConversationMessage], ConversationMessage],
    toolbox: Toolbox,
    stream: Literal[False] = False,
    **extra: Any,
) -> ConversationMessage: ...


@overload
async def default_conversation_completion(
    *,
    instruction: Instruction | str | None,
    message: ConversationMessage,
    memory: Memory[Iterable[ConversationMessage], ConversationMessage],
    toolbox: Toolbox,
    stream: bool,
    **extra: Any,
) -> AsyncIterator[LMMStreamChunk] | ConversationMessage: ...


async def default_conversation_completion(
    *,
    instruction: Instruction | str | None,
    message: ConversationMessage,
    memory: Memory[Iterable[ConversationMessage], ConversationMessage],
    toolbox: Toolbox,
    stream: bool = False,
    **extra: Any,
) -> AsyncIterator[LMMStreamChunk] | ConversationMessage:
    with ctx.scope("conversation_completion"):
        context: list[LMMContextElement]

        messages: Iterable[ConversationMessage] | Missing = await memory.recall()
        if not_missing(messages):
            context = [
                *(message.as_lmm_context_element() for message in messages),
                message.as_lmm_context_element(),
            ]

        else:
            context = [message.as_lmm_context_element()]

        if stream:
            return ctx.stream(
                _conversation_stream,
                instruction=instruction,
                message=message,
                memory=memory,
                context=context,
                toolbox=toolbox,
                **extra,
            )

        else:
            return await _conversation_completion(
                instruction=instruction,
                message=message,
                memory=memory,
                context=context,
                toolbox=toolbox,
                **extra,
            )


async def _conversation_completion(
    instruction: Instruction | str | None,
    message: ConversationMessage,
    memory: Memory[Iterable[ConversationMessage], ConversationMessage],
    context: list[LMMContextElement],
    toolbox: Toolbox,
    **extra: Any,
) -> ConversationMessage:
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
                await memory.remember(
                    message,
                    response_message,
                )
                return response_message

            case LMMToolRequests() as tool_requests:
                ctx.log_debug("Received conversation tool calls")

                responses: list[LMMToolResponse] = await toolbox.respond_all(tool_requests)

                if direct_content := [
                    response.content for response in responses if response.direct
                ]:
                    response_message: ConversationMessage = ConversationMessage(
                        role="model",
                        created=datetime.now(UTC),
                        content=MultimodalContent.of(*direct_content),
                    )
                    await memory.remember(
                        message,
                        response_message,
                    )
                    return response_message

                else:
                    context.extend([tool_requests, *responses])

        recursion_level += 1  # continue with next recursion level

    raise RuntimeError("LMM exceeded limit of recursive calls")


async def _conversation_stream(
    instruction: Instruction | str | None,
    message: ConversationMessage,
    memory: Memory[Iterable[ConversationMessage], ConversationMessage],
    context: list[LMMContextElement],
    toolbox: Toolbox,
    **extra: Any,
) -> AsyncGenerator[LMMStreamChunk]:
    input_stream = AsyncQueue[LMMStreamInput]()
    input_stream.enqueue(
        LMMStreamChunk.of(
            message.content,
            eod=True,  # we provide single input chunk here
        )
    )
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
                    chunk.content,
                    merge_text=True,
                )
                # on turn end finalize the stream - we are streaming only a single response here
                if chunk.eod:
                    input_stream.finish()
                    response_message: ConversationMessage = ConversationMessage(
                        role="model",
                        created=datetime.now(UTC),
                        content=accumulated_content,
                    )
                    await memory.remember(
                        message,
                        response_message,
                    )

                    yield chunk
                    return  # end of streaming for conversation completion

                else:
                    yield chunk

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
