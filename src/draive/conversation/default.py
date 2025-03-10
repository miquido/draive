from asyncio import CancelledError, Task
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Sequence
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
    LMMToolResponses,
    lmm_invoke,
    lmm_stream,
)
from draive.multimodal import Multimodal, MultimodalContent
from draive.prompts import Prompt
from draive.tools import Toolbox
from draive.utils import Memory, Processing, ProcessingEvent

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


async def _conversation_stream(
    instruction: Instruction | str | None,
    memory: Memory[Sequence[ConversationMessage], ConversationMessage],
    context: list[LMMContextElement],
    toolbox: Toolbox,
    **extra: Any,
) -> AsyncGenerator[LMMStreamChunk | ProcessingEvent]:
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

    output_queue = AsyncQueue[LMMStreamChunk | ProcessingEvent]()
    with Processing.context() as processing_events:
        processing_handler_task: Task[None] = _spawn_processing_handler(
            output_queue=output_queue,
            processing_events=processing_events,
        )

        try:
            _spawn_lmm_handler(
                processing_handler_task=processing_handler_task,
                input_stream=input_stream,
                output_queue=output_queue,
                toolbox=toolbox,
                instruction=instruction,
                context=context,
                **extra,
            )
            accumulated_content: MultimodalContent = MultimodalContent.of()

            async for output in output_queue:
                if isinstance(output, LMMStreamChunk):
                    accumulated_content = accumulated_content.appending(*output.content.parts)
                yield output

            response_message = ConversationMessage(
                role="model",
                created=datetime.now(UTC),
                content=accumulated_content,
            )
            await memory.remember(response_message)
            ctx.record(ResultTrace.of(response_message))

        finally:
            processing_handler_task.cancel()


async def _handle_tool_call(
    request: LMMToolRequest,
    /,
    *,
    toolbox: Toolbox,
    output: AsyncQueue[LMMStreamInput],
) -> None:
    output.enqueue(await toolbox.respond(request))


def _spawn_processing_handler(
    output_queue: AsyncQueue[LMMStreamChunk | ProcessingEvent],
    processing_events: AsyncIterable[ProcessingEvent],
) -> Task[None]:
    async def consume_processing_output() -> None:
        try:
            async for event in processing_events:
                output_queue.enqueue(event)

        except CancelledError:
            pass  # just cancelled

        except BaseException as exc:
            ctx.log_error(
                "Processing events stream failed!",
                exception=exc,
            )
            # catching exceptions to avoid breaking main task of output streaming

    return ctx.spawn(consume_processing_output)


def _spawn_lmm_handler(  # noqa: PLR0913
    processing_handler_task: Task[None],
    input_stream: AsyncQueue[LMMStreamInput],
    output_queue: AsyncQueue[LMMStreamChunk | ProcessingEvent],
    toolbox: Toolbox,
    instruction: Instruction | str | None,
    context: list[LMMContextElement],
    **extra: Any,
) -> Task[None]:
    async def properties_generator() -> AsyncGenerator[LMMStreamProperties, None]:
        # within conversation completion we are not allowing multi turn within one call
        # we might need to refine tool avaliablility passing and recursion/round updates
        recursion_level: int = 0
        while recursion_level < toolbox.repeated_calls_limit:
            yield LMMStreamProperties(
                instruction=instruction,
                tools=toolbox.available_tools(),
                tool_selection=toolbox.tool_selection(repetition_level=recursion_level),
            )

            recursion_level += 1

        raise RuntimeError("LMM exceeded limit of recursive calls")

    async def consume_lmm_output() -> None:
        try:
            async for element in await lmm_stream(
                properties=properties_generator(),
                input=input_stream,
                context=context,
                **extra,
            ):
                match element:
                    case LMMStreamChunk() as chunk:
                        output_queue.enqueue(chunk)

                        # on turn end finalize the stream
                        # - we are streaming only a single response here
                        if chunk.eod:
                            return  # end of streaming for conversation completion

                    case LMMToolRequest() as tool_request:
                        ctx.spawn(
                            _handle_tool_call,
                            tool_request,
                            toolbox=toolbox,
                            output=input_stream,
                        )

        except BaseException as exc:
            output_queue.finish(exception=exc)
            raise exc

        finally:
            processing_handler_task.cancel()
            input_stream.finish()
            output_queue.finish()

    return ctx.spawn(consume_lmm_output)
