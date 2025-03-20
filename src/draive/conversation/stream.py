from asyncio import Task
from collections.abc import AsyncGenerator, Sequence
from datetime import UTC, datetime
from typing import Any

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
    lmm_stream,
)
from draive.multimodal import Multimodal, MultimodalContent
from draive.prompts import Prompt
from draive.tools import Toolbox
from draive.utils import Processing, ProcessingEvent

__all__ = [
    "conversation_stream",
]


async def conversation_stream(
    *,
    instruction: Instruction | str | None,
    input: ConversationMessage | Prompt | Multimodal,  # noqa: A002
    memory: ConversationMemory,
    toolbox: Toolbox,
    **extra: Any,
) -> AsyncGenerator[LMMStreamChunk | ProcessingEvent]:
    recalled_messages: Sequence[ConversationMessage] = await memory.recall()
    context: list[LMMContextElement]
    match input:
        case ConversationMessage() as message:
            await memory.remember(message)
            context = [
                *(recalled.as_lmm_context_element() for recalled in recalled_messages),
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
                *(recalled.as_lmm_context_element() for recalled in recalled_messages),
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
                *(recalled.as_lmm_context_element() for recalled in recalled_messages),
                message.as_lmm_context_element(),
            ]

    ctx.record(
        ArgumentsTrace.of(
            instruction=instruction,
            context=context,
        )
    )

    if not isinstance(context[-1], LMMInput):
        raise ValueError(f"Streaming input has to end with LMMInput, received {type(context[-1])}")

    input_queue = AsyncQueue[LMMStreamInput](
        LMMStreamChunk.of(
            context[-1].content,
            eod=True,  # we provide single input chunk through this interface
        )
    )
    # we are using last element as stream input, we have to remove it from context
    del context[-1]

    output_queue = AsyncQueue[LMMStreamChunk | ProcessingEvent]()

    async def report_event(
        event: ProcessingEvent,
        /,
    ) -> None:
        output_queue.enqueue(event)

    with ctx.updated(ctx.state(Processing).updated(event_reporting=report_event)):
        lmm_task = _spawn_lmm_handler(
            input_queue=input_queue,
            output_queue=output_queue,
            toolbox=toolbox,
            instruction=instruction,
            context=context,
            **extra,
        )

        async for output in output_queue:
            yield output

        response_message = ConversationMessage(
            role="model",
            created=datetime.now(UTC),
            content=await lmm_task,
        )
        await memory.remember(response_message)
        ctx.record(ResultTrace.of(response_message))


async def _handle_tool_call(
    request: LMMToolRequest,
    /,
    *,
    toolbox: Toolbox,
    output: AsyncQueue[LMMStreamInput],
) -> None:
    output.enqueue(await toolbox.respond(request))


def _spawn_lmm_handler(
    input_queue: AsyncQueue[LMMStreamInput],
    output_queue: AsyncQueue[LMMStreamChunk | ProcessingEvent],
    toolbox: Toolbox,
    instruction: Instruction | str | None,
    context: list[LMMContextElement],
    **extra: Any,
) -> Task[MultimodalContent]:
    async def properties_generator() -> AsyncGenerator[LMMStreamProperties, None]:
        # within conversation completion we are not allowing multi turn within one call
        # we might need to refine tool avaliablility passing and recursion/round updates
        recursion_level: int = 0
        while recursion_level < toolbox.repeated_calls_limit:
            if input_queue.is_finished or output_queue.is_finished:
                raise StopAsyncIteration()  # we are no longer streaming

            yield LMMStreamProperties(
                instruction=instruction,
                tools=toolbox.available_tools(),
                tool_selection=toolbox.tool_selection(repetition_level=recursion_level),
            )

            recursion_level += 1

        raise RuntimeError("LMM exceeded limit of recursive calls")

    async def consume_lmm_output() -> MultimodalContent:
        try:
            accumulated_content: MultimodalContent = MultimodalContent.empty
            async for element in await lmm_stream(
                properties=properties_generator(),
                input=input_queue,
                context=context,
                **extra,
            ):
                match element:
                    case LMMStreamChunk() as chunk:
                        accumulated_content = accumulated_content.extending(chunk.content)
                        output_queue.enqueue(chunk)

                        if chunk.eod:
                            # on turn end finalize the stream
                            # - we are streaming only a single response here
                            input_queue.finish()
                            output_queue.finish()
                            # end of streaming for conversation completion
                            return accumulated_content

                    case LMMToolRequest() as tool_request:
                        ctx.spawn(
                            _handle_tool_call,
                            tool_request,
                            toolbox=toolbox,
                            output=input_queue,
                        )

        except BaseException as exc:
            input_queue.finish()
            output_queue.finish(exception=exc)
            raise exc

        return accumulated_content

    return ctx.spawn(consume_lmm_output)
