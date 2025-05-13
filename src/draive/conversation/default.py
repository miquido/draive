from asyncio import Task, gather
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from typing import Any, Literal, overload

from haiway import AsyncQueue, ctx

from draive.conversation.types import ConversationMemory, ConversationMessage
from draive.guardrails import GuardrailsModeration
from draive.instructions import Instruction
from draive.lmm import (
    LMM,
    LMMCompletion,
    LMMContextElement,
    LMMStreamChunk,
    LMMToolRequests,
    LMMToolResponses,
)
from draive.lmm.types import LMMToolRequest, LMMToolResponse
from draive.multimodal import Multimodal, MultimodalContent
from draive.tools import Toolbox
from draive.utils import ProcessingEvent
from draive.utils.processing import Processing

__all__ = ("conversation_completion",)


@overload
async def conversation_completion(
    *,
    instruction: Instruction | None,
    input: ConversationMessage | Multimodal,
    memory: ConversationMemory,
    toolbox: Toolbox,
    stream: Literal[False] = False,
    **extra: Any,
) -> ConversationMessage: ...


@overload
async def conversation_completion(
    *,
    instruction: Instruction | None,
    input: ConversationMessage | Multimodal,
    memory: ConversationMemory,
    toolbox: Toolbox,
    stream: Literal[True],
    **extra: Any,
) -> AsyncIterator[LMMStreamChunk | ProcessingEvent]: ...


async def conversation_completion(
    *,
    instruction: Instruction | None,
    input: ConversationMessage | Multimodal,  # noqa: A002
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
                await GuardrailsModeration.check_input(message.content)
                await memory.remember(message)
                context = [
                    *(message.as_lmm_context_element() for message in recalled_messages),
                    message.as_lmm_context_element(),
                ]

            case content:
                message = ConversationMessage.user(
                    created=datetime.now(UTC),
                    content=MultimodalContent.of(content),
                )
                await GuardrailsModeration.check_input(message.content)
                await memory.remember(message)
                context = [
                    *(message.as_lmm_context_element() for message in recalled_messages),
                    message.as_lmm_context_element(),
                ]

        if stream:
            return await _conversation_completion_stream(
                instruction=instruction,
                context=context,
                memory=memory,
                toolbox=toolbox,
            )

        else:
            return await _conversation_completion(
                instruction=instruction,
                context=context,
                memory=memory,
                toolbox=toolbox,
            )


async def _conversation_completion(
    instruction: Instruction | None,
    context: list[LMMContextElement],
    memory: ConversationMemory,
    toolbox: Toolbox,
    **extra: Any,
) -> ConversationMessage:
    recursion_level: int = 0
    while recursion_level <= toolbox.repeated_calls_limit:
        match await LMM.completion(
            instruction=instruction,
            context=context,
            tools=toolbox.available_tools(),
            tool_selection=toolbox.tool_selection(repetition_level=recursion_level),
            output="text",
            **extra,
        ):
            case LMMCompletion() as completion:
                ctx.log_debug("Received conversation result")
                response_message: ConversationMessage = ConversationMessage.model(
                    created=datetime.now(UTC),
                    content=completion.content,
                )

                await memory.remember(response_message)

                return response_message

            case LMMToolRequests() as tool_requests:
                ctx.log_debug("Received conversation tool calls")

                tool_responses: LMMToolResponses = await toolbox.respond_all(tool_requests)

                if direct_content := [
                    response.content
                    for response in tool_responses.responses
                    if response.handling == "direct_result"
                ]:
                    response_message: ConversationMessage = ConversationMessage.model(
                        created=datetime.now(UTC),
                        content=MultimodalContent.of(*direct_content),
                    )
                    await memory.remember(response_message)

                    return response_message

                else:
                    context.extend(
                        [
                            tool_requests,
                            tool_responses,
                        ]
                    )

        recursion_level += 1  # continue with next recursion level

    raise RuntimeError("LMM exceeded limit of tool calls")


async def _conversation_completion_stream(
    instruction: Instruction | None,
    context: list[LMMContextElement],
    memory: ConversationMemory,
    toolbox: Toolbox,
    **extra: Any,
) -> AsyncIterator[LMMStreamChunk | ProcessingEvent]:
    output_queue = AsyncQueue[LMMStreamChunk | ProcessingEvent]()

    async def report_event(
        event: ProcessingEvent,
        /,
    ) -> None:
        output_queue.enqueue(event)

    with ctx.updated(ctx.state(Processing).updated(event_reporting=report_event)):

        async def consume_lmm_output() -> None:
            recursion_level: int = 0
            try:
                while recursion_level <= toolbox.repeated_calls_limit:
                    pending_tool_requests: list[LMMToolRequest] = []
                    pending_tool_responses: list[Task[LMMToolResponse]] = []
                    accumulated_content: MultimodalContent = MultimodalContent.empty

                    async for element in await LMM.completion(
                        instruction=instruction,
                        context=context,
                        tool_selection=toolbox.tool_selection(repetition_level=recursion_level),
                        tools=toolbox.available_tools(),
                        output="text",
                        stream=True,
                        **extra,
                    ):
                        match element:
                            case LMMStreamChunk() as chunk:
                                accumulated_content = accumulated_content.extending(chunk.content)
                                output_queue.enqueue(chunk)

                                if chunk.eod:
                                    # end of streaming for conversation completion
                                    response_message = ConversationMessage.model(
                                        created=datetime.now(UTC),
                                        content=accumulated_content,
                                    )
                                    await memory.remember(response_message)

                                    return output_queue.finish()

                            case LMMToolRequest() as tool_request:
                                # we could start processing immediately
                                pending_tool_requests.append(tool_request)
                                pending_tool_responses.append(
                                    ctx.spawn(toolbox.respond, tool_request)
                                )

                    assert pending_tool_requests  # nosec: B101
                    tool_requests: LMMToolRequests = LMMToolRequests.of(
                        pending_tool_requests,
                        content=accumulated_content or None,
                    )
                    tool_responses: LMMToolResponses = LMMToolResponses.of(
                        await gather(*pending_tool_responses),
                    )
                    if direct_content := [
                        response.content
                        for response in tool_responses.responses
                        if response.handling == "direct_result"
                    ]:
                        tools_content = MultimodalContent.of(*direct_content)
                        output_queue.enqueue(LMMStreamChunk.of(tools_content))
                        response_message = ConversationMessage.model(
                            created=datetime.now(UTC),
                            content=MultimodalContent.of(accumulated_content, tools_content),
                        )
                        await memory.remember(response_message)
                        return output_queue.finish()

                    else:
                        context.extend(
                            [
                                tool_requests,
                                tool_responses,
                            ]
                        )

                    recursion_level += 1  # continue with next recursion level

                raise RuntimeError("LMM exceeded limit of tool calls")

            except BaseException as exc:
                output_queue.finish(exception=exc)
                raise exc

        ctx.spawn(consume_lmm_output)

        return output_queue
