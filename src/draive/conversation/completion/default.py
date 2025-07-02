from asyncio import Task, gather
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from typing import Any, Literal, overload
from uuid import UUID, uuid4

from haiway import AsyncQueue, ctx

from draive.conversation.completion.types import ConversationMemory, ConversationMessage
from draive.conversation.types import (
    ConversationElement,
    ConversationEvent,
    ConversationMessageChunk,
    ConversationStreamElement,
)
from draive.guardrails import GuardrailsModeration
from draive.instructions import Instruction
from draive.lmm import (
    LMM,
    LMMCompletion,
    LMMContextElement,
    LMMException,
    LMMInstruction,
    LMMStreamChunk,
    LMMToolRequest,
    LMMToolRequests,
    LMMToolResponse,
    LMMToolResponses,
)
from draive.multimodal import MultimodalContent
from draive.tools import Toolbox
from draive.utils import Processing, ProcessingEvent

__all__ = ("conversation_completion",)


@overload
async def conversation_completion(
    *,
    instruction: Instruction | None,
    input: ConversationMessage,
    memory: ConversationMemory,
    toolbox: Toolbox,
    stream: Literal[False] = False,
    **extra: Any,
) -> ConversationMessage: ...


@overload
async def conversation_completion(
    *,
    instruction: Instruction | None,
    input: ConversationMessage,
    memory: ConversationMemory,
    toolbox: Toolbox,
    stream: Literal[True],
    **extra: Any,
) -> AsyncIterator[ConversationStreamElement]: ...


async def conversation_completion(
    *,
    instruction: Instruction | None,
    input: ConversationMessage,  # noqa: A002
    memory: ConversationMemory,
    toolbox: Toolbox,
    stream: bool = False,
    **extra: Any,
) -> AsyncIterator[ConversationStreamElement] | ConversationMessage:
    with ctx.scope("conversation_completion"):
        # relying on memory recall correctness
        recalled_messages: Sequence[ConversationElement] = await memory.recall()
        await GuardrailsModeration.check_input(input.content)
        context: list[LMMContextElement] = [
            *(
                message.to_lmm_context()
                for message in recalled_messages
                if isinstance(message, ConversationMessage)
            ),
            input.to_lmm_context(),
        ]

        if stream:
            return await _conversation_completion_stream(
                instruction=instruction,
                input=input,
                context=context,
                memory=memory,
                toolbox=toolbox,
            )

        else:
            return await _conversation_completion(
                instruction=instruction,
                input=input,
                context=context,
                memory=memory,
                toolbox=toolbox,
            )


async def _conversation_completion(
    instruction: Instruction | None,
    input: ConversationMessage,  # noqa: A002
    context: list[LMMContextElement],
    memory: ConversationMemory,
    toolbox: Toolbox,
    **extra: Any,
) -> ConversationMessage:
    repetition_level: int = 0
    formatted_instruction: LMMInstruction | None = Instruction.formatted(instruction)
    while True:
        match await LMM.completion(
            instruction=formatted_instruction,
            context=context,
            tools=toolbox.available_tools(repetition_level=repetition_level),
            **extra,
        ):
            case LMMCompletion() as completion:
                ctx.log_debug("Received conversation result")
                response_message: ConversationMessage = ConversationMessage.model(
                    created=datetime.now(UTC),
                    content=completion.content,
                )

                await memory.remember(input, response_message)

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

        repetition_level += 1  # continue with next repetition level


async def _conversation_completion_stream(  # noqa: C901
    instruction: Instruction | None,
    input: ConversationMessage,  # noqa: A002
    context: list[LMMContextElement],
    memory: ConversationMemory,
    toolbox: Toolbox,
    **extra: Any,
) -> AsyncIterator[ConversationStreamElement]:
    output_queue = AsyncQueue[ConversationStreamElement]()
    # completion message identifier to match chunks and events
    message_identifier: UUID = uuid4()

    async def report_event(
        event: ProcessingEvent,
        /,
    ) -> None:
        output_queue.enqueue(
            ConversationEvent.of(
                event.name,
                identifier=event.identifier,
                content=event.content,
                meta=event.meta,
            )
        )

    with ctx.updated(ctx.state(Processing).updated(event_reporting=report_event)):

        async def consume_lmm_output() -> None:
            repetition_level: int = 0
            try:
                formatted_instruction: LMMInstruction | None = Instruction.formatted(instruction)
                while True:
                    pending_tool_requests: list[LMMToolRequest] = []
                    pending_tool_responses: list[Task[LMMToolResponse]] = []
                    accumulated_content: MultimodalContent = MultimodalContent.empty

                    async for element in await LMM.completion(
                        instruction=formatted_instruction,
                        context=context,
                        tools=toolbox.available_tools(repetition_level=repetition_level),
                        output="text",
                        stream=True,
                        **extra,
                    ):
                        match element:
                            case LMMStreamChunk() as chunk:
                                accumulated_content = accumulated_content.appending(chunk.content)
                                # we are ending the stream when all tool results are provided
                                # and a chunk is marked as final, final mark before all tool
                                # calls are resolved typically means that there will be no more
                                # tool related content, also models may produce content and tools
                                if chunk.eod and not pending_tool_requests:
                                    if not accumulated_content:
                                        raise LMMException("Empty completion content")

                                    output_queue.enqueue(  # send the last part
                                        ConversationMessageChunk.model(
                                            chunk.content,
                                            message_identifier=message_identifier,
                                            eod=True,
                                        )
                                    )

                                    # end of streaming for conversation completion
                                    await memory.remember(
                                        input,
                                        ConversationMessage.model(
                                            accumulated_content,
                                            identifier=message_identifier,
                                            created=datetime.now(UTC),
                                        ),
                                    )

                                    return output_queue.finish()

                                elif chunk.content:  # skip empty chunks
                                    output_queue.enqueue(
                                        ConversationMessageChunk.model(
                                            chunk.content,
                                            message_identifier=message_identifier,
                                            eod=False,
                                        )
                                    )

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
                        output_queue.enqueue(
                            ConversationMessageChunk.model(
                                tools_content,
                                message_identifier=message_identifier,
                                eod=True,
                            )
                        )

                        await memory.remember(
                            input,
                            ConversationMessage.model(
                                MultimodalContent.of(accumulated_content, tools_content),
                                identifier=message_identifier,
                                created=datetime.now(UTC),
                            ),
                        )

                        return output_queue.finish()

                    else:
                        context.extend(
                            [
                                tool_requests,
                                tool_responses,
                            ]
                        )

                    repetition_level += 1  # continue with next repetition level

            except BaseException as exc:
                output_queue.finish(exception=exc)
                raise exc

        ctx.spawn(consume_lmm_output)

        return output_queue
