from asyncio import Task, wait
from asyncio.tasks import FIRST_COMPLETED
from collections.abc import AsyncGenerator, AsyncIterator, MutableSet, Sequence
from datetime import UTC, datetime
from typing import Any, Literal, overload
from uuid import UUID, uuid4

from haiway import ctx

from draive.conversation.completion.types import ConversationMemory, ConversationMessage
from draive.conversation.types import (
    ConversationElement,
    ConversationEvent,
    ConversationMessageChunk,
    ConversationStreamElement,
)
from draive.instructions import Instruction
from draive.lmm import (
    LMM,
    LMMCompletion,
    LMMContextElement,
    LMMInstruction,
    LMMStreamChunk,
    LMMToolRequest,
    LMMToolRequests,
    LMMToolResponse,
    LMMToolResponses,
)
from draive.multimodal import MultimodalContent
from draive.tools import Toolbox

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
    async with ctx.scope("conversation_completion"):
        # relying on memory recall correctness
        recalled_messages: Sequence[ConversationElement] = await memory.recall()
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
    formatted_instruction: LMMInstruction | None = Instruction.formatted(instruction)
    tools_turn: int = 0
    result: LMMCompletion
    result_extension: MultimodalContent = MultimodalContent.empty
    while True:
        ctx.log_debug("...requesting completion...")
        match await LMM.completion(
            instruction=formatted_instruction,
            context=context,
            tools=toolbox.available_tools(tools_turn=tools_turn),
            **extra,
        ):
            case LMMCompletion() as completion:
                ctx.log_debug("...received result...")
                if result_extension:
                    result = completion.updated(
                        content=result_extension.appending(completion.content)
                    )

                else:
                    result = completion

                break  # proceed to resolving

            case LMMToolRequests() as tool_requests:
                ctx.log_debug(f"...received tool requests (turn {tools_turn})...")
                # skip tool_requests.content - no need for extra comments
                tool_responses: LMMToolResponses = await toolbox.respond_all(tool_requests)

                if completion := tool_responses.completion(extension=result_extension):
                    ctx.log_debug("...received tools direct result...")
                    result = completion
                    break  # proceed to resolving

                elif extension := tool_responses.completion_extension():
                    ctx.log_debug("...received tools result extension...")
                    result_extension = result_extension.appending(extension)

                ctx.log_debug("...received tools responses...")
                context.extend((tool_requests, tool_responses))

        tools_turn += 1  # continue with next turn

    ctx.log_debug("...finalizing message...")
    response_message: ConversationMessage = ConversationMessage.model(
        created=datetime.now(UTC),
        content=result,
        meta=result.meta,
    )

    ctx.log_debug("...remembering...")
    await memory.remember(input, response_message)

    ctx.log_debug("... response message finished!")
    return response_message


async def _conversation_completion_stream(  # noqa: C901, PLR0915
    instruction: Instruction | None,
    input: ConversationMessage,  # noqa: A002
    context: list[LMMContextElement],
    memory: ConversationMemory,
    toolbox: Toolbox,
    **extra: Any,
) -> AsyncIterator[ConversationStreamElement]:
    # completion message identifier to match chunks and events
    message_identifier: UUID = uuid4()

    async def consume_lmm_output() -> AsyncGenerator[ConversationStreamElement]:  # noqa: C901, PLR0912, PLR0915
        tools_turn: int = 0
        formatted_instruction: LMMInstruction | None = Instruction.formatted(instruction)
        result: MultimodalContent = MultimodalContent.empty
        while True:
            pending_tool_requests: list[LMMToolRequest] = []
            pending_tool_responses: MutableSet[Task[LMMToolResponse]] = set()
            accumulated_content: MultimodalContent = MultimodalContent.empty
            ctx.log_debug("...requesting completion...")
            async for element in await LMM.completion(
                instruction=formatted_instruction,
                context=context,
                tools=toolbox.available_tools(tools_turn=tools_turn),
                output="text",
                stream=True,
                **extra,
            ):
                match element:
                    case LMMStreamChunk() as chunk:
                        ctx.log_debug("...received result chunk...")
                        # we are ending the stream when all tool results are provided
                        # and a chunk is marked as final, final mark before all tool
                        # calls are resolved typically means that there will be no more
                        # tool related content, also models may produce content and tools
                        if chunk.eod and not pending_tool_requests:
                            result = result.appending(accumulated_content.appending(chunk.content))
                            yield ConversationMessageChunk.model(
                                message_identifier=message_identifier,
                                content=chunk.content,
                                eod=True,
                            )

                            break  # proceed to finalization (outer loop)

                        elif chunk.content:  # skip empty chunks
                            accumulated_content = accumulated_content.appending(chunk.content)
                            yield ConversationMessageChunk.model(
                                message_identifier=message_identifier,
                                content=chunk.content,
                                eod=False,
                            )

                    case LMMToolRequest() as tool_request:
                        ctx.log_debug(f"...received tool requests (turn {tools_turn})...")
                        pending_tool_requests.append(tool_request)
                        # start processing immediately
                        pending_tool_responses.add(ctx.spawn(toolbox.respond, tool_request))
                        yield ConversationEvent.of(
                            category="tool.call",
                            meta={
                                "identifier": tool_request.identifier,
                                "tool": tool_request.tool,
                                "status": "started",
                            },
                        )

            if not pending_tool_responses or not pending_tool_requests:
                break  # proceed to finalization

            tool_requests: LMMToolRequests = LMMToolRequests.of(
                pending_tool_requests,
                content=accumulated_content,
            )

            tools_completion: bool = False
            responses: list[LMMToolResponse] = []
            while pending_tool_responses:
                completed, pending_tool_responses = await wait(
                    pending_tool_responses,
                    return_when=FIRST_COMPLETED,
                )
                for response in completed:
                    tool_response: LMMToolResponse = response.result()
                    yield ConversationEvent.of(
                        category="tool.call",
                        meta={
                            "identifier": tool_response.identifier,
                            "tool": tool_response.tool,
                            "status": "completed",
                            "handling": tool_response.handling,
                        },
                    )

                    if tool_response.handling == "completion":
                        ctx.log_debug("...received tools direct result...")
                        yield ConversationMessageChunk.model(
                            message_identifier=message_identifier,
                            content=tool_response.content,
                            eod=False,
                        )

                        tools_completion = True
                        result = result.appending(tool_response.content)

                    elif tool_response.handling == "extension":
                        ctx.log_debug("...received tools result extension...")
                        yield ConversationMessageChunk.model(
                            message_identifier=message_identifier,
                            content=tool_response.content,
                            eod=False,
                        )

                        responses.append(tool_response)
                        result = result.appending(tool_response.content)

                    else:
                        responses.append(tool_response)

            if tools_completion:
                # send empty end of data part
                yield ConversationMessageChunk.model(
                    message_identifier=message_identifier,
                    content=MultimodalContent.empty,
                    eod=True,
                )

                break  # proceed to finalization

            ctx.log_debug("...received tools responses...")
            tool_responses: LMMToolResponses = LMMToolResponses.of(responses)
            context.extend((tool_requests, tool_responses))

            tools_turn += 1  # continue with next turn

        ctx.log_debug("...finalizing message...")
        response_message: ConversationMessage = ConversationMessage.model(
            created=datetime.now(UTC),
            identifier=message_identifier,
            content=result,
        )

        ctx.log_debug("...remembering...")
        await memory.remember(input, response_message)

        ctx.log_debug("... response message finished!")

    return consume_lmm_output()
