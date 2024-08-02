from collections.abc import AsyncGenerator, Sequence
from datetime import UTC, datetime
from typing import Any, Literal, overload
from uuid import uuid4

from draive.conversation.model import (
    ConversationMessage,
    ConversationMessageChunk,
    ConversationResponseStream,
)
from draive.instructions import Instruction
from draive.lmm import Toolbox, ToolStatus, lmm_invocation
from draive.scope import ctx
from draive.types import (
    LMMCompletion,
    LMMCompletionChunk,
    LMMContextElement,
    LMMToolRequests,
    LMMToolResponse,
    Memory,
    MultimodalContent,
)
from draive.utils import Missing, not_missing

__all__: list[str] = [
    "lmm_conversation_completion",
]


@overload
async def lmm_conversation_completion(
    *,
    instruction: Instruction | str,
    message: ConversationMessage,
    memory: Memory[Sequence[ConversationMessage], ConversationMessage],
    toolbox: Toolbox,
    stream: Literal[True],
    **extra: Any,
) -> ConversationResponseStream: ...


@overload
async def lmm_conversation_completion(
    *,
    instruction: Instruction | str,
    message: ConversationMessage,
    memory: Memory[Sequence[ConversationMessage], ConversationMessage],
    toolbox: Toolbox,
    stream: Literal[False] = False,
    **extra: Any,
) -> ConversationMessage: ...


@overload
async def lmm_conversation_completion(
    *,
    instruction: Instruction | str,
    message: ConversationMessage,
    memory: Memory[Sequence[ConversationMessage], ConversationMessage],
    toolbox: Toolbox,
    stream: bool,
    **extra: Any,
) -> ConversationResponseStream | ConversationMessage: ...


async def lmm_conversation_completion(
    *,
    instruction: Instruction | str,
    message: ConversationMessage,
    memory: Memory[Sequence[ConversationMessage], ConversationMessage],
    toolbox: Toolbox,
    stream: bool = False,
    **extra: Any,
) -> ConversationResponseStream | ConversationMessage:
    with ctx.nested(
        "lmm_conversation_completion",
    ):
        context: list[LMMContextElement]

        messages: Sequence[ConversationMessage] | Missing = await memory.recall()
        if not_missing(messages):
            context = [
                *(message.as_lmm_context_element() for message in messages),
                message.as_lmm_context_element(),
            ]

        else:
            context = [message.as_lmm_context_element()]

        if stream:
            return ctx.stream(
                _lmm_conversation_completion_stream(
                    instruction=instruction,
                    message=message,
                    memory=memory,
                    context=context,
                    toolbox=toolbox,
                    **extra,
                ),
            )

        else:
            return await _lmm_conversation_completion(
                instruction=instruction,
                message=message,
                memory=memory,
                context=context,
                toolbox=toolbox,
                **extra,
            )


async def _lmm_conversation_completion(
    instruction: Instruction | str,
    message: ConversationMessage,
    memory: Memory[Sequence[ConversationMessage], ConversationMessage],
    context: list[LMMContextElement],
    toolbox: Toolbox,
    **extra: Any,
) -> ConversationMessage:
    recursion_level: int = 0
    while recursion_level <= toolbox.recursion_limit:
        match await lmm_invocation(
            instruction=instruction,
            context=context,
            tools=toolbox.available_tools(),
            tool_selection=toolbox.tool_selection(recursion_level=recursion_level),
            output="text",
            stream=False,
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

                responses: list[LMMToolResponse] = await toolbox.respond(tool_requests)

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


async def _lmm_conversation_completion_stream(
    instruction: Instruction | str,
    message: ConversationMessage,
    memory: Memory[Sequence[ConversationMessage], ConversationMessage],
    context: list[LMMContextElement],
    toolbox: Toolbox,
    **extra: Any,
) -> AsyncGenerator[ConversationMessageChunk | ToolStatus, None]:
    response_identifier: str = uuid4().hex
    response_content: MultimodalContent = MultimodalContent.of()  # empty

    recursion_level: int = 0
    require_callback: bool = True
    while require_callback and recursion_level <= toolbox.recursion_limit:
        require_callback = False
        async for part in await lmm_invocation(
            instruction=instruction,
            context=context,
            tools=toolbox.available_tools(),
            tool_selection=toolbox.tool_selection(recursion_level=recursion_level),
            output="text",
            stream=True,
            **extra,
        ):
            match part:
                case LMMCompletionChunk() as chunk:
                    ctx.log_debug("Received conversation result chunk")
                    response_content = response_content.extending(
                        chunk.content,
                        merge_text=True,
                    )

                    yield ConversationMessageChunk(
                        identifier=response_identifier,
                        content=chunk.content,
                    )
                    # keep yielding parts

                case LMMToolRequests() as tool_requests:
                    ctx.log_debug("Received conversation tool calls")

                    responses: list[LMMToolResponse] = []
                    async for update in toolbox.stream(tool_requests):
                        match update:
                            case LMMToolResponse() as response:
                                responses.append(response)

                            case ToolStatus() as status:
                                yield status

                    assert len(responses) == len(  # nosec: B101
                        tool_requests.requests
                    ), "Tool responses count should match requests count"

                    if direct_content := [
                        response.content for response in responses if response.direct
                    ]:
                        direct_response_content: MultimodalContent = MultimodalContent.of(
                            *direct_content
                        )
                        response_content = response_content.extending(
                            direct_response_content,
                            merge_text=True,
                        )

                        yield ConversationMessageChunk(
                            identifier=response_identifier,
                            content=direct_response_content,
                        )

                    else:
                        context.extend([tool_requests, *responses])
                        require_callback = True  # request lmm again with tool results

        recursion_level += 1  # continue with next recursion level

    ctx.log_debug("Remembering conversation result")
    # remember messages when finishing stream
    await memory.remember(
        message,
        ConversationMessage(
            identifier=response_identifier,
            role="model",
            created=datetime.now(UTC),
            content=response_content,
        ),
    )
