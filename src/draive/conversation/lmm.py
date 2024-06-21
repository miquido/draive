from collections.abc import AsyncGenerator, Sequence
from datetime import UTC, datetime
from typing import Any, Literal, overload
from uuid import uuid4

from draive.conversation.model import (
    ConversationMessage,
    ConversationMessageChunk,
    ConversationResponseStream,
)
from draive.helpers import ConstantMemory
from draive.lmm import (
    AnyTool,
    Toolbox,
    lmm_invocation,
)
from draive.scope import ctx
from draive.types import (
    Instruction,
    LMMCompletion,
    LMMCompletionChunk,
    LMMContextElement,
    LMMInput,
    LMMToolRequests,
    LMMToolResponse,
    Memory,
    MultimodalContent,
    MultimodalContentConvertible,
    ToolCallStatus,
)
from draive.utils import Missing, not_missing

__all__: list[str] = [
    "lmm_conversation_completion",
]


@overload
async def lmm_conversation_completion(
    *,
    instruction: Instruction | str,
    input: ConversationMessage | MultimodalContent | MultimodalContentConvertible,  # noqa: A002
    memory: Memory[Sequence[ConversationMessage], ConversationMessage]
    | Sequence[ConversationMessage]
    | None = None,
    tools: Toolbox | Sequence[AnyTool] | None = None,
    stream: Literal[True],
    **extra: Any,
) -> ConversationResponseStream: ...


@overload
async def lmm_conversation_completion(
    *,
    instruction: Instruction | str,
    input: ConversationMessage | MultimodalContent | MultimodalContentConvertible,  # noqa: A002
    memory: Memory[Sequence[ConversationMessage], ConversationMessage]
    | Sequence[ConversationMessage]
    | None = None,
    tools: Toolbox | Sequence[AnyTool] | None = None,
    stream: Literal[False] = False,
    **extra: Any,
) -> ConversationMessage: ...


@overload
async def lmm_conversation_completion(
    *,
    instruction: Instruction | str,
    input: ConversationMessage | MultimodalContent | MultimodalContentConvertible,  # noqa: A002
    memory: Memory[Sequence[ConversationMessage], ConversationMessage]
    | Sequence[ConversationMessage]
    | None = None,
    tools: Toolbox | Sequence[AnyTool] | None = None,
    stream: bool,
    **extra: Any,
) -> ConversationResponseStream | ConversationMessage: ...


async def lmm_conversation_completion(
    *,
    instruction: Instruction | str,
    input: ConversationMessage | MultimodalContent | MultimodalContentConvertible,  # noqa: A002
    memory: Memory[Sequence[ConversationMessage], ConversationMessage]
    | Sequence[ConversationMessage]
    | None = None,
    tools: Toolbox | Sequence[AnyTool] | None = None,
    stream: bool = False,
    **extra: Any,
) -> ConversationResponseStream | ConversationMessage:
    with ctx.nested(
        "lmm_conversation_completion",
    ):
        toolbox: Toolbox
        match tools:
            case None:
                toolbox = Toolbox()

            case Toolbox() as tools:
                toolbox = tools

            case [*tools]:
                toolbox = Toolbox(*tools)

        context: list[LMMContextElement] = []

        conversation_memory: Memory[Sequence[ConversationMessage], ConversationMessage]
        match memory:
            case None:
                conversation_memory = ConstantMemory([])

            case Memory() as memory:
                messages: Sequence[ConversationMessage] | Missing = await memory.recall()
                if not_missing(messages):
                    context.extend(message.as_lmm_context_element() for message in messages)
                conversation_memory = memory

            case [*memory_messages]:
                context.extend(message.as_lmm_context_element() for message in memory_messages)
                conversation_memory = ConstantMemory(memory_messages)

        request_message: ConversationMessage
        match input:
            case ConversationMessage() as message:
                context.append(LMMInput.of(message.content))
                request_message = message

            case content:
                context.append(LMMInput.of(content))
                request_message = ConversationMessage(
                    role="user",
                    created=datetime.now(UTC),
                    content=MultimodalContent.of(content),
                )

        if stream:
            return ctx.stream(
                generator=_lmm_conversation_completion_stream(
                    instruction=instruction,
                    request_message=request_message,
                    conversation_memory=conversation_memory,
                    context=context,
                    toolbox=toolbox,
                    **extra,
                ),
            )

        else:
            return await _lmm_conversation_completion(
                instruction=instruction,
                request_message=request_message,
                conversation_memory=conversation_memory,
                context=context,
                toolbox=toolbox,
                **extra,
            )


async def _lmm_conversation_completion(
    instruction: Instruction | str,
    request_message: ConversationMessage,
    conversation_memory: Memory[Sequence[ConversationMessage], ConversationMessage],
    context: list[LMMContextElement],
    toolbox: Toolbox,
    **extra: Any,
) -> ConversationMessage:
    for recursion_level in toolbox.call_range:
        match await lmm_invocation(
            instruction=instruction,
            context=context,
            tools=toolbox.available_tools(recursion_level=recursion_level),
            require_tool=toolbox.tool_suggestion(recursion_level=recursion_level),
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
                await conversation_memory.remember(
                    request_message,
                    response_message,
                )
                return response_message

            case LMMToolRequests() as tool_requests:
                ctx.log_debug("Received conversation tool calls")
                context.append(tool_requests)
                responses: list[LMMToolResponse] = await toolbox.respond(tool_requests)

                if direct_content := [
                    response.content for response in responses if response.direct
                ]:
                    response_message: ConversationMessage = ConversationMessage(
                        role="model",
                        created=datetime.now(UTC),
                        content=MultimodalContent.of(*direct_content),
                    )
                    await conversation_memory.remember(
                        request_message,
                        response_message,
                    )
                    return response_message

                else:
                    context.extend(responses)

    # fail if we have not provided a result until this point
    raise RuntimeError("Failed to produce conversation completion")


async def _lmm_conversation_completion_stream(
    instruction: Instruction | str,
    request_message: ConversationMessage,
    conversation_memory: Memory[Sequence[ConversationMessage], ConversationMessage],
    context: list[LMMContextElement],
    toolbox: Toolbox,
    **extra: Any,
) -> AsyncGenerator[ConversationMessageChunk | ToolCallStatus, None]:
    response_identifier: str = uuid4().hex
    response_content: MultimodalContent = MultimodalContent.of()  # empty

    for recursion_level in toolbox.call_range:
        async for part in await lmm_invocation(
            instruction=instruction,
            context=context,
            tools=toolbox.available_tools(recursion_level=recursion_level),
            require_tool=toolbox.tool_suggestion(recursion_level=recursion_level),
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
                    assert (  # nosec: B101
                        not response_content
                    ), "Tools and completion message should not be used at the same time"

                    responses: list[LMMToolResponse] = []
                    async for update in toolbox.stream(tool_requests):
                        match update:
                            case LMMToolResponse() as response:
                                responses.append(response)

                            case ToolCallStatus() as status:
                                yield status

                    assert len(responses) == len(  # nosec: B101
                        tool_requests.requests
                    ), "Tool responses count should match requests count"

                    if direct_content := [
                        response.content for response in responses if response.direct
                    ]:
                        response_content = MultimodalContent.of(*direct_content)
                        yield ConversationMessageChunk(
                            identifier=response_identifier,
                            content=response_content,
                        )
                        # exit the loop - we have final result

                    else:
                        context.extend([tool_requests, *responses])
                        break  # request lmm again with tool results using outer loop
        else:
            break  # exit the loop with result

    if response_content:
        ctx.log_debug("Remembering conversation result")
        # remember messages when finishing stream
        await conversation_memory.remember(
            request_message,
            ConversationMessage(
                identifier=response_identifier,
                role="model",
                created=datetime.now(UTC),
                content=response_content,
            ),
        )
    else:
        # fail if we have not provided a result until this point
        raise RuntimeError("Failed to produce conversation completion")
