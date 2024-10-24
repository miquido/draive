import json
from collections.abc import AsyncGenerator, Sequence
from typing import Any, Literal, cast, overload

from draive.instructions import Instruction
from draive.lmm import LMMToolSelection, ToolSpecification
from draive.metrics import ArgumentsTrace, ResultTrace, TokenUsage
from draive.mistral.client import MistralClient
from draive.mistral.config import MistralChatConfig
from draive.mistral.errors import MistralException
from draive.mistral.models import ChatCompletionResponse, ChatMessage, ChatMessageResponse
from draive.scope import ctx
from draive.types import (
    LMMCompletion,
    LMMCompletionChunk,
    LMMContextElement,
    LMMInput,
    LMMOutput,
    LMMOutputStream,
    LMMOutputStreamChunk,
    LMMToolRequest,
    LMMToolRequests,
    LMMToolResponse,
)

__all__ = [
    "mistral_lmm_invocation",
]


@overload
async def mistral_lmm_invocation(
    *,
    instruction: Instruction | str,
    context: Sequence[LMMContextElement],
    tools: Sequence[ToolSpecification] | None = None,
    tool_selection: LMMToolSelection = "auto",
    output: Literal["text", "json"] = "text",
    stream: Literal[True],
    **extra: Any,
) -> LMMOutputStream: ...


@overload
async def mistral_lmm_invocation(
    *,
    instruction: Instruction | str,
    context: Sequence[LMMContextElement],
    tools: Sequence[ToolSpecification] | None = None,
    tool_selection: LMMToolSelection = "auto",
    output: Literal["text", "json"] = "text",
    stream: Literal[False] = False,
    **extra: Any,
) -> LMMOutput: ...


@overload
async def mistral_lmm_invocation(
    *,
    instruction: Instruction | str,
    context: Sequence[LMMContextElement],
    tools: Sequence[ToolSpecification] | None = None,
    tool_selection: LMMToolSelection = "auto",
    output: Literal["text", "json"] = "text",
    stream: bool = False,
    **extra: Any,
) -> LMMOutputStream | LMMOutput: ...


async def mistral_lmm_invocation(  # noqa: PLR0913
    *,
    instruction: Instruction | str,
    context: Sequence[LMMContextElement],
    tools: Sequence[ToolSpecification] | None = None,
    tool_selection: LMMToolSelection = "auto",
    output: Literal["text", "json"] = "text",
    stream: bool = False,
    **extra: Any,
) -> LMMOutputStream | LMMOutput:
    with ctx.nested(  # pyright: ignore[reportDeprecated]
        "mistral_lmm_invocation",
        metrics=[
            ArgumentsTrace.of(
                instruction=instruction,
                context=context,
                tools=tools,
                tool_selection=tool_selection,
                output=output,
                stream=stream,
                **extra,
            ),
        ],
    ):
        ctx.log_debug("Requested Mistral lmm")
        client: MistralClient = ctx.dependency(MistralClient)  # pyright: ignore[reportDeprecated]
        config: MistralChatConfig = ctx.state(MistralChatConfig).updated(**extra)
        ctx.record(config)

        match output:
            case "text":
                config = config.updated(response_format={"type": "text"})

            case "json":
                if tools:
                    ctx.log_warning(
                        "Attempting to use Mistral in JSON mode with tools which is not supported."
                        " Using text mode instead..."
                    )
                    config = config.updated(response_format={"type": "text"})

                else:
                    config = config.updated(response_format={"type": "json_object"})

        messages: list[ChatMessage] = [
            ChatMessage(
                role="system",
                content=Instruction.of(instruction).format(),
            ),
            *[_convert_context_element(element=element) for element in context],
        ]

        if stream:
            return ctx.stream(
                _chat_completion_stream(
                    client=client,
                    config=config,
                    messages=messages,
                    tools=tools,
                    tool_selection=tool_selection,
                ),
            )

        else:
            return await _chat_completion(
                client=client,
                config=config,
                messages=messages,
                tools=tools,
                tool_selection=tool_selection,
            )


def _convert_context_element(
    element: LMMContextElement,
) -> ChatMessage:
    match element:
        case LMMInput() as input:
            return ChatMessage(
                role="user",
                content=input.content.as_string(),
            )

        case LMMCompletion() as completion:
            return ChatMessage(
                role="assistant",
                content=completion.content.as_string(),
            )

        case LMMToolRequests() as tool_requests:
            return ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "id": request.identifier,
                        "function": {
                            "name": request.tool,
                            "arguments": json.dumps(request.arguments),
                        },
                    }
                    for request in tool_requests.requests
                ],
            )

        case LMMToolResponse() as tool_response:
            return ChatMessage(
                role="tool",
                name=tool_response.tool,
                content=tool_response.content.as_string(),
            )


async def _chat_completion(
    *,
    client: MistralClient,
    config: MistralChatConfig,
    messages: list[ChatMessage],
    tools: Sequence[ToolSpecification] | None,
    tool_selection: LMMToolSelection,
) -> LMMOutput:
    completion: ChatCompletionResponse
    match tool_selection:
        case "auto":
            completion = await client.chat_completion(
                config=config,
                messages=messages,
                tools=cast(
                    list[dict[str, object]],
                    tools,
                ),
                tool_choice="auto",
            )

        case "none":
            completion = await client.chat_completion(
                config=config,
                messages=messages,
                tools=[],
                tool_choice="none",
            )

        case "required":
            completion = await client.chat_completion(
                config=config,
                messages=messages,
                tools=cast(
                    list[dict[str, object]],
                    tools,
                ),
                tool_choice="any",
            )

        case tool:
            assert tool in (tools or []), "Can't suggest a tool without using it"  # nosec: B101
            completion = await client.chat_completion(
                config=config,
                messages=messages,
                tools=cast(
                    list[dict[str, object]],
                    [tool],  # mistral can't be suggested with concrete tool
                ),
                tool_choice="any",
            )

    if usage := completion.usage:
        ctx.record(
            TokenUsage.for_model(
                config.model,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
            ),
        )

    if not completion.choices:
        raise MistralException("Invalid Mistral completion - missing messages!", completion)

    completion_message: ChatMessageResponse = completion.choices[0].message

    if (tool_calls := completion_message.tool_calls) and (tools := tools):
        ctx.record(ResultTrace.of(tool_calls))

        return LMMToolRequests(
            requests=[
                LMMToolRequest(
                    identifier=call.id,
                    tool=call.function.name,
                    arguments=json.loads(call.function.arguments)
                    if isinstance(call.function.arguments, str)
                    else call.function.arguments,
                )
                for call in tool_calls
            ]
        )

    elif message := completion_message.content:
        ctx.record(ResultTrace.of(message))
        match message:
            case str(content):
                return LMMCompletion.of(content)

            # API docs say that it can be only a string in response
            # however library model allows list as well
            case other:
                return LMMCompletion.of(*other)

    else:
        raise MistralException("Invalid Mistral completion", completion)


async def _chat_completion_stream(
    *,
    client: MistralClient,
    config: MistralChatConfig,
    messages: list[ChatMessage],
    tools: Sequence[ToolSpecification] | None,
    tool_selection: LMMToolSelection,
) -> AsyncGenerator[LMMOutputStreamChunk, None]:
    ctx.log_debug("Mistral streaming api is not supported yet, using regular response...")
    output: LMMOutput = await _chat_completion(
        client=client,
        config=config,
        messages=messages,
        tools=tools,
        tool_selection=tool_selection,
    )

    match output:
        case LMMCompletion() as completion:
            yield LMMCompletionChunk.of(completion.content)

        case other:
            yield other
