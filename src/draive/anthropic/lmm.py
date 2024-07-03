from base64 import b64encode
from collections.abc import AsyncGenerator, Sequence
from typing import Any, Literal, cast, overload

from anthropic.types import (
    ImageBlockParam,
    Message,
    MessageParam,
    TextBlock,
    TextBlockParam,
    ToolParam,
    ToolUseBlock,
)

from draive.anthropic.client import AnthropicClient
from draive.anthropic.config import AnthropicConfig
from draive.anthropic.errors import AnthropicException
from draive.metrics import ArgumentsTrace, ResultTrace, TokenUsage
from draive.parameters import DataModel, ToolSpecification
from draive.scope import ctx
from draive.types import (
    AudioBase64Content,
    AudioDataContent,
    AudioURLContent,
    ImageBase64Content,
    ImageDataContent,
    ImageURLContent,
    Instruction,
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
    MultimodalContent,
    MultimodalContentElement,
    TextContent,
    VideoBase64Content,
    VideoDataContent,
    VideoURLContent,
)

__all__ = [
    "anthropic_lmm_invocation",
]


@overload
async def anthropic_lmm_invocation(
    *,
    instruction: Instruction | str,
    context: Sequence[LMMContextElement],
    tools: Sequence[ToolSpecification] | None = None,
    tool_requirement: ToolSpecification | bool | None = False,
    output: Literal["text", "json"] = "text",
    stream: Literal[True],
    **extra: Any,
) -> LMMOutputStream: ...


@overload
async def anthropic_lmm_invocation(
    *,
    instruction: Instruction | str,
    context: Sequence[LMMContextElement],
    tools: Sequence[ToolSpecification] | None = None,
    tool_requirement: ToolSpecification | bool | None = False,
    output: Literal["text", "json"] = "text",
    stream: Literal[False] = False,
    **extra: Any,
) -> LMMOutput: ...


@overload
async def anthropic_lmm_invocation(
    *,
    instruction: Instruction | str,
    context: Sequence[LMMContextElement],
    tools: Sequence[ToolSpecification] | None = None,
    tool_requirement: ToolSpecification | bool | None = False,
    output: Literal["text", "json"] = "text",
    stream: bool = False,
    **extra: Any,
) -> LMMOutputStream | LMMOutput: ...


async def anthropic_lmm_invocation(  # noqa: PLR0913
    *,
    instruction: Instruction | str,
    context: Sequence[LMMContextElement],
    tools: Sequence[ToolSpecification] | None = None,
    tool_requirement: ToolSpecification | bool | None = False,
    output: Literal["text", "json"] = "text",
    stream: bool = False,
    **extra: Any,
) -> LMMOutputStream | LMMOutput:
    with ctx.nested(
        "anthropic_lmm_invocation",
        metrics=[
            ArgumentsTrace.of(
                instruction=instruction,
                context=context,
                tools=tools,
                tool_requirement=tool_requirement,
                output=output,
                stream=stream,
                **extra,
            ),
        ],
    ):
        ctx.log_debug("Requested Anthropic lmm")
        client: AnthropicClient = ctx.dependency(AnthropicClient)
        config: AnthropicConfig = ctx.state(AnthropicConfig).updated(**extra)
        ctx.record(config)

        messages: list[MessageParam] = [
            _convert_context_element(element=element) for element in context
        ]

        if stream:
            return ctx.stream(
                generator=_completion_stream(
                    client=client,
                    config=config,
                    instruction=Instruction.of(instruction).format(),
                    messages=messages,
                    tools=tools,
                    tool_requirement=tool_requirement,
                ),
            )

        else:
            return await _completion(
                client=client,
                config=config,
                instruction=Instruction.of(instruction).format(),
                messages=messages,
                tools=tools,
                tool_requirement=tool_requirement,
            )


def _convert_content_element(  # noqa: C901
    element: MultimodalContentElement,
) -> TextBlockParam | ImageBlockParam:
    match element:
        case TextContent() as text:
            return {
                "type": "text",
                "text": text.text,
            }

        case ImageURLContent() as image:
            # TODO: we could download the media to have data instead
            raise ValueError("Unsupported message content", element)

        case ImageBase64Content() as image:
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image.mime_type or "image/png",
                    "data": image.image_base64,
                },
            }

        case ImageDataContent() as image:
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image.mime_type or "image/png",
                    "data": b64encode(image.image_data).decode("utf-8"),
                },
            }

        case AudioURLContent():
            raise ValueError("Unsupported message content", element)

        case AudioBase64Content():
            raise ValueError("Unsupported message content", element)

        case AudioDataContent():
            raise ValueError("Unsupported message content", element)

        case VideoURLContent():
            raise ValueError("Unsupported message content", element)

        case VideoBase64Content():
            raise ValueError("Unsupported message content", element)

        case VideoDataContent():
            raise ValueError("Unsupported message content", element)

        case DataModel() as data:
            return {
                "type": "text",
                "text": data.as_json(),
            }


def _convert_context_element(
    element: LMMContextElement,
) -> MessageParam:
    match element:
        case LMMInput() as input:
            return {
                "role": "user",
                "content": [
                    _convert_content_element(element=element) for element in input.content.parts
                ],
            }

        case LMMCompletion() as completion:
            # TODO: Anthropic models generating media?
            return {
                "role": "assistant",
                "content": completion.content.as_string(),
            }

        case LMMToolRequests() as tool_requests:
            return {
                "role": "assistant",
                "content": [
                    {
                        "id": request.identifier,
                        "type": "tool_use",
                        "name": request.tool,
                        "input": request.arguments,
                    }
                    for request in tool_requests.requests
                ],
            }

        case LMMToolResponse() as tool_response:
            return {
                "role": "user",
                "content": [
                    {
                        "tool_use_id": tool_response.identifier,
                        "type": "tool_result",
                        "is_error": tool_response.error,
                        "content": [
                            _convert_content_element(element=part)
                            for part in tool_response.content.parts
                        ],
                    }
                ],
            }


async def _completion(  # noqa: PLR0913, PLR0912
    *,
    client: AnthropicClient,
    config: AnthropicConfig,
    instruction: str,
    messages: list[MessageParam],
    tools: Sequence[ToolSpecification] | None,
    tool_requirement: ToolSpecification | bool | None,
) -> LMMOutput:
    completion: Message
    match tool_requirement:
        case None:
            completion = await client.completion(
                config=config,
                instruction=instruction,
                messages=messages,
                tools=[
                    ToolParam(
                        name=tool["function"]["name"],
                        description=tool["function"]["description"],
                        input_schema=cast(
                            dict[str, Any],
                            tool["function"]["parameters"],
                        ),
                    )
                    for tool in tools or []
                ],
                tool_requirement=None,
            )

        case bool(required):
            completion = await client.completion(
                config=config,
                instruction=instruction,
                messages=messages,
                tools=[
                    ToolParam(
                        name=tool["function"]["name"],
                        description=tool["function"]["description"],
                        input_schema=cast(
                            dict[str, Any],
                            tool["function"]["parameters"],
                        ),
                    )
                    for tool in tools or []
                ],
                tool_requirement=required,
            )

        case tool:
            completion = await client.completion(
                config=config,
                instruction=instruction,
                messages=messages,
                tools=[
                    ToolParam(
                        name=tool["function"]["name"],
                        description=tool["function"]["description"],
                        input_schema=cast(
                            dict[str, Any],
                            tool["function"]["parameters"],
                        ),
                    )
                    for tool in tools or []
                ],
                tool_requirement={
                    "type": "tool",
                    "name": tool["function"]["name"],
                },
            )

    ctx.record(
        TokenUsage.for_model(
            config.model,
            input_tokens=completion.usage.input_tokens or 0,
            output_tokens=completion.usage.output_tokens or 0,
        ),
    )

    message_parts: list[TextBlock] = []
    tool_calls: list[ToolUseBlock] = []
    for part in completion.content:
        match part:
            case TextBlock() as text:
                message_parts.append(text)

            case ToolUseBlock() as call:
                tool_calls.append(call)

    match completion.stop_reason:
        case "tool_use":
            if (tool_calls := tool_calls) and (tools := tools):
                ctx.record(ResultTrace.of(tool_calls))
                return LMMToolRequests(
                    requests=[
                        LMMToolRequest(
                            identifier=call.id,
                            tool=call.name,
                            arguments=cast(dict[str, Any], call.input),
                        )
                        for call in tool_calls
                    ]
                )

            else:
                raise AnthropicException("Invalid Anthropic completion", completion)

        case "end_turn":
            if (tool_calls := tool_calls) and (tools := tools):
                ctx.record(ResultTrace.of(tool_calls))
                return LMMToolRequests(
                    requests=[
                        LMMToolRequest(
                            identifier=call.id,
                            tool=call.name,
                            arguments=cast(dict[str, Any], call.input),
                        )
                        for call in tool_calls
                    ]
                )

            else:
                ctx.record(ResultTrace.of(message_parts))
                return LMMCompletion.of(
                    MultimodalContent.of(*[TextContent(text=part.text) for part in message_parts])
                )

        case other:
            raise AnthropicException(f"Unexpected finish reason: {other}")


async def _completion_stream(  # noqa: PLR0913
    *,
    client: AnthropicClient,
    config: AnthropicConfig,
    instruction: str,
    messages: list[MessageParam],
    tools: Sequence[ToolSpecification] | None,
    tool_requirement: ToolSpecification | bool | None,
) -> AsyncGenerator[LMMOutputStreamChunk, None]:
    ctx.log_debug("Anthropic streaming api is not supported yet, using regular response...")
    output: LMMOutput = await _completion(
        client=client,
        config=config,
        instruction=instruction,
        messages=messages,
        tools=tools,
        tool_requirement=tool_requirement,
    )

    match output:
        case LMMCompletion() as completion:
            yield LMMCompletionChunk.of(completion.content)

        case other:
            yield other
