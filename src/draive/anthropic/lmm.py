from base64 import b64encode
from collections.abc import Iterable
from typing import Any, Literal, cast

from anthropic.types import (
    ImageBlockParam,
    Message,
    MessageParam,
    TextBlock,
    TextBlockParam,
    ToolParam,
    ToolUseBlock,
)
from haiway import ArgumentsTrace, ResultTrace, ctx

from draive.anthropic.client import AnthropicClient
from draive.anthropic.config import AnthropicConfig
from draive.anthropic.types import AnthropicException
from draive.instructions import Instruction
from draive.lmm import (
    LMMCompletion,
    LMMContextElement,
    LMMInput,
    LMMInvocation,
    LMMOutput,
    LMMToolRequest,
    LMMToolRequests,
    LMMToolResponse,
    LMMToolSelection,
    ToolSpecification,
)
from draive.metrics import TokenUsage
from draive.multimodal import MediaContent, MultimodalContent, MultimodalContentElement, TextContent
from draive.parameters import DataModel, ParametersSpecification

__all__ = [
    "anthropic_lmm",
]


def anthropic_lmm(
    client: AnthropicClient | None = None,
    /,
) -> LMMInvocation:
    client = client or AnthropicClient.shared()

    async def lmm_invocation(  # noqa: PLR0913
        *,
        instruction: Instruction | str | None,
        context: Iterable[LMMContextElement],
        tool_selection: LMMToolSelection,
        tools: Iterable[ToolSpecification] | None,
        output: Literal["auto", "text"] | ParametersSpecification,
        prefill: MultimodalContent | None = None,
        **extra: Any,
    ) -> LMMOutput:
        with ctx.scope("anthropic_lmm_invocation"):
            ctx.record(
                ArgumentsTrace.of(
                    instruction=instruction,
                    context=context,
                    prefill=prefill,
                    tool_selection=tool_selection,
                    tools=tools,
                    output=output,
                    **extra,
                )
            )
            config: AnthropicConfig = ctx.state(AnthropicConfig).updated(**extra)
            ctx.record(config)

            if prefill:
                context = [*context, LMMCompletion.of(prefill)]

            return await _completion(
                client=client,
                config=config,
                instruction=Instruction.formatted(instruction),
                messages=[_convert_context_element(element=element) for element in context],
                tools=tools,
                tool_selection=tool_selection,
            )

    return LMMInvocation(invoke=lmm_invocation)


def _convert_content_element(
    element: MultimodalContentElement,
) -> TextBlockParam | ImageBlockParam:
    match element:
        case TextContent() as text:
            return {
                "type": "text",
                "text": text.text,
            }

        case MediaContent() as media:
            if media.kind != "image" or isinstance(media.source, str):
                raise ValueError("Unsupported message content", media)

            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": cast(Any, media.media),
                    "data": b64encode(media.source).decode(),
                },
            }

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


async def _completion(  # noqa: PLR0913, PLR0912, C901
    *,
    client: AnthropicClient,
    config: AnthropicConfig,
    instruction: str | None,
    messages: list[MessageParam],
    tools: Iterable[ToolSpecification] | None,
    tool_selection: LMMToolSelection,
) -> LMMOutput:
    completion: Message
    match tool_selection:
        case "auto":
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
                tool_choice="auto",
            )

        case "none":
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
                tool_choice="none",
            )

        case "required":
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
                tool_choice="any",
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
                tool_choice={
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

    message_parts: list[TextBlock]
    match messages[-1]:
        case {"role": "assistant", "content": str() as content_text}:
            message_parts = [TextBlock(type="text", text=content_text)]

        case {"role": "assistant", "content": content_parts}:
            message_parts = [  # currently supporting only text prefills
                TextBlock(type="text", text=part.text)
                for part in content_parts
                if isinstance(part, TextBlock)
            ]

        case _:
            message_parts = []

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

        case "end_turn" | "stop_sequence":
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
                    MultimodalContent.of(
                        *[TextContent(text=part.text) for part in message_parts],
                    )
                )

        case other:
            raise AnthropicException(f"Unexpected finish reason: {other}")
