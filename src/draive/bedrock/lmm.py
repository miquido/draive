from base64 import b64decode, b64encode
from collections.abc import Iterable
from typing import Any, Literal, cast

from haiway import ArgumentsTrace, ResultTrace, ctx

from draive.bedrock.client import SHARED, BedrockClient
from draive.bedrock.config import BedrockChatConfig
from draive.bedrock.models import ChatCompletionResponse, ChatMessage, ChatMessageContent, ChatTool
from draive.bedrock.types import BedrockException
from draive.instructions import Instruction
from draive.lmm import LMMInvocation, LMMToolSelection, ToolSpecification
from draive.metrics.tokens import TokenUsage
from draive.parameters import DataModel, ParametersSpecification
from draive.types import (
    AudioBase64Content,
    AudioURLContent,
    ImageBase64Content,
    ImageURLContent,
    LMMCompletion,
    LMMContextElement,
    LMMInput,
    LMMOutput,
    LMMToolRequest,
    LMMToolRequests,
    LMMToolResponse,
    MultimodalContent,
    MultimodalContentElement,
    TextContent,
    VideoBase64Content,
    VideoURLContent,
)

__all__ = [
    "bedrock_lmm",
]


def bedrock_lmm(
    client: BedrockClient = SHARED,
    /,
) -> LMMInvocation:
    async def lmm_invocation(
        *,
        instruction: Instruction | str | None,
        context: Iterable[LMMContextElement],
        tool_selection: LMMToolSelection,
        tools: Iterable[ToolSpecification] | None,
        output: Literal["auto", "text"] | ParametersSpecification,
        **extra: Any,
    ) -> LMMOutput:
        with ctx.scope("bedrock_lmm_invocation"):
            ctx.record(
                ArgumentsTrace.of(
                    instruction=instruction,
                    context=context,
                    tools=tools,
                    tool_selection=tool_selection,
                    output=output,
                    **extra,
                ),
            )
            config: BedrockChatConfig = ctx.state(BedrockChatConfig).updated(**extra)
            ctx.record(config)

            messages: list[ChatMessage] = [_convert_context_element(element) for element in context]

            tools_list: list[ChatTool] = [_convert_tool(tool) for tool in tools or []]
            require_tool: bool
            match tool_selection:
                case "required":
                    require_tool = True

                case _:
                    require_tool = False

            return await _chat_completion(
                client=client,
                config=config,
                instruction=Instruction.formatted(instruction),
                messages=messages,
                tools_list=tools_list,
                require_tool=require_tool,
            )

    return LMMInvocation(invoke=lmm_invocation)


def _convert_content_element(  # noqa: C901
    element: MultimodalContentElement,
) -> ChatMessageContent:
    match element:
        case TextContent() as text:
            return {"text": text.text}

        case ImageURLContent() as image:
            raise ValueError("Unsupported message content", element)

        case ImageBase64Content() as image:
            image_format: Literal["png", "jpeg", "gif"]
            match image.mime_type:
                case "image/png":
                    image_format = "png"

                case "image/jpeg":
                    image_format = "jpeg"

                case "image/gif":
                    image_format = "gif"

                case _:
                    image_format = "png"

            return {
                "image": {
                    "format": image_format,
                    "source": {"bytes": b64decode(image.image_base64)},
                }
            }

        case AudioURLContent():
            raise ValueError("Unsupported message content", element)

        case AudioBase64Content():
            raise ValueError("Unsupported message content", element)

        case VideoURLContent():
            raise ValueError("Unsupported message content", element)

        case VideoBase64Content():
            raise ValueError("Unsupported message content", element)

        case DataModel() as data:
            return {"text": data.as_json()}


def _convert_context_element(
    element: LMMContextElement,
) -> ChatMessage:
    match element:
        case LMMInput() as input:
            return ChatMessage(
                role="user",
                content=[_convert_content_element(part) for part in input.content.parts],
            )

        case LMMCompletion() as completion:
            return ChatMessage(
                role="assistant",
                content=[_convert_content_element(part) for part in completion.content.parts],
            )

        case LMMToolRequests() as requests:
            return ChatMessage(
                role="assistant",
                content=[
                    {
                        "toolUse": {
                            "toolUseId": request.identifier,
                            "name": request.tool,
                            "input": request.arguments,
                        }
                    }
                    for request in requests.requests
                ],
            )

        case LMMToolResponse() as response:
            return ChatMessage(
                role="user",
                content=[
                    {
                        "toolResult": {
                            "toolUseId": response.identifier,
                            "content": [
                                cast(Any, _convert_content_element(part))
                                for part in response.content.parts
                            ],
                            "status": "error" if response.error else "success",
                        },
                    }
                ],
            )


def _convert_tool(tool: ToolSpecification) -> ChatTool:
    return {
        "name": tool["function"]["name"],
        "description": tool["function"]["description"],
        "inputSchema": {"json": tool["function"]["parameters"]},
    }


async def _chat_completion(  # noqa: PLR0913
    *,
    client: BedrockClient,
    config: BedrockChatConfig,
    instruction: str | None,
    messages: list[ChatMessage],
    tools_list: list[ChatTool],
    require_tool: bool,
) -> LMMOutput:
    completion: ChatCompletionResponse = await client.chat_completion(
        config=config,
        instruction=instruction,
        messages=messages,
        tools=tools_list,
        require_tool=require_tool,
    )

    ctx.record(
        TokenUsage.for_model(
            config.model,
            input_tokens=completion["usage"]["inputTokens"],
            output_tokens=completion["usage"]["outputTokens"],
        ),
    )

    message_parts: list[MultimodalContentElement] = []
    tool_calls: list[LMMToolRequest] = []
    for part in completion["output"]["message"]["content"]:
        match part:
            case {"text": str() as text}:
                message_parts.append(TextContent(text=text))

            case {"image": {"format": str() as data_format, "source": {"bytes": bytes() as data}}}:
                mime_type: Any
                match data_format:
                    case "png":
                        mime_type = "image/png"

                    case "jpeg":
                        mime_type = "image/jpeg"

                    case "gif":
                        mime_type = "image/gif"

                    case _:  # pyright: ignore[reportUnnecessaryComparison]
                        mime_type = None

                message_parts.append(
                    ImageBase64Content(
                        mime_type=mime_type,
                        image_base64=b64encode(data).decode(),
                    )
                )

            case {
                "toolUse": {
                    "toolUseId": str() as identifier,
                    "name": str() as tool,
                    "input": arguments,
                }
            }:
                tool_calls.append(
                    LMMToolRequest(
                        identifier=identifier,
                        tool=tool,
                        arguments=arguments,
                    )
                )

            case _:
                pass

    match completion["stopReason"]:
        case "end_turn" | "stop_sequence":
            message_completion = LMMCompletion(content=MultimodalContent.of(*message_parts))
            ctx.record(ResultTrace.of(message_completion))
            return message_completion

        case "tool_use":
            tools_completion = LMMToolRequests(requests=tool_calls)
            ctx.record(ResultTrace.of(tools_completion))
            return tools_completion

        case _:
            raise BedrockException("Invalid Bedrock response")
