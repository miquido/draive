from collections.abc import Iterable
from typing import Any, Literal, cast

from haiway import ArgumentsTrace, ResultTrace, ctx

from draive.bedrock.client import BedrockClient
from draive.bedrock.config import BedrockChatConfig
from draive.bedrock.models import ChatCompletionResponse, ChatMessage, ChatMessageContent, ChatTool
from draive.bedrock.types import BedrockException
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
from draive.metrics.tokens import TokenUsage
from draive.multimodal import (
    MediaContent,
    MultimodalContent,
    MultimodalContentElement,
    TextContent,
)
from draive.parameters import DataModel, ParametersSpecification

__all__ = [
    "bedrock_lmm",
]


def bedrock_lmm(
    client: BedrockClient | None = None,
    /,
) -> LMMInvocation:
    client = client or BedrockClient.shared()

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


def _convert_content_element(
    element: MultimodalContentElement,
) -> ChatMessageContent:
    match element:
        case TextContent() as text:
            return {"text": text.text}

        case MediaContent() as media:
            if media.kind != "image" or isinstance(media.source, str):
                raise ValueError("Unsupported message content", media)

            image_format: Literal["png", "jpeg", "gif"]
            match media.media:
                case "image/png":
                    image_format = "png"

                case "image/jpeg":
                    image_format = "jpeg"

                case "image/gif":
                    image_format = "gif"

                case _:
                    raise ValueError("Unsupported message content", media)

            return {
                "image": {
                    "format": image_format,
                    "source": {"bytes": media.source},
                }
            }

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
                media_type: Any
                match data_format:
                    case "png":
                        media_type = "image/png"

                    case "jpeg":
                        media_type = "image/jpeg"

                    case "gif":
                        media_type = "image/gif"

                    case _:  # pyright: ignore[reportUnnecessaryComparison]
                        media_type = "image"

                message_parts.append(
                    MediaContent.data(
                        data,
                        media=media_type,
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
