from collections.abc import AsyncIterator, Iterable
from typing import Any, Literal, cast, overload

from haiway import ArgumentsTrace, ResultTrace, ctx

from draive.bedrock.client import BedrockClient
from draive.bedrock.config import BedrockChatConfig
from draive.bedrock.models import ChatCompletionResponse, ChatMessage, ChatMessageContent, ChatTool
from draive.bedrock.types import BedrockException
from draive.instructions import Instruction
from draive.lmm import (
    LMM,
    LMMCompletion,
    LMMContext,
    LMMContextElement,
    LMMInput,
    LMMOutput,
    LMMOutputSelection,
    LMMToolRequest,
    LMMToolRequests,
    LMMToolResponses,
    LMMToolSelection,
    LMMToolSpecification,
)
from draive.lmm.types import LMMStreamOutput
from draive.metrics.tokens import TokenUsage
from draive.multimodal import (
    MediaData,
    MediaReference,
    Multimodal,
    MultimodalContent,
    MultimodalContentElement,
    TextContent,
)
from draive.parameters import DataModel

__all__ = ("bedrock_lmm",)


def bedrock_lmm(
    client: BedrockClient | None = None,
    /,
) -> LMM:
    client = client or BedrockClient.shared()

    @overload
    async def lmm_completion(
        self,
        *,
        instruction: Instruction | None,
        context: LMMContext,
        tool_selection: LMMToolSelection,
        tools: Iterable[LMMToolSpecification] | None,
        prefill: Multimodal | None = None,
        config: BedrockChatConfig | None = None,
        output: LMMOutputSelection,
        stream: Literal[False] = False,
        **extra: Any,
    ) -> LMMOutput: ...

    @overload
    async def lmm_completion(
        self,
        *,
        instruction: Instruction | None,
        context: LMMContext,
        tool_selection: LMMToolSelection,
        tools: Iterable[LMMToolSpecification] | None,
        output: LMMOutputSelection,
        prefill: Multimodal | None = None,
        config: BedrockChatConfig | None = None,
        stream: Literal[True],
        **extra: Any,
    ) -> AsyncIterator[LMMStreamOutput]: ...

    async def lmm_completion(
        self,
        *,
        instruction: Instruction | None,
        context: LMMContext,
        tool_selection: LMMToolSelection,
        tools: Iterable[LMMToolSpecification] | None,
        output: LMMOutputSelection,
        prefill: Multimodal | None = None,
        config: BedrockChatConfig | None = None,
        stream: bool = False,
        **extra: Any,
    ) -> AsyncIterator[LMMStreamOutput] | LMMOutput:
        if stream:
            raise NotImplementedError("bedrock streaming is not implemented yet")

        with ctx.scope("bedrock_lmm_completion"):
            completion_config: BedrockChatConfig = ctx.state(BedrockChatConfig).updated(**extra)
            ctx.record(
                ArgumentsTrace.of(
                    config=completion_config,
                    instruction=instruction,
                    context=context,
                    tools=tools,
                    tool_selection=tool_selection,
                    output=output,
                    **extra,
                ),
            )

            match output:
                case "auto" | "text":
                    pass

                case "image":
                    raise NotImplementedError("image output is not supported by bedrock")

                case "audio":
                    raise NotImplementedError("audio output is not supported by bedrock")

                case "video":
                    raise NotImplementedError("video output is not supported by bedrock")

                case _:
                    raise NotImplementedError("model output is not supported by bedrock")

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
                config=completion_config,
                instruction=Instruction.formatted(instruction),
                messages=messages,
                tools_list=tools_list,
                require_tool=require_tool,
            )

    return LMM(completing=lmm_completion)


def _convert_content_element(
    element: MultimodalContentElement,
) -> ChatMessageContent:
    match element:
        case TextContent() as text:
            return {"text": text.text}

        case MediaData() as media_data:
            if media_data.kind != "image":
                raise ValueError("Unsupported message content", media_data)

            image_format: Literal["png", "jpeg", "gif"]
            match media_data.media:
                case "image/png":
                    image_format = "png"

                case "image/jpeg":
                    image_format = "jpeg"

                case "image/gif":
                    image_format = "gif"

                case _:
                    raise ValueError("Unsupported message content", media_data)

            return {
                "image": {
                    "format": image_format,
                    "source": {"bytes": media_data.data},
                }
            }

        case MediaReference() as media_reference:
            raise ValueError("Unsupported message content", media_reference)

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

        case LMMToolResponses() as tool_responses:
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
                    for response in tool_responses.responses
                ],
            )


def _convert_tool(tool: LMMToolSpecification) -> ChatTool:
    return {
        "name": tool["name"],
        "description": tool["description"] or "",
        "inputSchema": {"json": tool["parameters"]},
    }


async def _chat_completion(
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
            cached_tokens=None,
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
                    MediaData.of(
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
            message_completion = LMMCompletion.of(MultimodalContent.of(*message_parts))
            ctx.record(ResultTrace.of(message_completion))
            return message_completion

        case "tool_use":
            tools_completion = LMMToolRequests(requests=tool_calls)
            ctx.record(ResultTrace.of(tools_completion))
            return tools_completion

        case _:
            raise BedrockException("Invalid Bedrock response")
