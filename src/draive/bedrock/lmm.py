from base64 import b64decode, b64encode
from collections.abc import AsyncGenerator, Sequence
from typing import Any, Literal, cast, overload

from draive.bedrock.client import BedrockClient
from draive.bedrock.config import BedrockChatConfig
from draive.bedrock.errors import BedrockException
from draive.bedrock.models import ChatCompletionResponse, ChatMessage, ChatMessageContent, ChatTool
from draive.instructions import Instruction
from draive.lmm import LMMToolSelection, ToolSpecification
from draive.metrics import ArgumentsTrace, ResultTrace
from draive.metrics.tokens import TokenUsage
from draive.parameters import DataModel
from draive.scope import ctx
from draive.types import (
    AudioBase64Content,
    AudioURLContent,
    ImageBase64Content,
    ImageURLContent,
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
    VideoURLContent,
)

__all__ = [
    "bedrock_lmm_invocation",
]


@overload
async def bedrock_lmm_invocation(
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
async def bedrock_lmm_invocation(
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
async def bedrock_lmm_invocation(
    *,
    instruction: Instruction | str,
    context: Sequence[LMMContextElement],
    tools: Sequence[ToolSpecification] | None = None,
    tool_selection: LMMToolSelection = "auto",
    output: Literal["text", "json"] = "text",
    stream: bool = False,
    **extra: Any,
) -> LMMOutputStream | LMMOutput: ...


async def bedrock_lmm_invocation(  # noqa: PLR0913
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
        "bedrock_lmm_invocation",
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
        ctx.log_debug("Requested Bedrock lmm")
        client: BedrockClient = ctx.dependency(BedrockClient)  # pyright: ignore[reportDeprecated]
        config: BedrockChatConfig = ctx.state(BedrockChatConfig).updated(**extra)
        ctx.record(config)

        instruction_string: str
        match instruction:
            case str() as string:
                instruction_string = string

            case instruction:
                instruction_string = instruction.format()

        messages: list[ChatMessage] = [_convert_context_element(element) for element in context]

        if messages[-1].get("role") == "assistant":
            del messages[-1]  # ignore prefill

        tools_list: list[ChatTool] = [_convert_tool(tool) for tool in tools or []]
        require_tool: bool
        match tool_selection:
            case "required":
                require_tool = True

            case _:
                require_tool = False

        if stream:
            return ctx.stream(
                _chat_completion_stream(
                    client=client,
                    config=config,
                    instruction=instruction_string,
                    messages=messages,
                    tools_list=tools_list,
                    require_tool=require_tool,
                ),
            )

        else:
            return await _chat_completion(
                client=client,
                config=config,
                instruction=instruction_string,
                messages=messages,
                tools_list=tools_list,
                require_tool=require_tool,
            )


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
    instruction: str,
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
    print(completion)

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


async def _chat_completion_stream(  # noqa: PLR0913
    *,
    client: BedrockClient,
    config: BedrockChatConfig,
    instruction: str,
    messages: list[ChatMessage],
    tools_list: list[ChatTool],
    require_tool: bool,
) -> AsyncGenerator[LMMOutputStreamChunk, None]:
    ctx.log_debug("Bedrock streaming api is not supported yet, using regular response...")
    output: LMMOutput = await _chat_completion(
        client=client,
        config=config,
        instruction=instruction,
        messages=messages,
        tools_list=tools_list,
        require_tool=require_tool,
    )

    match output:
        case LMMCompletion() as completion:
            yield LMMCompletionChunk.of(completion.content)

        case other:
            yield other
