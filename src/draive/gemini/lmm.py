from base64 import b64encode
from collections.abc import AsyncGenerator, Sequence
from typing import Any, Literal, cast, overload
from uuid import uuid4

from draive.gemini.client import GeminiClient
from draive.gemini.config import GeminiConfig
from draive.gemini.errors import GeminiException
from draive.gemini.models import (
    GeminiDataMessageContent,
    GeminiDataReferenceMessageContent,
    GeminiFunctionCall,
    GeminiFunctionCallMessageContent,
    GeminiFunctionResponse,
    GeminiFunctionResponseMessageContent,
    GeminiFunctionsTool,
    GeminiFunctionToolSpecification,
    GeminiGenerationResult,
    GeminiMessage,
    GeminiMessageContent,
    GeminiMessageContentBlob,
    GeminiMessageContentReference,
    GeminiTextMessageContent,
)
from draive.metrics import ArgumentsTrace, ResultTrace, TokenUsage
from draive.parameters import ToolSpecification
from draive.scope import ctx
from draive.types import (
    AudioBase64Content,
    AudioDataContent,
    AudioURLContent,
    ImageBase64Content,
    ImageDataContent,
    ImageURLContent,
    LMMCompletion,
    LMMCompletionChunk,
    LMMContextElement,
    LMMInput,
    LMMInstruction,
    LMMOutput,
    LMMOutputStream,
    LMMOutputStreamChunk,
    LMMToolRequest,
    LMMToolRequests,
    LMMToolResponse,
    MultimodalContentElement,
    VideoBase64Content,
    VideoDataContent,
    VideoURLContent,
)

__all__ = [
    "gemini_lmm_invocation",
]


@overload
async def gemini_lmm_invocation(
    *,
    context: Sequence[LMMContextElement],
    tools: Sequence[ToolSpecification] | None = None,
    require_tool: ToolSpecification | bool = False,
    output: Literal["text", "json"] = "text",
    stream: Literal[True],
    **extra: Any,
) -> LMMOutputStream: ...


@overload
async def gemini_lmm_invocation(
    *,
    context: Sequence[LMMContextElement],
    tools: Sequence[ToolSpecification] | None = None,
    require_tool: ToolSpecification | bool = False,
    output: Literal["text", "json"] = "text",
    stream: Literal[False] = False,
    **extra: Any,
) -> LMMOutput: ...


@overload
async def gemini_lmm_invocation(
    *,
    context: Sequence[LMMContextElement],
    tools: Sequence[ToolSpecification] | None = None,
    require_tool: ToolSpecification | bool = False,
    output: Literal["text", "json"] = "text",
    stream: bool = False,
    **extra: Any,
) -> LMMOutputStream | LMMOutput: ...


async def gemini_lmm_invocation(
    *,
    context: Sequence[LMMContextElement],
    tools: Sequence[ToolSpecification] | None = None,
    require_tool: ToolSpecification | bool = False,
    output: Literal["text", "json"] = "text",
    stream: bool = False,
    **extra: Any,
) -> LMMOutputStream | LMMOutput:
    with ctx.nested(
        "gemini_lmm_completion",
        metrics=[
            ArgumentsTrace.of(
                context=context,
                tools=tools,
                require_tool=require_tool,
                output=output,
                stream=stream,
                **extra,
            ),
        ],
    ):
        ctx.log_debug("Requested Gemini lmm")
        client: GeminiClient = ctx.dependency(GeminiClient)
        config: GeminiConfig = ctx.state(GeminiConfig).updated(**extra)
        ctx.record(config)

        match output:
            case "text":
                config = config.updated(response_format="text/plain")

            case "json":
                if tools:
                    ctx.log_warning(
                        "Attempting to use Gemini in JSON mode with tools which is not supported."
                        " Using text mode instead..."
                    )
                    config = config.updated(response_format="text/plain")

                else:
                    config = config.updated(response_format="application/json")

        instruction: str = ""
        messages: list[GeminiMessage] = []
        for element in context:
            match element:
                case LMMInstruction() as instruction_element:
                    instruction += instruction_element.content

                case other:
                    messages.append(_convert_context_element(element=other))

        if stream:
            return ctx.stream(
                generator=_generation_stream(
                    client=client,
                    config=config,
                    instruction=instruction,
                    messages=messages,
                    tools=tools,
                    require_tool=require_tool,
                ),
            )

        else:
            return await _generate(
                client=client,
                config=config,
                instruction=instruction,
                messages=messages,
                tools=tools,
                require_tool=require_tool,
            )


def _convert_content_element(  # noqa: C901, PLR0911
    element: MultimodalContentElement,
) -> GeminiMessageContent:
    match element:
        case str() as string:
            return GeminiTextMessageContent(text=string)

        case ImageURLContent() as image:
            return GeminiDataReferenceMessageContent(
                reference=GeminiMessageContentReference(
                    mime_type=image.mime_type or "image",
                    uri=image.image_url,
                )
            )

        case ImageBase64Content() as image:
            return GeminiDataMessageContent(
                data=GeminiMessageContentBlob(
                    mime_type=image.mime_type or "image",
                    data=image.image_base64,
                )
            )

        case ImageDataContent() as image:
            return GeminiDataMessageContent(
                data=GeminiMessageContentBlob(
                    mime_type=image.mime_type or "image",
                    data=b64encode(image.image_data).decode("utf-8"),
                )
            )

        case AudioURLContent() as audio:
            return GeminiDataReferenceMessageContent(
                reference=GeminiMessageContentReference(
                    mime_type=audio.mime_type or "audio",
                    uri=audio.audio_url,
                )
            )

        case AudioBase64Content() as audio:
            return GeminiDataMessageContent(
                data=GeminiMessageContentBlob(
                    mime_type=audio.mime_type or "audio",
                    data=audio.audio_base64,
                )
            )

        case AudioDataContent() as audio:
            return GeminiDataMessageContent(
                data=GeminiMessageContentBlob(
                    mime_type=audio.mime_type or "audio",
                    data=b64encode(audio.audio_data).decode("utf-8"),
                )
            )

        case VideoURLContent() as video:
            return GeminiDataReferenceMessageContent(
                reference=GeminiMessageContentReference(
                    mime_type=video.mime_type or "video",
                    uri=video.video_url,
                )
            )

        case VideoBase64Content() as video:
            return GeminiDataMessageContent(
                data=GeminiMessageContentBlob(
                    mime_type=video.mime_type or "video",
                    data=video.video_base64,
                )
            )

        case VideoDataContent() as video:
            return GeminiDataMessageContent(
                data=GeminiMessageContentBlob(
                    mime_type=video.mime_type or "video",
                    data=b64encode(video.video_data).decode("utf-8"),
                )
            )


def _convert_context_element(
    element: LMMContextElement,
) -> GeminiMessage:
    match element:
        case LMMInstruction():
            raise ValueError("Instruction has to be processed separately")

        case LMMInput() as input:
            return GeminiMessage(
                role="user",
                content=[
                    _convert_content_element(element=element) for element in input.content.parts
                ],
            )

        case LMMCompletion() as completion:
            return GeminiMessage(
                role="model",
                content=[
                    _convert_content_element(element=element)
                    for element in completion.content.parts
                ],
            )

        case LMMToolRequests() as tool_requests:
            return GeminiMessage(
                role="model",
                content=[
                    GeminiFunctionCallMessageContent(
                        function_call=GeminiFunctionCall(
                            name=request.tool,
                            arguments=request.arguments,
                        ),
                    )
                    for request in tool_requests.requests
                ],
            )

        case LMMToolResponse() as tool_response:
            return GeminiMessage(
                role="model",
                content=[
                    GeminiFunctionResponseMessageContent(
                        function_response=GeminiFunctionResponse(
                            name=tool_response.tool,
                            response=tool_response.content.as_dict(),
                        ),
                    )
                ],
            )


def _convert_content_part(  # noqa: PLR0911
    part: GeminiMessageContent,
) -> MultimodalContentElement:
    match part:
        case GeminiTextMessageContent() as text:
            return text.text

        case GeminiDataMessageContent() as data:
            mime_type: str = data.data.mime_type
            if mime_type.startswith("image"):
                return ImageBase64Content(
                    mime_type=mime_type,
                    image_base64=data.data.data,
                )

            elif mime_type.startswith("audio"):
                return AudioBase64Content(
                    mime_type=mime_type,
                    audio_base64=data.data.data,
                )

            elif mime_type.startswith("video"):
                return VideoBase64Content(
                    mime_type=mime_type,
                    video_base64=data.data.data,
                )

            else:
                raise GeminiException("Unsupported result content data %s", data)

        case GeminiDataReferenceMessageContent() as reference:
            mime_type: str = reference.reference.mime_type
            if mime_type.startswith("image"):
                return ImageURLContent(
                    mime_type=mime_type,
                    image_url=reference.reference.uri,
                )

            elif mime_type.startswith("audio"):
                return AudioURLContent(
                    mime_type=mime_type,
                    audio_url=reference.reference.uri,
                )

            elif mime_type.startswith("video"):
                return VideoURLContent(
                    mime_type=mime_type,
                    video_url=reference.reference.uri,
                )

            else:
                raise GeminiException("Unsupported result content reference %s", reference)

        case other:
            raise GeminiException("Unsupported result content %s", other)


async def _generate(  # noqa: PLR0913, C901, PLR0912
    *,
    client: GeminiClient,
    config: GeminiConfig,
    instruction: str,
    messages: list[GeminiMessage],
    tools: Sequence[ToolSpecification] | None,
    require_tool: ToolSpecification | bool,
) -> LMMOutput:
    result: GeminiGenerationResult
    match require_tool:
        case bool(required):
            result = await client.generate(
                config=config,
                instruction=instruction,
                messages=messages,
                tools=[
                    GeminiFunctionsTool(
                        functions=[
                            GeminiFunctionToolSpecification(
                                name=tool["function"]["name"],
                                description=tool["function"]["description"],
                                parameters=cast(dict[str, Any], tool["function"]["parameters"]),
                            )
                            for tool in tools or []
                        ]
                    )
                ],
                suggest_tools=required,
            )

        case tool:
            assert tool in (tools or []), "Can't suggest a tool without using it"  # nosec: B101
            result = await client.generate(
                config=config,
                instruction=instruction,
                messages=messages,
                tools=[
                    GeminiFunctionsTool(
                        functions=[
                            GeminiFunctionToolSpecification(
                                name=tool["function"]["name"],
                                description=tool["function"]["description"],
                                parameters=cast(dict[str, Any], tool["function"]["parameters"]),
                            )
                        ]
                    )
                ],
                suggest_tools=True,
            )

    if usage := result.usage:
        ctx.record(
            TokenUsage.for_model(
                config.model,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.generated_tokens,
            ),
        )

    if not result.choices:
        raise GeminiException("Invalid Gemini completion - missing messages!", result)

    generated_message: GeminiMessage = result.choices[0].content

    message_parts: list[
        GeminiTextMessageContent | GeminiDataReferenceMessageContent | GeminiDataMessageContent
    ] = []
    tool_calls: list[GeminiFunctionCallMessageContent] = []
    for part in generated_message.content:
        match part:
            case GeminiTextMessageContent() as text:
                message_parts.append(text)

            case GeminiFunctionCallMessageContent() as call:
                tool_calls.append(call)

            case GeminiDataReferenceMessageContent() as reference:
                message_parts.append(reference)

            case GeminiDataMessageContent() as data:
                message_parts.append(data)

            case other:
                raise GeminiException("Invalid Gemini completion part", other)

    if tool_calls and (tools := tools):
        assert not message_parts, "Unexpected content when calling tools"  # nosec: B101
        ctx.record(ResultTrace.of(tool_calls))

        return LMMToolRequests(
            requests=[
                LMMToolRequest(
                    identifier=uuid4().hex,
                    tool=call.function_call.name,
                    arguments=call.function_call.arguments,
                )
                for call in tool_calls
            ]
        )

    elif message_parts:
        ctx.record(ResultTrace.of(message_parts))
        return LMMCompletion.of(*[_convert_content_part(part) for part in message_parts])

    else:
        raise GeminiException("Invalid Gemini completion", result)


async def _generation_stream(  # noqa: PLR0913
    *,
    client: GeminiClient,
    config: GeminiConfig,
    instruction: str,
    messages: list[GeminiMessage],
    tools: Sequence[ToolSpecification] | None,
    require_tool: ToolSpecification | bool,
) -> AsyncGenerator[LMMOutputStreamChunk, None]:
    ctx.log_warning("Gemini streaming api is not supported yet, using regular response...")
    output: LMMOutput = await _generate(
        client=client,
        config=config,
        instruction=instruction,
        messages=messages,
        tools=tools,
        require_tool=require_tool,
    )

    match output:
        case LMMCompletion() as completion:
            yield LMMCompletionChunk.of(completion.content)

        case other:
            yield other
