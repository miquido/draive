from base64 import b64encode
from collections.abc import AsyncGenerator, Sequence
from typing import Any, Literal, cast, overload
from uuid import uuid4

from draive.gemini.client import GeminiClient
from draive.gemini.config import GeminiConfig
from draive.gemini.errors import GeminiException
from draive.gemini.models import (
    GeminiChoice,
    GeminiDataMessageContent,
    GeminiDataReferenceMessageContent,
    GeminiFunctionCallMessageContent,
    GeminiFunctionsTool,
    GeminiFunctionToolSpecification,
    GeminiGenerationResult,
    GeminiMessage,
    GeminiMessageContent,
    GeminiRequestMessage,
    GeminiTextMessageContent,
)
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
    MultimodalContentElement,
    TextContent,
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
    instruction: Instruction | str,
    context: Sequence[LMMContextElement],
    tools: Sequence[ToolSpecification] | None = None,
    tool_requirement: ToolSpecification | bool | None = False,
    output: Literal["text", "json"] = "text",
    stream: Literal[True],
    **extra: Any,
) -> LMMOutputStream: ...


@overload
async def gemini_lmm_invocation(
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
async def gemini_lmm_invocation(
    *,
    instruction: Instruction | str,
    context: Sequence[LMMContextElement],
    tools: Sequence[ToolSpecification] | None = None,
    tool_requirement: ToolSpecification | bool | None = False,
    output: Literal["text", "json"] = "text",
    stream: bool = False,
    **extra: Any,
) -> LMMOutputStream | LMMOutput: ...


async def gemini_lmm_invocation(  # noqa: PLR0913
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
        "gemini_lmm_invocation",
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

        messages: list[GeminiRequestMessage] = [
            _convert_context_element(element=element) for element in context
        ]

        if stream:
            return ctx.stream(
                generator=_generation_stream(
                    client=client,
                    config=config,
                    instruction=Instruction.of(instruction).format(),
                    messages=messages,
                    tools=tools,
                    tool_requirement=tool_requirement,
                ),
            )

        else:
            return await _generate(
                client=client,
                config=config,
                instruction=Instruction.of(instruction).format(),
                messages=messages,
                tools=tools,
                tool_requirement=tool_requirement,
            )


def _convert_content_element(  # noqa: C901, PLR0911
    element: MultimodalContentElement,
) -> dict[str, Any]:
    match element:
        case TextContent() as text:
            return {"text": text.text}

        case ImageURLContent() as image:
            return {
                "fileData": {
                    "mimeType": image.mime_type or "image",
                    "fileUri": image.image_url,
                }
            }

        case ImageBase64Content() as image:
            return {
                "inlineData": {
                    "mimeType": image.mime_type or "image",
                    "data": image.image_base64,
                }
            }

        case ImageDataContent() as image:
            return {
                "inlineData": {
                    "mimeType": image.mime_type or "image",
                    "data": b64encode(image.image_data).decode("utf-8"),
                }
            }

        case AudioURLContent() as audio:
            return {
                "fileData": {
                    "mimeType": audio.mime_type or "audio",
                    "fileUri": audio.audio_url,
                }
            }

        case AudioBase64Content() as audio:
            return {
                "inlineData": {
                    "mimeType": audio.mime_type or "audio",
                    "data": audio.audio_base64,
                }
            }

        case AudioDataContent() as audio:
            return {
                "inlineData": {
                    "mimeType": audio.mime_type or "audio",
                    "data": b64encode(audio.audio_data).decode("utf-8"),
                }
            }

        case VideoURLContent() as video:
            return {
                "fileData": {
                    "mimeType": video.mime_type or "video",
                    "fileUri": video.video_url,
                }
            }

        case VideoBase64Content() as video:
            return {
                "inlineData": {
                    "mimeType": video.mime_type or "video",
                    "data": video.video_base64,
                }
            }

        case VideoDataContent() as video:
            return {
                "inlineData": {
                    "mimeType": video.mime_type or "video",
                    "data": b64encode(video.video_data).decode("utf-8"),
                }
            }

        case DataModel() as data:
            return {"text": data.as_json()}


def _convert_context_element(
    element: LMMContextElement,
) -> GeminiRequestMessage:
    match element:
        case LMMInput() as input:
            return {
                "role": "user",
                "parts": [
                    _convert_content_element(element=element) for element in input.content.parts
                ],
            }

        case LMMCompletion() as completion:
            return {
                "role": "model",
                "parts": [
                    _convert_content_element(element=element)
                    for element in completion.content.parts
                ],
            }

        case LMMToolRequests() as tool_requests:
            return {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "name": request.tool,
                            "args": request.arguments,
                        },
                    }
                    for request in tool_requests.requests
                ],
            }

        case LMMToolResponse() as tool_response:
            return {
                "role": "model",
                "parts": [
                    {
                        "functionResponse": {
                            "name": tool_response.tool,
                            "response": tool_response.content.as_dict(),
                        },
                    }
                ],
            }


def _convert_content_part(  # noqa: PLR0911
    part: GeminiMessageContent,
) -> MultimodalContentElement:
    match part:
        case GeminiTextMessageContent() as text:
            return TextContent(text=text.text)

        case GeminiDataMessageContent() as data:
            mime_type: str = data.data.mime_type
            if mime_type.startswith("image"):
                return ImageBase64Content(
                    mime_type=cast(
                        Literal["image/jpeg", "image/png", "image/gif"] | None,
                        mime_type
                        if mime_type in {"image/jpeg", "image/png", "image/gif"}
                        else None,
                    ),
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
                    mime_type=cast(
                        Literal["image/jpeg", "image/png", "image/gif"] | None,
                        mime_type
                        if mime_type in {"image/jpeg", "image/png", "image/gif"}
                        else None,
                    ),
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
    messages: list[GeminiRequestMessage],
    tools: Sequence[ToolSpecification] | None,
    tool_requirement: ToolSpecification | bool | None,
) -> LMMOutput:
    result: GeminiGenerationResult
    match tool_requirement:
        case None:
            result = await client.generate(
                config=config,
                instruction=instruction,
                messages=messages,
                tools=[],
                require_tools=None,
            )

        case bool(required):
            result = await client.generate(
                config=config,
                instruction=instruction,
                messages=messages,
                tools=[
                    GeminiFunctionsTool(
                        functionDeclarations=[
                            # those models are the same, can safely cast
                            cast(GeminiFunctionToolSpecification, tool["function"])
                            for tool in tools or []
                        ]
                    )
                ],
                require_tools=required,
            )

        case tool:
            assert tool in (tools or []), "Can't suggest a tool without using it"  # nosec: B101
            result = await client.generate(
                config=config,
                instruction=instruction,
                messages=messages,
                tools=[
                    GeminiFunctionsTool(
                        functionDeclarations=[
                            # those models are the same, can safely cast
                            cast(GeminiFunctionToolSpecification, tool["function"])
                            for tool in tools or []
                        ]
                    )
                ],
                require_tools=True,
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

    result_choice: GeminiChoice = result.choices[0]
    result_message: GeminiMessage
    match result_choice.finish_reason:
        case "STOP":
            if message := result_choice.content:
                result_message = message

            else:
                raise GeminiException(
                    "Invalid Gemini response - missing message content: %s", result
                )

        case "MAX_TOKENS":
            raise GeminiException("Gemini response finish caused by token limit: %s", result)

        case "SAFETY":
            raise GeminiException("Gemini response finish caused by safety reason: %s", result)

        case "RECITATION":
            raise GeminiException("Gemini response finish caused by recitation reason: %s", result)

        case "OTHER":
            raise GeminiException("Gemini response finish caused by unknown reason: %s", result)

    message_parts: list[
        GeminiTextMessageContent | GeminiDataReferenceMessageContent | GeminiDataMessageContent
    ] = []
    tool_calls: list[GeminiFunctionCallMessageContent] = []
    for part in result_message.content:
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
        if __debug__ and message_parts:
            ctx.log_debug("Gemini has generated a message and tool calls, ignoring the message...")

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
    messages: list[GeminiRequestMessage],
    tools: Sequence[ToolSpecification] | None,
    tool_requirement: ToolSpecification | bool | None,
) -> AsyncGenerator[LMMOutputStreamChunk, None]:
    ctx.log_debug("Gemini streaming api is not supported yet, using regular response...")
    output: LMMOutput = await _generate(
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
