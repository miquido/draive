from base64 import b64encode
from collections.abc import Iterable, Sequence
from copy import copy
from typing import Any, Literal, cast, get_args
from uuid import uuid4

from haiway import ArgumentsTrace, ResultTrace, ctx

from draive.gemini.client import GeminiClient
from draive.gemini.config import GeminiConfig
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
from draive.gemini.types import GeminiException
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
from draive.multimodal.media import MediaType
from draive.parameters import DataModel, ParametersSpecification

__all__ = [
    "gemini_lmm",
]


def gemini_lmm(
    client: GeminiClient | None = None,
    /,
) -> LMMInvocation:
    client = client or GeminiClient.shared()

    async def lmm_invocation(
        *,
        instruction: Instruction | str | None,
        context: Iterable[LMMContextElement],
        tool_selection: LMMToolSelection,
        tools: Iterable[ToolSpecification] | None,
        output: Literal["auto", "text"] | ParametersSpecification,
        **extra: Any,
    ) -> LMMOutput:
        with ctx.scope("gemini_lmm_invocation"):
            ctx.record(
                ArgumentsTrace.of(
                    instruction=instruction,
                    context=context,
                    tool_selection=tool_selection,
                    tools=tools,
                    output=output,
                    **extra,
                ),
            )

            config: GeminiConfig = ctx.state(GeminiConfig).updated(**extra)
            ctx.record(config)

            match output:
                case "auto" | "text":
                    config = config.updated(response_format="text/plain")

                case _:  # TODO: utilize json schema within API
                    if tools:
                        ctx.log_warning(
                            "Attempting to use Gemini in JSON mode with tools which is"
                            " not supported. Using text mode instead..."
                        )
                        config = config.updated(response_format="text/plain")

                    else:
                        config = config.updated(response_format="application/json")

            return await _generate(
                client=client,
                config=config,
                instruction=Instruction.formatted(instruction) or "",
                messages=[_convert_context_element(element=element) for element in context],
                tool_selection=tool_selection,
                tools=tools,
            )

    return LMMInvocation(invoke=lmm_invocation)


def _convert_content_element(
    element: MultimodalContentElement,
) -> dict[str, Any]:
    match element:
        case TextContent() as text:
            return {"text": text.text}

        case MediaContent() as media:
            match media.source:
                case str() as reference:
                    return {
                        "fileData": {
                            "mimeType": media.media,
                            "fileUri": reference,
                        }
                    }

                case bytes() as data:
                    return {
                        "inlineData": {
                            "mimeType": media.media,
                            "data": b64encode(data).decode(),
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


def _convert_content_part(
    part: GeminiMessageContent,
) -> MultimodalContentElement:
    match part:
        case GeminiTextMessageContent() as text:
            return TextContent(text=text.text)

        case GeminiDataMessageContent() as data:
            mime_type: str = data.data.mime_type
            if mime_type in get_args(MediaType):
                return MediaContent.base64(
                    data.data.data,
                    media=cast(MediaType, mime_type),
                )

            else:
                raise GeminiException("Unsupported result content data %s", data)

        case GeminiDataReferenceMessageContent() as reference:
            mime_type: str = reference.reference.mime_type
            if mime_type in get_args(MediaType):
                return MediaContent.url(
                    reference.reference.uri,
                    media=cast(MediaType, mime_type),
                )

            else:
                raise GeminiException("Unsupported result content reference %s", reference)

        case other:
            raise GeminiException("Unsupported result content %s", other)


async def _generate(  # noqa: PLR0913, C901, PLR0912, PLR0915
    *,
    client: GeminiClient,
    config: GeminiConfig,
    instruction: str,
    messages: list[GeminiRequestMessage],
    tool_selection: LMMToolSelection,
    tools: Iterable[ToolSpecification] | None,
) -> LMMOutput:
    result: GeminiGenerationResult
    converted_tools: Sequence[GeminiFunctionToolSpecification] = []
    for tool in tools or []:
        tool_function: GeminiFunctionToolSpecification = cast(
            # those models are the same, can safely cast
            GeminiFunctionToolSpecification,
            tool["function"],
        )
        # AIStudio api requires to delete properties if those are empty...
        if "parameters" in tool_function and not tool_function["parameters"]["properties"]:
            tool_function = copy(tool_function)
            del tool_function["parameters"]

        converted_tools.append(tool_function)

    match tool_selection:
        case "auto":
            result = await client.generate(
                config=config,
                instruction=instruction,
                messages=messages,
                tools=[GeminiFunctionsTool(functionDeclarations=converted_tools)],
                tool_calling_mode="AUTO",
            )

        case "none":
            result = await client.generate(
                config=config,
                instruction=instruction,
                messages=messages,
                tools=[],
                tool_calling_mode="NONE",
            )

        case "required":
            result = await client.generate(
                config=config,
                instruction=instruction,
                messages=messages,
                tools=[GeminiFunctionsTool(functionDeclarations=converted_tools)],
                tool_calling_mode="ANY",
            )

        case tool:
            assert tool in (tools or []), "Can't suggest a tool without using it"  # nosec: B101
            tool_function: GeminiFunctionToolSpecification = cast(
                # those models are the same, can safely cast
                GeminiFunctionToolSpecification,
                tool["function"],
            )
            # AIStudio api requires to delete properties if those are empty...
            if "parameters" in tool_function and not tool_function["parameters"]["properties"]:
                tool_function = copy(tool_function)
                del tool_function["parameters"]

            result = await client.generate(
                config=config,
                instruction=instruction,
                messages=messages,
                tools=[GeminiFunctionsTool(functionDeclarations=[tool_function])],
                tool_calling_mode="ANY",
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
        return LMMCompletion.of(
            MultimodalContent.of(
                *[_convert_content_part(part) for part in message_parts],
            )
        )

    else:
        raise GeminiException("Invalid Gemini completion", result)
