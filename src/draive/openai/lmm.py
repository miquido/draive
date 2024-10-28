import json
from collections.abc import Iterable
from typing import Any, Literal, cast

from haiway import ArgumentsTrace, ResultTrace, ctx, not_missing
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionContentPartParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from draive.instructions import Instruction
from draive.lmm import LMMInvocation, LMMToolSelection, ToolSpecification
from draive.metrics import TokenUsage
from draive.openai.client import SHARED, OpenAIClient
from draive.openai.config import OpenAIChatConfig, OpenAISystemFingerprint
from draive.openai.types import OpenAIException
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
    "openai_lmm",
]


def openai_lmm(
    client: OpenAIClient = SHARED,
    /,
) -> LMMInvocation:
    async def openai_lmm_invocation(  # noqa: PLR0913
        *,
        instruction: Instruction | str | None,
        context: Iterable[LMMContextElement],
        prefill: MultimodalContent | None,
        tool_selection: LMMToolSelection,
        tools: Iterable[ToolSpecification] | None,
        output: Literal["auto", "text"] | ParametersSpecification,
        **extra: Any,
    ) -> LMMOutput:
        with ctx.scope("openai_lmm_invocation"):
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
            config: OpenAIChatConfig = ctx.state(OpenAIChatConfig).updated(**extra)
            ctx.record(config)

            match output:
                case "auto" | "text":
                    config = config.updated(response_format={"type": "text"})

                case _:  # TODO: utilize json schema within api
                    config = config.updated(response_format={"type": "json_object"})

            messages: list[ChatCompletionMessageParam] = [
                _convert_context_element(config=config, element=element) for element in context
            ]

            if instruction:
                messages = [
                    {
                        "role": "system",
                        "content": Instruction.of(instruction).format(),
                    },
                    *messages,
                ]

            if messages[-1].get("role") == "assistant":
                del messages[-1]  # OpenAI does not support prefill, simply ignore it

            return await _chat_completion(
                client=client,
                config=config,
                messages=messages,
                tools=tools,
                tool_selection=tool_selection,
            )

    return openai_lmm_invocation


def _convert_content_element(
    element: MultimodalContentElement,
    config: OpenAIChatConfig,
) -> ChatCompletionContentPartParam:
    match element:
        case TextContent() as text:
            return {
                "type": "text",
                "text": text.text,
            }

        case ImageURLContent() as image:
            return {
                "type": "image_url",
                "image_url": {
                    "url": image.image_url,
                    "detail": cast(Literal["auto", "low", "high"], config.vision_details)
                    if not_missing(config.vision_details)
                    else "auto",
                },
            }

        case ImageBase64Content() as image:
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image.mime_type or 'image/jpeg'};base64,{image.image_base64}",
                    "detail": cast(Literal["auto", "low", "high"], config.vision_details)
                    if not_missing(config.vision_details)
                    else "auto",
                },
            }

        case AudioURLContent():
            # TODO: OpenAI models with audio?
            raise ValueError("Unsupported message content", element)

        case AudioBase64Content():
            # TODO: we could upload media using openAI endpoint to have url instead
            raise ValueError("Unsupported message content", element)

        case VideoURLContent():
            # TODO: OpenAI models with video?
            raise ValueError("Unsupported message content", element)

        case VideoBase64Content():
            # TODO: we could upload media using openAI endpoint to have url instead
            raise ValueError("Unsupported message content", element)

        case DataModel() as data:
            return {
                "type": "text",
                "text": data.as_json(),
            }


def _convert_context_element(
    element: LMMContextElement,
    config: OpenAIChatConfig,
) -> ChatCompletionMessageParam:
    match element:
        case LMMInput() as input:
            return {
                "role": "user",
                "content": [
                    _convert_content_element(
                        element=element,
                        config=config,
                    )
                    for element in input.content.parts
                ],
            }

        case LMMCompletion() as completion:
            # TODO: OpenAI models generating media?
            return {
                "role": "assistant",
                "content": completion.content.as_string(),
            }

        case LMMToolRequests() as tool_requests:
            return {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": request.identifier,
                        "type": "function",
                        "function": {
                            "name": request.tool,
                            "arguments": json.dumps(request.arguments),
                        },
                    }
                    for request in tool_requests.requests
                ],
            }

        case LMMToolResponse() as tool_response:
            return {
                "role": "tool",
                "tool_call_id": tool_response.identifier,
                "content": tool_response.content.as_string(),
            }


async def _chat_completion(  # noqa: C901, PLR0912
    *,
    client: OpenAIClient,
    config: OpenAIChatConfig,
    messages: list[ChatCompletionMessageParam],
    tools: Iterable[ToolSpecification] | None,
    tool_selection: LMMToolSelection,
) -> LMMOutput:
    completion: ChatCompletion
    match tool_selection:
        case "auto":
            completion = await client.chat_completion(
                config=config,
                messages=messages,
                tools=cast(
                    list[ChatCompletionToolParam],
                    tools,
                ),
                tool_choice="auto",
            )

        case "none":
            completion = await client.chat_completion(
                config=config,
                messages=messages,
                tools=None,
                tool_choice="none",
            )

        case "required":
            completion = await client.chat_completion(
                config=config,
                messages=messages,
                tools=cast(
                    list[ChatCompletionToolParam],
                    tools,
                ),
                tool_choice="required",
            )

        case tool:
            completion = await client.chat_completion(
                config=config,
                messages=messages,
                tools=cast(
                    list[ChatCompletionToolParam],
                    tools,
                ),
                tool_choice={
                    "type": "function",
                    "function": {
                        "name": tool["function"]["name"],
                    },
                },
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
        raise OpenAIException("Invalid OpenAI completion - missing messages!", completion)

    if fingerprint := completion.system_fingerprint:
        ctx.record(OpenAISystemFingerprint(system_fingerprint=fingerprint))

    completion_message: ChatCompletionMessage = completion.choices[0].message
    match completion.choices[0].finish_reason:
        case "tool_calls":
            if (tool_calls := completion_message.tool_calls) and (tools := tools):
                ctx.record(ResultTrace.of(tool_calls))
                return LMMToolRequests(
                    requests=[
                        LMMToolRequest(
                            identifier=call.id,
                            tool=call.function.name,
                            arguments=json.loads(call.function.arguments),
                        )
                        for call in tool_calls
                    ]
                )

            else:
                raise OpenAIException("Invalid OpenAI completion", completion)

        case "stop":
            if (tool_calls := completion_message.tool_calls) and (tools := tools):
                ctx.record(ResultTrace.of(tool_calls))
                return LMMToolRequests(
                    requests=[
                        LMMToolRequest(
                            identifier=call.id,
                            tool=call.function.name,
                            arguments=json.loads(call.function.arguments),
                        )
                        for call in tool_calls
                    ]
                )

            elif content := completion_message.content:
                ctx.record(ResultTrace.of(content))
                return LMMCompletion.of(content)

            else:
                raise OpenAIException("Invalid OpenAI completion", completion)

        case other:
            raise OpenAIException(f"Unexpected finish reason: {other}")
