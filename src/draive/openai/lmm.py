import json
from collections.abc import AsyncGenerator, Sequence
from typing import Any, Literal, cast, overload
from uuid import uuid4

from openai import AsyncStream as OpenAIAsyncStream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionContentPartParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDeltaToolCall

from draive.metrics import ArgumentsTrace, ResultTrace, TokenUsage
from draive.openai.client import OpenAIClient
from draive.openai.config import OpenAIChatConfig, OpenAISystemFingerprint
from draive.openai.errors import OpenAIException
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
from draive.utils import not_missing

__all__ = [
    "openai_lmm_invocation",
]


@overload
async def openai_lmm_invocation(
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
async def openai_lmm_invocation(
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
async def openai_lmm_invocation(
    *,
    instruction: Instruction | str,
    context: Sequence[LMMContextElement],
    tools: Sequence[ToolSpecification] | None = None,
    tool_requirement: ToolSpecification | bool | None = False,
    output: Literal["text", "json"] = "text",
    stream: bool = False,
    **extra: Any,
) -> LMMOutputStream | LMMOutput: ...


async def openai_lmm_invocation(  # noqa: PLR0913
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
        "openai_lmm_invocation",
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
        ctx.log_debug("Requested OpenAI lmm")
        client: OpenAIClient = ctx.dependency(OpenAIClient)
        config: OpenAIChatConfig = ctx.state(OpenAIChatConfig).updated(**extra)
        ctx.record(config)

        match output:
            case "text":
                config = config.updated(response_format={"type": "text"})

            case "json":
                config = config.updated(response_format={"type": "json_object"})

        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": Instruction.of(instruction).format(),
            },
            *[_convert_context_element(config=config, element=element) for element in context],
        ]

        if stream:
            return ctx.stream(
                generator=_chat_completion_stream(
                    client=client,
                    config=config,
                    messages=messages,
                    tools=tools,
                    tool_requirement=tool_requirement,
                ),
            )

        else:
            return await _chat_completion(
                client=client,
                config=config,
                messages=messages,
                tools=tools,
                tool_requirement=tool_requirement,
            )


def _convert_content_element(  # noqa: C901
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
            # TODO: we could upload media using openAI endpoint to have url instead
            return {
                "type": "text",
                "text": image.image_description or "MISSING IMAGE",
            }

        case ImageDataContent() as image:
            # TODO: we could upload media using openAI endpoint to have url instead
            return {
                "type": "text",
                "text": image.image_description or "MISSING IMAGE",
            }

        case AudioURLContent():
            # TODO: OpenAI models with audio?
            raise ValueError("Unsupported message content", element)

        case AudioBase64Content():
            # TODO: we could upload media using openAI endpoint to have url instead
            raise ValueError("Unsupported message content", element)

        case AudioDataContent():
            # TODO: we could upload media using openAI endpoint to have url instead
            raise ValueError("Unsupported message content", element)

        case VideoURLContent():
            # TODO: OpenAI models with video?
            raise ValueError("Unsupported message content", element)

        case VideoBase64Content():
            # TODO: we could upload media using openAI endpoint to have url instead
            raise ValueError("Unsupported message content", element)

        case VideoDataContent():
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


async def _chat_completion(  # noqa: PLR0912, C901
    *,
    client: OpenAIClient,
    config: OpenAIChatConfig,
    messages: list[ChatCompletionMessageParam],
    tools: Sequence[ToolSpecification] | None,
    tool_requirement: ToolSpecification | bool | None,
) -> LMMOutput:
    completion: ChatCompletion
    match tool_requirement:
        case None:
            completion = await client.chat_completion(
                config=config,
                messages=messages,
                tools=[],
                tool_requirement=None,
            )

        case bool(required):
            completion = await client.chat_completion(
                config=config,
                messages=messages,
                tools=cast(
                    list[ChatCompletionToolParam],
                    tools,
                ),
                tool_requirement=required,
            )

        case tool:
            completion = await client.chat_completion(
                config=config,
                messages=messages,
                tools=cast(
                    list[ChatCompletionToolParam],
                    tools,
                ),
                tool_requirement={
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
            if content := completion_message.content:
                ctx.record(ResultTrace.of(content))
                # TODO: OpenAI models generating media?
                return LMMCompletion.of(content)

            elif (tool_calls := completion_message.tool_calls) and (tools := tools):
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

        case other:
            raise OpenAIException(f"Unexpected finish reason: {other}")


async def _chat_completion_stream(  # noqa: C901, PLR0912, PLR0915
    *,
    client: OpenAIClient,
    config: OpenAIChatConfig,
    messages: list[ChatCompletionMessageParam],
    tools: Sequence[ToolSpecification] | None,
    tool_requirement: ToolSpecification | bool | None,
) -> AsyncGenerator[LMMOutputStreamChunk, None]:
    completion_stream: OpenAIAsyncStream[ChatCompletionChunk]
    match tool_requirement:
        case None:
            completion_stream = await client.chat_completion(
                config=config,
                messages=messages,
                tools=[],
                tool_requirement=None,
                stream=True,
            )

        case bool() as required:
            completion_stream = await client.chat_completion(
                config=config,
                messages=messages,
                tools=cast(
                    list[ChatCompletionToolParam],
                    tools,
                ),
                tool_requirement=required,
                stream=True,
            )

        case tool:
            assert tool in (tools or []), "Can't suggest a tool without using it"  # nosec: B101
            completion_stream = await client.chat_completion(
                config=config,
                messages=messages,
                tools=cast(
                    list[ChatCompletionToolParam],
                    tools,
                ),
                tool_requirement={
                    "type": "function",
                    "function": {
                        "name": tool["function"]["name"],
                    },
                },
                stream=True,
            )

    accumulated_completion: str = ""
    requested_tool_calls: list[ChoiceDeltaToolCall] = []
    async for part in completion_stream:
        if choices := part.choices:  # usage part does not contain choices
            # we are always requesting single result - no need to take care of indices
            element: Choice = choices[0]
            if element.delta.content is not None:
                part_text: str = element.delta.content
                if not part_text:
                    continue  # skip empty parts
                accumulated_completion += part_text
                # TODO: OpenAI models generating media?
                yield LMMCompletionChunk.of(part_text)

            elif tool_calls := element.delta.tool_calls:
                # tool calls come in parts, we have to merge them manually
                for call in tool_calls:
                    try:
                        tool_call: ChoiceDeltaToolCall = next(
                            tool_call
                            for tool_call in requested_tool_calls
                            if tool_call.index == call.index
                        )

                        if call.id:
                            if tool_call.id is not None:
                                tool_call.id += call.id
                            else:
                                tool_call.id = call.id
                        else:
                            pass

                        if call.function is None:
                            continue

                        if tool_call.function is None:
                            tool_call.function = call.function
                            continue

                        if call.function.name:
                            if tool_call.function.name is not None:
                                tool_call.function.name += call.function.name
                            else:
                                tool_call.function.name = call.function.name
                        else:
                            pass

                        if call.function.arguments:
                            if tool_call.function.arguments is not None:
                                tool_call.function.arguments += call.function.arguments
                            else:
                                tool_call.function.arguments = call.function.arguments
                        else:
                            pass

                    except (StopIteration, StopAsyncIteration):
                        requested_tool_calls.append(call)

            elif finish_reason := element.finish_reason:
                match finish_reason:
                    case "tool_calls":
                        ctx.record(ResultTrace.of(requested_tool_calls))
                        yield LMMToolRequests(
                            requests=[
                                LMMToolRequest(
                                    identifier=call.id or uuid4().hex,
                                    tool=call.function.name,
                                    arguments=json.loads(call.function.arguments)
                                    if call.function.arguments
                                    else {},
                                )
                                for call in requested_tool_calls
                                if call.function and call.function.name
                            ]
                        )

                    case "stop":
                        if requested_tool_calls:
                            if accumulated_completion:
                                ctx.record(
                                    ResultTrace.of((accumulated_completion, requested_tool_calls))
                                )
                            else:
                                ctx.record(ResultTrace.of(requested_tool_calls))

                            yield LMMToolRequests(
                                requests=[
                                    LMMToolRequest(
                                        identifier=call.id or uuid4().hex,
                                        tool=call.function.name,
                                        arguments=json.loads(call.function.arguments)
                                        if call.function.arguments
                                        else {},
                                    )
                                    for call in requested_tool_calls
                                    if call.function and call.function.name
                                ]
                            )

                        else:
                            ctx.record(ResultTrace.of(accumulated_completion))

                    case other:
                        raise OpenAIException(f"Unexpected finish reason: {other}")

            else:
                ctx.log_warning("Unexpected OpenAI streaming part: %s", part)

        elif usage := part.usage:  # record usage if able (expected in the last part)
            ctx.record(
                TokenUsage.for_model(
                    config.model,
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                ),
            )

            if fingerprint := part.system_fingerprint:
                ctx.record(OpenAISystemFingerprint(system_fingerprint=fingerprint))

        else:
            ctx.log_warning("Unexpected OpenAI streaming part: %s", part)
