import json
from base64 import b64encode
from collections.abc import AsyncGenerator, AsyncIterator, Iterable
from typing import Any, Literal, cast
from uuid import uuid4

from haiway import ArgumentsTrace, ResultTrace, ctx, not_missing
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionContentPartParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDeltaToolCall

from draive.instructions import Instruction
from draive.lmm import (
    LMMCompletion,
    LMMContextElement,
    LMMInput,
    LMMInvocation,
    LMMOutput,
    LMMStream,
    LMMStreamChunk,
    LMMStreamInput,
    LMMStreamOutput,
    LMMStreamProperties,
    LMMToolRequest,
    LMMToolRequests,
    LMMToolResponse,
    LMMToolSelection,
    ToolSpecification,
)
from draive.metrics import TokenUsage
from draive.multimodal import (
    MediaContent,
    MultimodalContent,
    MultimodalContentElement,
    TextContent,
)
from draive.openai.client import OpenAIClient
from draive.openai.config import OpenAIChatConfig, OpenAISystemFingerprint
from draive.openai.types import OpenAIException
from draive.parameters import DataModel, ParametersSpecification

__all__ = [
    "openai_lmm",
    "openai_streaming_lmm",
]


def openai_lmm(
    client: OpenAIClient | None = None,
    /,
) -> LMMInvocation:
    client = client or OpenAIClient.shared()

    async def lmm_invocation(
        *,
        instruction: Instruction | str | None,
        context: Iterable[LMMContextElement],
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

            return await _chat_completion(
                client=client,
                config=config,
                messages=messages,
                tools=tools,
                tool_selection=tool_selection,
            )

    return LMMInvocation(invoke=lmm_invocation)


def openai_streaming_lmm(
    client: OpenAIClient | None = None,
    /,
) -> LMMStream:
    client = client or OpenAIClient.shared()

    async def lmm_stream(
        *,
        properties: AsyncIterator[LMMStreamProperties],
        input: AsyncIterator[LMMStreamInput],  # noqa: A002
        context: Iterable[LMMContextElement] | None,
        **extra: Any,
    ) -> AsyncIterator[LMMStreamOutput]:
        config: OpenAIChatConfig = ctx.state(OpenAIChatConfig).updated(**extra)
        ctx.record(config)

        return _chat_stream(
            client=client,
            config=config,
            properties=properties,
            input=input,
            context=[
                _convert_context_element(config=config, element=element) for element in context
            ]
            if context
            else [],
        )

    return LMMStream(prepare=lmm_stream)


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

        case MediaContent() as media:
            if media.kind != "image":
                raise ValueError("Unsupported message content", media)

            url: str
            match media.source:
                case str() as string:
                    url = string

                case bytes() as data:
                    url = f"data:{media.media};base64,{b64encode(data).decode()}"

            return {
                "type": "image_url",
                "image_url": {
                    "url": url,
                    "detail": cast(Literal["auto", "low", "high"], config.vision_details)
                    if not_missing(config.vision_details)
                    else "auto",
                },
            }

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


async def _chat_stream(  # noqa: C901, PLR0912, PLR0915
    *,
    client: OpenAIClient,
    config: OpenAIChatConfig,
    properties: AsyncIterator[LMMStreamProperties],
    input: AsyncIterator[LMMStreamInput],  # noqa: A002
    context: list[ChatCompletionMessageParam],
) -> AsyncGenerator[LMMStreamOutput, None]:
    # track requested tool calls out of the loop
    pending_tool_calls: set[str] = set()
    # before each call check for updated properties - this supposed to be an infinite loop
    async for current_properties in properties:
        # for each call accumulate input first
        input_buffer: MultimodalContent = MultimodalContent.of()
        async for chunk in input:
            match chunk:
                # gether input content chunks until marked as end
                case LMMStreamChunk() as content_chunk:
                    input_buffer = input_buffer.appending(content_chunk.content)
                    if content_chunk.eod:
                        context.append(
                            {
                                "role": "user",
                                "content": [
                                    _convert_content_element(
                                        element=element,
                                        config=config,
                                    )
                                    for element in input_buffer.parts
                                ],
                            }
                        )
                        break  # we are supporting only completed input messages with this api

                # accumulate tool results directly in context
                case tool_result:
                    pending_tool_calls.remove(tool_result.identifier)
                    context.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_result.identifier,
                            "content": tool_result.content.as_string(),
                        }
                    )
                    # when there is no pending input and we got all requested tool results
                    if not input_buffer and not pending_tool_calls:
                        break  # then we can request completion again

        else:  # finalize streaming if input is finished
            return

        # prepare context for the next request by extending with instructions if any
        request_context: list[ChatCompletionMessageParam]
        if instruction := current_properties.instruction:
            request_context = [
                {
                    "role": "system",
                    "content": Instruction.of(instruction).format(),
                },
                *context,
            ]

        else:
            request_context = context

        accumulated_result: MultimodalContent = MultimodalContent.of()
        accumulated_tool_calls: list[ChoiceDeltaToolCall] = []
        async for part in await client.chat_completion(
            config=config,
            messages=request_context,
            tools=cast(
                list[ChatCompletionToolParam],
                current_properties.tools or [],
            ),
            tool_choice="auto" if current_properties.tools else "none",
            stream=True,
        ):
            if part.choices:  # usage part does not contain choices
                # we are always requesting single result - no need to take care of indices
                element: Choice = part.choices[0]
                # get the tool calls parts first
                if tool_calls := element.delta.tool_calls:
                    # tool calls come in parts, we have to merge them manually
                    for call in tool_calls:
                        try:
                            tool_call: ChoiceDeltaToolCall = next(
                                tool_call
                                for tool_call in accumulated_tool_calls
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
                            accumulated_tool_calls.append(call)

                # then process content
                if element.delta.content is not None:
                    content_chunk: LMMStreamChunk = LMMStreamChunk.of(element.delta.content)
                    accumulated_result = accumulated_result.appending(
                        content_chunk.content,
                        merge_text=True,
                    )
                    yield content_chunk

                if finish_reason := element.finish_reason:
                    match finish_reason:
                        case "stop":
                            # send completion chunk - openAI sends it without an actual content
                            yield LMMStreamChunk.of(
                                MultimodalContent.of(),
                                eod=True,
                            )

                        case "tool_calls":
                            if accumulated_tool_calls:
                                context.append(
                                    {
                                        "role": "assistant",
                                        "tool_calls": [
                                            {
                                                "id": request.id,
                                                "type": "function",
                                                "function": {
                                                    "name": request.function.name,
                                                    "arguments": request.function.arguments or "{}",
                                                },
                                            }
                                            for request in accumulated_tool_calls
                                            if request.id
                                            and request.function
                                            and request.function.name
                                        ],
                                    }
                                )
                            if accumulated_result:
                                context.append(
                                    {
                                        "role": "assistant",
                                        "content": accumulated_result.as_string(),
                                    }
                                )

                            for call in accumulated_tool_calls:
                                if not call.function:
                                    continue  # skip partial calls
                                if not call.function.name:
                                    continue  # skip calls with missing names

                                call_identifier: str = call.id or uuid4().hex
                                pending_tool_calls.add(call_identifier)
                                # send tool requests when ensured that all were completed
                                yield LMMToolRequest(
                                    identifier=call_identifier,
                                    tool=call.function.name,
                                    arguments=json.loads(call.function.arguments)
                                    if call.function.arguments
                                    else {},
                                )

                        case other:
                            raise OpenAIException(f"Unexpected finish reason: {other}")

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
