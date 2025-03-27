import json
from collections.abc import AsyncGenerator, AsyncIterator
from itertools import chain
from typing import Any, cast
from uuid import uuid4

from haiway import as_list, ctx
from mistralai.models import (
    CompletionResponseStreamChoice,
    DeltaMessage,
    MessagesTypedDict,
    ToolCall,
)

from draive.instructions import Instruction
from draive.lmm import (
    LMMContext,
    LMMStream,
    LMMToolRequest,
)
from draive.lmm.types import LMMStreamChunk, LMMStreamInput, LMMStreamOutput, LMMStreamProperties
from draive.metrics import TokenUsage
from draive.mistral.api import MistralAPI
from draive.mistral.config import MistralChatConfig
from draive.mistral.lmm import (
    content_chunk_as_content_element,
    content_element_as_content_chunk,
    context_element_as_messages,
    tools_as_tool_config,
)
from draive.mistral.types import MistralException
from draive.mistral.utils import unwrap_missing_to_none, unwrap_missing_to_unset
from draive.multimodal import MultimodalContent

__all__ = [
    "MistralLMMStreaming",
]


class MistralLMMStreaming(MistralAPI):
    def lmm_streaming(self) -> LMMStream:
        async def prepare_stream(
            *,
            properties: AsyncIterator[LMMStreamProperties],
            input: AsyncIterator[LMMStreamInput],  # noqa: A002
            context: LMMContext | None,
            **extra: Any,
        ) -> AsyncIterator[LMMStreamOutput]:
            return self.prepare_lmm_stream(
                properties=properties,
                input=input,
                context=context,
                **extra,
            )

        return LMMStream(prepare=prepare_stream)

    async def prepare_lmm_stream(  # noqa: C901, PLR0912, PLR0915
        self,
        *,
        properties: AsyncIterator[LMMStreamProperties],
        input: AsyncIterator[LMMStreamInput],  # noqa: A002
        context: LMMContext | None,
        config: MistralChatConfig | None = None,
        **extra: Any,
    ) -> AsyncGenerator[LMMStreamOutput]:
        # prepare config
        chat_config: MistralChatConfig = config or ctx.state(MistralChatConfig).updated(**extra)
        ctx.record(chat_config)
        # track current context messages
        messages: list[MessagesTypedDict] = list(
            chain.from_iterable([context_element_as_messages(element) for element in context or []])
        )
        # track requested tool calls out of the loop
        pending_tool_calls: set[str] = set()
        # before each call check for updated properties - this supposed to be an infinite loop
        async for current_properties in properties:
            # start from accumulating input first
            input_buffer: MultimodalContent = MultimodalContent.empty
            async for chunk in input:
                match chunk:
                    # gether input content chunks until marked as end
                    case LMMStreamChunk() as content_chunk:
                        input_buffer = input_buffer.extending(content_chunk.content)
                        if content_chunk.eod:
                            messages.append(
                                {
                                    "role": "user",
                                    "content": [
                                        content_element_as_content_chunk(element)
                                        for element in input_buffer.parts
                                    ],
                                }
                            )
                            break  # we are supporting only completed input messages

                    # accumulate tool results directly in context
                    case tool_result:
                        pending_tool_calls.remove(tool_result.identifier)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_result.identifier,
                                "content": [
                                    content_element_as_content_chunk(element)
                                    for element in tool_result.content.parts
                                ],
                            }
                        )
                        # when there is no pending input and we got all requested tool results
                        if not input_buffer and not pending_tool_calls:
                            break  # then we can request completion again

            else:  # finalize streaming if input is finished
                return

            # prepare context for the next request by extending with instructions if any
            request_context: list[MessagesTypedDict]
            if instruction := current_properties.instruction:
                request_context = [
                    {
                        "role": "system",
                        "content": Instruction.formatted(instruction),
                    },
                    *messages,
                ]

            else:
                request_context = messages

            tool_choice, tools_list = tools_as_tool_config(
                current_properties.tools,
                tool_selection=current_properties.tool_selection,
            )

            accumulated_result: MultimodalContent = MultimodalContent.empty
            accumulated_tool_calls: list[ToolCall] = []
            async with await self._client.chat.stream_async(
                model=chat_config.model,
                messages=request_context,
                temperature=chat_config.temperature,
                top_p=unwrap_missing_to_none(chat_config.top_p),
                max_tokens=unwrap_missing_to_unset(chat_config.max_tokens),
                stop=as_list(unwrap_missing_to_none(chat_config.stop_sequences)),
                random_seed=unwrap_missing_to_unset(chat_config.seed),
                response_format={"type": "text"},
                tools=tools_list,
                tool_choice=tool_choice,
                stream=True,
            ) as response_stream:
                async for completion_chunk in response_stream:
                    if usage := completion_chunk.data.usage:
                        ctx.record(
                            TokenUsage.for_model(
                                completion_chunk.data.model,
                                input_tokens=usage.prompt_tokens,
                                cached_tokens=None,
                                output_tokens=usage.completion_tokens,
                            ),
                        )
                        # send completion chunk - openAI sends it without an actual content
                        yield LMMStreamChunk.of(
                            MultimodalContent.empty,
                            eod=True,
                        )

                    if not completion_chunk.data.choices:
                        raise MistralException(
                            "Invalid Mistral completion - missing choices!",
                            completion_chunk.data,
                        )

                    completion_choice: CompletionResponseStreamChoice = (
                        completion_chunk.data.choices[0]
                    )

                    completion_delta: DeltaMessage = completion_choice.delta
                    if content := completion_delta.content:
                        match content:
                            case str() as string:
                                yield LMMStreamChunk.of(string)

                            case chunks:
                                yield LMMStreamChunk.of(
                                    MultimodalContent.of(
                                        *[
                                            content_chunk_as_content_element(chunk)
                                            for chunk in chunks
                                        ]
                                    )
                                )

                    if tool_calls := completion_delta.tool_calls:
                        assert tools_list, "Requesting tool call without tools"  # nosec: B101
                        if not accumulated_tool_calls:
                            accumulated_tool_calls = sorted(
                                tool_calls,
                                key=lambda call: call.index or 0,
                            )

                        else:
                            for tool_call in tool_calls:
                                assert tool_call.index, "Can't identify function call without index"  # nosec: B101

                                # "null" is a dafault value...
                                if tool_call.id and tool_call.id != "null":
                                    accumulated_tool_calls[tool_call.index].id = tool_call.id

                                if tool_call.function.name:
                                    accumulated_tool_calls[
                                        tool_call.index
                                    ].function.name += tool_call.function.name

                                if isinstance(tool_call.function.arguments, str):
                                    assert isinstance(  # nosec: B101
                                        accumulated_tool_calls[tool_call.index].function.arguments,
                                        str,
                                    )
                                    accumulated_tool_calls[  # pyright: ignore[reportOperatorIssue]
                                        tool_call.index
                                    ].function.arguments += tool_call.function.arguments

                                else:
                                    assert isinstance(  # nosec: B101
                                        accumulated_tool_calls[tool_call.index].function.arguments,
                                        dict,
                                    )
                                    accumulated_tool_calls[tool_call.index].function.arguments = {
                                        **cast(
                                            dict,
                                            accumulated_tool_calls[
                                                tool_call.index
                                            ].function.arguments,
                                        ),
                                        **tool_call.function.arguments,
                                    }

                    match completion_choice.finish_reason:
                        case None:
                            pass  # continue streaming

                        case "stop":
                            if accumulated_result:
                                messages.append(
                                    {
                                        "role": "assistant",
                                        "content": [
                                            content_element_as_content_chunk(element)
                                            for element in accumulated_result.parts
                                        ],
                                    }
                                )

                            # send completion chunk if needed
                            yield LMMStreamChunk.of(
                                MultimodalContent.empty,
                                eod=True,
                            )
                            break  # and break the loop

                        case "tool_calls":
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
                                    if isinstance(call.function.arguments, str)
                                    else call.function.arguments,
                                )

                            # include accumulated result if needed
                            if accumulated_result:
                                messages.append(
                                    {
                                        "role": "assistant",
                                        "content": [
                                            content_element_as_content_chunk(element)
                                            for element in accumulated_result.parts
                                        ],
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

                            else:
                                messages.append(
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

                            break  # and break the loop

                        case "length":
                            raise MistralException(
                                "Invalid Mistral completion - exceeded maximum length!",
                                completion_chunk.data,
                            )

                        case "error":
                            raise MistralException(
                                "Mistral completion generation failed!",
                                completion_chunk.data,
                            )
