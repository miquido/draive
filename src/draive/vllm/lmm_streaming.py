import json
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from itertools import chain
from typing import Any, cast
from uuid import uuid4

from haiway import MISSING, as_list, ctx
from openai import NOT_GIVEN
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDeltaToolCall
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from draive.instructions import Instruction
from draive.lmm import (
    LMMContext,
    LMMContextElement,
    LMMStream,
    LMMStreamChunk,
    LMMStreamInput,
    LMMStreamOutput,
    LMMStreamProperties,
    LMMToolRequest,
)
from draive.metrics import TokenUsage
from draive.multimodal import MultimodalContent
from draive.vllm.api import VLLMAPI
from draive.vllm.config import VLLMChatConfig
from draive.vllm.lmm import (
    content_element_as_content_part,
    context_element_as_messages,
    tools_as_tool_config,
)
from draive.vllm.types import VLLMException
from draive.vllm.utils import unwrap_missing

__all__ = [
    "VLLMLMMStreaming",
]


class VLLMLMMStreaming(VLLMAPI):
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
        config: VLLMChatConfig | None = None,
        **extra: Any,
    ) -> AsyncGenerator[LMMStreamOutput]:
        chat_config: VLLMChatConfig = config or ctx.state(VLLMChatConfig).updated(**extra)
        ctx.record(chat_config)

        context_elements: Sequence[LMMContextElement]
        match context:
            case None:
                context_elements = ()

            case [*elements]:
                context_elements = elements

        messages_context: list[ChatCompletionMessageParam] = list(
            chain.from_iterable(
                [
                    context_element_as_messages(element, config=chat_config)
                    for element in context_elements
                ]
            )
        )

        # track requested tool calls out of the loop
        pending_tool_calls: set[str] = set()
        # before each call check for updated properties - this supposed to be an infinite loop
        async for current_properties in properties:
            # for each call accumulate input first
            input_buffer: MultimodalContent = MultimodalContent.empty
            async for chunk in input:
                match chunk:
                    # gether input content chunks until marked as end
                    case LMMStreamChunk() as content_chunk:
                        input_buffer = input_buffer.extending(content_chunk.content)
                        if content_chunk.eod:
                            messages_context.append(
                                {
                                    "role": "user",
                                    "content": [
                                        content_element_as_content_part(
                                            element,
                                            config=chat_config,
                                        )
                                        for element in input_buffer.parts
                                    ],
                                }
                            )
                            break  # we are supporting only completed input messages with this api

                    # accumulate tool results directly in context
                    case tool_result:
                        pending_tool_calls.remove(tool_result.identifier)
                        messages_context.append(
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
            request_messages: list[ChatCompletionMessageParam]
            if instruction := current_properties.instruction:
                request_messages = [
                    {
                        "role": "system",
                        "content": Instruction.formatted(instruction),
                    },
                    *messages_context,
                ]

            else:
                request_messages = messages_context

            tool_choice, tools_list = tools_as_tool_config(
                current_properties.tools,
                tool_selection=current_properties.tool_selection,
            )

            accumulated_result: MultimodalContent = MultimodalContent.empty
            accumulated_tool_calls: list[ChoiceDeltaToolCall] = []
            async for part in await self._client.chat.completions.create(
                messages=request_messages,
                model=chat_config.model,
                frequency_penalty=unwrap_missing(chat_config.frequency_penalty),
                max_tokens=unwrap_missing(chat_config.max_tokens),
                n=1,
                seed=unwrap_missing(chat_config.seed),
                temperature=chat_config.temperature,
                tools=tools_list,
                tool_choice=tool_choice,
                parallel_tool_calls=unwrap_missing(chat_config.parallel_tool_calls)
                if tools_list
                else NOT_GIVEN,
                top_p=unwrap_missing(chat_config.top_p),
                timeout=unwrap_missing(chat_config.timeout),
                stop=as_list(cast(Sequence[str], chat_config.stop_sequences))
                if chat_config.stop_sequences is not MISSING
                else NOT_GIVEN,
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
                        accumulated_result = accumulated_result.extending(content_chunk.content)
                        yield content_chunk

                    if finish_reason := element.finish_reason:
                        if finish_reason in ("length", "content_filter"):
                            raise VLLMException(f"Unexpected finish reason: {finish_reason}")

                        if accumulated_tool_calls:
                            messages_context.append(
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
                                        if request.id and request.function and request.function.name
                                    ],
                                }
                            )
                            # send tool calls
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

                        else:
                            messages_context.append(
                                {
                                    "role": "assistant",
                                    "content": accumulated_result.as_string(),
                                }
                            )
                            # send completion chunk
                            yield LMMStreamChunk.of(
                                MultimodalContent.empty,
                                eod=True,
                            )

                if usage := part.usage:  # record usage if able (expected in the last part)
                    ctx.record(
                        TokenUsage.for_model(
                            part.model,
                            input_tokens=usage.prompt_tokens,
                            cached_tokens=None,
                            output_tokens=usage.completion_tokens,
                        ),
                    )
