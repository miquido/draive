from collections.abc import Callable
from typing import cast

from openai import AsyncStream as OpenAIAsyncStream
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from draive.metrics import ArgumentsTrace, ResultTrace, TokenUsage
from draive.openai.chat_tools import (
    _execute_chat_tool_calls,  # pyright: ignore[reportPrivateUsage]
    _flush_chat_tool_calls,  # pyright: ignore[reportPrivateUsage]
)
from draive.openai.client import OpenAIClient
from draive.openai.config import OpenAIChatConfig
from draive.openai.errors import OpenAIException
from draive.scope import ctx
from draive.tools import Toolbox, ToolCallUpdate

__all__ = [
    "_chat_stream",
]


async def _chat_stream(  # noqa: PLR0913, C901, PLR0915, PLR0912
    *,
    client: OpenAIClient,
    config: OpenAIChatConfig,
    messages: list[ChatCompletionMessageParam],
    tools: Toolbox,
    send_update: Callable[[ToolCallUpdate | str], None],
    recursion_level: int = 0,
) -> str:
    with ctx.nested(
        "chat_stream",
        metrics=[ArgumentsTrace.of(messages=messages.copy())],
    ):
        completion_stream: OpenAIAsyncStream[ChatCompletionChunk]
        if recursion_level == config.recursion_limit:
            ctx.log_warning("Reaching limit of recursive OpenAI calls, ignoring tools...")
            completion_stream = await client.chat_completion(
                config=config,
                messages=messages,
                tools=None,
                stream=True,
            )

        else:
            tools_suggestion: ChatCompletionNamedToolChoiceParam | bool
            if recursion_level != 0:  # suggest/require tool call only initially
                tools_suggestion = False

            elif suggested_tool_name := tools.suggested_tool_name:
                tools_suggestion = {
                    "type": "function",
                    "function": {
                        "name": suggested_tool_name,
                    },
                }

            else:
                tools_suggestion = tools.suggest_tools

            completion_stream = await client.chat_completion(
                config=config,
                messages=messages,
                tools=cast(
                    list[ChatCompletionToolParam],
                    tools.available_tools if tools else [],
                ),
                tools_suggestion=tools_suggestion,
                stream=True,
            )

        while True:  # load chunks to decide what to do next
            head: ChatCompletionChunk
            try:
                head = await anext(completion_stream)

            except StopAsyncIteration as exc:
                # could not decide what to do before stream end
                raise OpenAIException("Invalid OpenAI completion stream") from exc

            if not head.choices:
                raise OpenAIException("Invalid OpenAI completion - missing deltas!", head)

            completion_head: ChoiceDelta = head.choices[0].delta

            if completion_head.tool_calls is not None and (tools := tools):
                tool_calls: list[ChatCompletionMessageToolCall] = await _flush_chat_tool_calls(
                    model=config.model,  # model for token usage tracking
                    tool_calls=completion_head.tool_calls,
                    completion_stream=completion_stream,
                )
                ctx.record(ResultTrace.of(tool_calls))

                tools_result: (
                    list[ChatCompletionMessageParam] | str
                ) = await _execute_chat_tool_calls(
                    tool_calls=tool_calls,
                    tools=tools,
                )

                if isinstance(tools_result, str):
                    send_update(tools_result)
                    return tools_result
                else:
                    messages.extend(tools_result)

                break  # after processing tool calls continue with recursion in outer context

            elif completion_head.content is not None:
                result: str = completion_head.content
                if result:  # provide head / first part if not empty
                    send_update(result)

                async for part in completion_stream:
                    if part.choices:  # usage part does not contain choices
                        # we are always requesting single result - no need to take care of indices
                        part_text: str = part.choices[0].delta.content or ""
                        if not part_text:
                            continue  # skip empty parts
                        result += part_text
                        send_update(result)

                    elif usage := part.usage:  # record usage if able (expected in last part)
                        ctx.record(
                            TokenUsage.for_model(
                                config.model,
                                input_tokens=usage.prompt_tokens,
                                output_tokens=usage.completion_tokens,
                            ),
                        )

                    else:
                        ctx.log_warning("Unexpected OpenAI streaming part: %s", part)
                        continue

                ctx.record(ResultTrace.of(result))
                return result  # we hav final result here

            else:
                continue  # iterate over the stream until can decide what to do or reach the end

    # recursion outside of context
    if recursion_level >= config.recursion_limit:
        raise OpenAIException("Reached limit of recursive calls of %d", config.recursion_limit)

    return await _chat_stream(
        client=client,
        config=config,
        messages=messages,
        tools=tools,
        send_update=send_update,
        recursion_level=recursion_level + 1,
    )
