from typing import cast

from openai import AsyncStream as OpenAIAsyncStream
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from draive.openai.chat_tools import (
    _execute_chat_tool_calls,  # pyright: ignore[reportPrivateUsage]
    _flush_chat_tool_calls,  # pyright: ignore[reportPrivateUsage]
)
from draive.openai.client import OpenAIClient
from draive.openai.config import OpenAIChatConfig
from draive.openai.errors import OpenAIException
from draive.scope import ArgumentsTrace, ResultTrace, ctx
from draive.tools import Toolbox, ToolCallUpdate
from draive.types import UpdateSend

__all__ = [
    "_chat_stream",
]


async def _chat_stream(  # noqa: PLR0913
    *,
    client: OpenAIClient,
    config: OpenAIChatConfig,
    messages: list[ChatCompletionMessageParam],
    tools: Toolbox | None,
    send_update: UpdateSend[ToolCallUpdate | str],
    recursion_level: int = 0,
) -> str:
    if recursion_level > config.recursion_limit:
        raise OpenAIException("Reached limit of recursive calls of %d", config.recursion_limit)

    with ctx.nested(
        "chat_stream",
        metrics=[ArgumentsTrace(messages=messages.copy())],
    ):
        completion_stream: OpenAIAsyncStream[ChatCompletionChunk] = await client.chat_completion(
            config=config,
            messages=messages,
            tools=cast(
                list[ChatCompletionToolParam],
                tools.available_tools if tools else [],
            ),
            suggested_tool={
                "type": "function",
                "function": {
                    "name": tools.suggested_tool_name,
                },
            }  # suggest/require tool call only initially
            if recursion_level == 0 and tools is not None and tools.suggested_tool_name
            else None,
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

            # TODO: record token usage - openAI does not provide usage insight when streaming
            # (or makes it differently than when using regular response and couldn't find it)

            if completion_head.tool_calls is not None and (tools := tools):
                tool_calls: list[ChatCompletionMessageToolCall] = await _flush_chat_tool_calls(
                    tool_calls=completion_head.tool_calls,
                    completion_stream=completion_stream,
                )
                messages.extend(
                    await _execute_chat_tool_calls(
                        tool_calls=tool_calls,
                        tools=tools,
                    )
                )
                ctx.record(ResultTrace(tool_calls))
                break  # after processing tool calls continue with recursion in outer context

            elif completion_head.content is not None:
                result: str = completion_head.content
                if result:  # provide head / first part if not empty
                    send_update(result)

                async for part in completion_stream:
                    # we are always requesting single result - no need to take care of indices
                    part_text: str = part.choices[0].delta.content or ""
                    if not part_text:
                        continue  # skip empty parts
                    result += part_text
                    send_update(result)

                ctx.record(ResultTrace(result))
                return result  # we hav final result here

            else:
                continue  # iterate over the stream until can decide what to do or reach the end

    # recursion outside of context
    return await _chat_stream(
        client=client,
        config=config,
        messages=messages,
        tools=tools,
        send_update=send_update,
        recursion_level=recursion_level + 1,
    )
