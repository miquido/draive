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
from draive.scope import ArgumentsTrace, ResultTrace, ctx
from draive.types import (
    ConversationStreamingPartialMessage,
    ConversationStreamingUpdate,
    ProgressUpdate,
    ToolException,
    Toolset,
)

__all__ = [
    "_chat_stream",
]


async def _chat_stream(
    *,
    client: OpenAIClient,
    config: OpenAIChatConfig,
    messages: list[ChatCompletionMessageParam],
    toolset: Toolset | None,
    progress: ProgressUpdate[ConversationStreamingUpdate],
) -> str:
    with ctx.nested(
        "chat_stream",
        ArgumentsTrace(messages=messages.copy()),
    ):
        completion_stream: OpenAIAsyncStream[ChatCompletionChunk] = await client.chat_completion(
            config=config,
            messages=messages,
            tools=cast(
                list[ChatCompletionToolParam],
                toolset.available_tools if toolset else [],
            ),
            stream=True,
        )

        while True:  # load chunks to decide what to do next
            head: ChatCompletionChunk
            try:
                head = await anext(completion_stream)

            except StopAsyncIteration as exc:
                # could not decide what to do before stream end
                raise ToolException("Invalid OpenAI completion stream") from exc

            if not head.choices:
                raise ToolException("Invalid OpenAI completion - missing deltas!", head)

            completion_head: ChoiceDelta = head.choices[0].delta

            # TODO: record token usage - openAI does not provide usage insight when streaming
            # (or makes it differently than when using regular response and couldn't find it)

            if completion_head.tool_calls is not None and (toolset := toolset):
                tool_calls: list[ChatCompletionMessageToolCall] = await _flush_chat_tool_calls(
                    tool_calls=completion_head.tool_calls,
                    completion_stream=completion_stream,
                )
                messages.extend(
                    await _execute_chat_tool_calls(
                        tool_calls=tool_calls,
                        toolset=toolset,
                        progress=progress,
                    )
                )
                ctx.record(ResultTrace(tool_calls))
                break  # after processing tool calls continue with recursion in outer context

            elif completion_head.content is not None:
                result: str = completion_head.content
                if result:  # provide head / first part if not empty
                    progress(update=ConversationStreamingPartialMessage(content=result))

                async for part in completion_stream:
                    # we are always requesting single result - no need to take care of indices
                    part_text: str = part.choices[0].delta.content or ""
                    if not part_text:
                        continue  # skip empty parts
                    result += part_text
                    progress(update=ConversationStreamingPartialMessage(content=result))

                ctx.record(ResultTrace(result))
                return result  # we hav final result here

            else:
                continue  # iterate over the stream until can decide what to do or reach the end

    # recursion outside of context
    return await _chat_stream(
        client=client,
        config=config,
        messages=messages,
        toolset=toolset,
        progress=progress,
    )
