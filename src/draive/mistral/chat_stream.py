from collections.abc import AsyncIterable, AsyncIterator
from typing import cast

from mistralai.models.chat_completion import (
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    ToolCall,
)

from draive.mistral.chat_response import _chat_response  # pyright: ignore[reportPrivateUsage]
from draive.mistral.chat_tools import (
    _execute_chat_tool_calls,  # pyright: ignore[reportPrivateUsage]
    _flush_chat_tool_calls,  # pyright: ignore[reportPrivateUsage]
)
from draive.mistral.client import MistralClient
from draive.mistral.config import MistralChatConfig
from draive.scope import ArgumentsTrace, ResultTrace, ctx
from draive.tools import Toolbox, ToolCallUpdate, ToolException
from draive.types import UpdateSend

__all__ = [
    "_chat_stream",
]


async def _chat_stream(
    *,
    client: MistralClient,
    config: MistralChatConfig,
    messages: list[ChatMessage],
    tools: Toolbox | None,
    send_update: UpdateSend[ToolCallUpdate | str],
) -> str:
    if tools is not None:
        ctx.log_warning(
            "Mistral streaming api is broken - can't properly call tools, waiting for full response"
        )
        message = await _chat_response(
            client=client,
            config=config,
            messages=messages,
            tools=tools,
        )
        send_update(message)
        return message

    with ctx.nested(
        "chat_stream",
        metrics=[ArgumentsTrace(messages=messages.copy())],
    ):
        completion_stream: AsyncIterable[
            ChatCompletionStreamResponse
        ] = await client.chat_completion(
            config=config,
            messages=messages,
            tools=cast(
                list[dict[str, object]],
                tools.available_tools if tools else [],
            ),
            stream=True,
        )
        completion_stream_iterator: AsyncIterator[
            ChatCompletionStreamResponse
        ] = completion_stream.__aiter__()

        while True:  # load chunks to decide what to do next
            head: ChatCompletionStreamResponse
            try:
                head = await anext(completion_stream_iterator)
            except StopAsyncIteration as exc:
                # could not decide what to do before stream end
                raise ToolException("Invalid Mistral completion stream") from exc

            if not head.choices:
                raise ToolException("Invalid Mistral completion - missing deltas!", head)

            completion_head: DeltaMessage = head.choices[0].delta

            # TODO: record token usage

            if completion_head.tool_calls is not None and (tools := tools):
                tool_calls: list[ToolCall] = await _flush_chat_tool_calls(
                    tool_calls=completion_head.tool_calls,
                    completion_stream=completion_stream_iterator,
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
    )
