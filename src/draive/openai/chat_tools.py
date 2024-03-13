from asyncio import gather
from collections.abc import Awaitable
from typing import cast

from openai import AsyncStream
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallParam,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaToolCall,
)

from draive.scope import ctx
from draive.tools import ToolsProgressContext
from draive.types import ProgressUpdate, ToolCallProgress, Toolset

__all__ = [
    "_execute_chat_tool_calls",
    "_flush_chat_tool_calls",
]


async def _execute_chat_tool_calls(
    *,
    tool_calls: list[ChatCompletionMessageToolCall],
    toolset: Toolset,
    progress: ProgressUpdate[ToolCallProgress] | None = None,
) -> list[ChatCompletionMessageParam]:
    tool_call_params: list[ChatCompletionMessageToolCallParam] = []
    tool_call_results: list[Awaitable[ChatCompletionMessageParam]] = []
    with ctx.updated(
        ToolsProgressContext(
            progress=progress or ctx.state(ToolsProgressContext).progress,
        ),
    ):
        for call in tool_calls:
            tool_call_results.append(
                _execute_chat_tool_call(
                    call_id=call.id,
                    name=call.function.name,
                    arguments=call.function.arguments,
                    toolset=toolset,
                )
            )
            tool_call_params.append(
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
            )

    return [
        {
            "role": "assistant",
            "tool_calls": tool_call_params,
        },
        *await gather(
            *tool_call_results,
            return_exceptions=False,
        ),
    ]


async def _execute_chat_tool_call(
    *,
    call_id: str,
    name: str,
    arguments: str,
    toolset: Toolset,
) -> ChatCompletionMessageParam:
    try:  # make sure that tool error won't blow up whole chain
        result = await toolset.call_tool(
            name,
            call_id=call_id,
            arguments=arguments,
        )
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": str(result),
        }

    # error should be already logged by ScopeContext
    except BaseException:
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": "Error",
        }


async def _flush_chat_tool_calls(  # noqa: C901, PLR0912
    *,
    tool_calls: list[ChoiceDeltaToolCall],
    completion_stream: AsyncStream[ChatCompletionChunk],
) -> list[ChatCompletionMessageToolCall]:
    # iterate over the stream to get full list of tool calls
    async for chunk in completion_stream:
        for call in chunk.choices[0].delta.tool_calls or []:
            try:
                tool_call: ChoiceDeltaToolCall = next(
                    tool_call for tool_call in tool_calls if tool_call.index == call.index
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
                tool_calls.append(call)

    # completed calls have exactly the same model
    return cast(list[ChatCompletionMessageToolCall], tool_calls)
