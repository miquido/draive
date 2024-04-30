from asyncio import gather
from collections.abc import Awaitable
from typing import Literal, cast, overload

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

from draive.tools import Toolbox

__all__ = [
    "_execute_chat_tool_calls",
    "_flush_chat_tool_calls",
]


async def _execute_chat_tool_calls(
    *,
    tool_calls: list[ChatCompletionMessageToolCall],
    tools: Toolbox,
) -> list[ChatCompletionMessageParam] | str:
    direct_result: Awaitable[str] | None = None
    tool_call_params: list[ChatCompletionMessageToolCallParam] = []
    tool_call_results: list[Awaitable[ChatCompletionMessageParam]] = []
    for call in tool_calls:
        # use only the first "direct result tool" requested, can't return more than one anyways
        # despite of that all tools will be called to ensure that all desired actions were executed
        if direct_result is None and tools.requires_direct_result(tool_name=call.function.name):
            direct_result = _execute_chat_tool_call(
                call_id=call.id,
                name=call.function.name,
                arguments=call.function.arguments,
                tools=tools,
                message_result=False,
            )
        else:
            tool_call_results.append(
                _execute_chat_tool_call(
                    call_id=call.id,
                    name=call.function.name,
                    arguments=call.function.arguments,
                    tools=tools,
                    message_result=True,
                ),
            )
            tool_call_params.append(
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                },
            )

    if direct_result is not None:
        results: tuple[str, ...] = await gather(
            direct_result,
            *tool_call_results,
            return_exceptions=False,
        )
        return results[0]  # return only the requested direct result
    else:
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


@overload
async def _execute_chat_tool_call(
    *,
    call_id: str,
    name: str,
    arguments: str,
    tools: Toolbox,
    message_result: Literal[True],
) -> ChatCompletionMessageParam: ...


@overload
async def _execute_chat_tool_call(
    *,
    call_id: str,
    name: str,
    arguments: str,
    tools: Toolbox,
    message_result: Literal[False],
) -> str: ...


async def _execute_chat_tool_call(
    *,
    call_id: str,
    name: str,
    arguments: str,
    tools: Toolbox,
    message_result: bool,
) -> ChatCompletionMessageParam | str:
    try:  # make sure that tool error won't blow up whole chain
        result: str = str(
            await tools.call_tool(
                name,
                call_id=call_id,
                arguments=arguments,
            )
        )
        if message_result:
            return {
                "role": "tool",
                "tool_call_id": call_id,
                "content": str(result),
            }
        else:
            return result

    # error should be already logged by ScopeContext
    except Exception as exc:
        if message_result:
            return {
                "role": "tool",
                "tool_call_id": call_id,
                "content": "Error",
            }

        else:  # TODO: think about allowing the error chat message
            raise exc


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
