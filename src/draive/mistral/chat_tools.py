from asyncio import gather
from collections.abc import AsyncIterable, Awaitable

from mistralai.models.chat_completion import ChatCompletionStreamResponse, ChatMessage, ToolCall

from draive.tools import Toolbox

__all__ = [
    "_execute_chat_tool_calls",
    "_flush_chat_tool_calls",
]


async def _execute_chat_tool_calls(
    *,
    tool_calls: list[ToolCall],
    tools: Toolbox,
) -> list[ChatMessage]:
    tool_call_results: list[Awaitable[ChatMessage]] = []
    for call in tool_calls:
        tool_call_results.append(
            _execute_chat_tool_call(
                call_id=call.id,
                name=call.function.name,
                arguments=call.function.arguments,
                tools=tools,
            )
        )

    return [
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=tool_calls,
        ),
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
    tools: Toolbox,
) -> ChatMessage:
    try:  # make sure that tool error won't blow up whole chain
        result = await tools.call_tool(
            name,
            call_id=call_id,
            arguments=arguments,
        )
        return ChatMessage(
            role="tool",
            name=name,
            content=str(result),
        )

    # error should be already logged by ScopeContext
    except BaseException:
        return ChatMessage(
            role="tool",
            name=name,
            content="Error",
        )


async def _flush_chat_tool_calls(  # noqa: PLR0912
    *,
    tool_calls: list[ToolCall],
    completion_stream: AsyncIterable[ChatCompletionStreamResponse],
) -> list[ToolCall]:
    # iterate over the stream to get full list of tool calls
    async for chunk in completion_stream:
        for call in chunk.choices[0].delta.tool_calls or []:
            try:
                tool_call: ToolCall = next(
                    tool_call for tool_call in tool_calls if tool_call.id == call.id
                )

                if call.id:
                    if tool_call.id != "null":
                        tool_call.id += call.id
                    else:
                        tool_call.id = call.id
                else:
                    pass

                if call.function.name:
                    if tool_call.function.name:
                        tool_call.function.name += call.function.name
                    else:
                        tool_call.function.name = call.function.name
                else:
                    pass

                if call.function.arguments:
                    if tool_call.function.arguments:
                        tool_call.function.arguments += call.function.arguments
                    else:
                        tool_call.function.arguments = call.function.arguments
                else:
                    pass

            except (StopIteration, StopAsyncIteration):
                tool_calls.append(call)

    return tool_calls
