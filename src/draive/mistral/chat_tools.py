import json
from asyncio import gather
from collections.abc import Awaitable
from typing import Any

from draive.mistral.models import ChatMessage, ToolCallResponse
from draive.tools import Toolbox

__all__ = [
    "_execute_chat_tool_calls",
]


async def _execute_chat_tool_calls(
    *,
    tool_calls: list[ToolCallResponse],
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
            tool_calls=[
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments
                        if isinstance(call.function.arguments, str)
                        else json.dumps(call.function.arguments),
                    },
                }
                for call in tool_calls
            ],
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
    arguments: dict[str, Any] | str,
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
