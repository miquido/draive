import json
from asyncio import gather
from collections.abc import Awaitable
from typing import Any, Literal, overload

from draive.mistral.models import ChatMessage, ToolCallResponse
from draive.tools import Toolbox

__all__ = [
    "_execute_chat_tool_calls",
]


async def _execute_chat_tool_calls(
    *,
    tool_calls: list[ToolCallResponse],
    tools: Toolbox,
) -> list[ChatMessage] | str:
    direct_result: Awaitable[str] | None = None
    tool_call_results: list[Awaitable[ChatMessage]] = []
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
                )
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


@overload
async def _execute_chat_tool_call(
    *,
    call_id: str,
    name: str,
    arguments: dict[str, Any] | str,
    tools: Toolbox,
    message_result: Literal[True],
) -> ChatMessage: ...


@overload
async def _execute_chat_tool_call(
    *,
    call_id: str,
    name: str,
    arguments: dict[str, Any] | str,
    tools: Toolbox,
    message_result: Literal[False],
) -> str: ...


async def _execute_chat_tool_call(
    *,
    call_id: str,
    name: str,
    arguments: dict[str, Any] | str,
    tools: Toolbox,
    message_result: bool,
) -> ChatMessage | str:
    try:  # make sure that tool error won't blow up whole chain
        result: str = str(
            await tools.call_tool(
                name,
                call_id=call_id,
                arguments=arguments,
            )
        )
        if message_result:
            return ChatMessage(
                role="tool",
                name=name,
                content=str(result),
            )
        else:
            return result

    # error should be already logged by ScopeContext
    except BaseException as exc:
        if message_result:
            return ChatMessage(
                role="tool",
                name=name,
                content="Error",
            )

        else:  # TODO: think about allowing the error chat message
            raise exc
