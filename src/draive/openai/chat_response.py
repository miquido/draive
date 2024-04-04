from typing import cast

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from draive.openai.chat_tools import (
    _execute_chat_tool_calls,  # pyright: ignore[reportPrivateUsage]
)
from draive.openai.client import OpenAIClient
from draive.openai.config import OpenAIChatConfig
from draive.openai.errors import OpenAIException
from draive.scope import (
    ArgumentsTrace,
    ResultTrace,
    TokenUsage,
    ctx,
)
from draive.tools import Toolbox

__all__ = [
    "_chat_response",
]


async def _chat_response(
    *,
    client: OpenAIClient,
    config: OpenAIChatConfig,
    messages: list[ChatCompletionMessageParam],
    tools: Toolbox | None,
    recursion_level: int = 0,
) -> str:
    if recursion_level > config.recursion_limit:
        raise OpenAIException("Reached limit of recursive calls of %d", config.recursion_limit)

    with ctx.nested(
        "chat_response",
        metrics=[ArgumentsTrace(messages=messages.copy())],
    ):
        completion: ChatCompletion = await client.chat_completion(
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
        )

        if usage := completion.usage:
            ctx.record(
                TokenUsage(
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                ),
            )

        if not completion.choices:
            raise OpenAIException("Invalid OpenAI completion - missing messages!", completion)

        completion_message: ChatCompletionMessage = completion.choices[0].message

        if (tool_calls := completion_message.tool_calls) and (tools := tools):
            messages.extend(
                await _execute_chat_tool_calls(
                    tool_calls=tool_calls,
                    tools=tools,
                )
            )
            ctx.record(ResultTrace(tool_calls))

        elif message := completion_message.content:
            ctx.record(ResultTrace(message))
            return message

        else:
            raise OpenAIException("Invalid OpenAI completion", completion)

    # recursion outside of context
    return await _chat_response(
        client=client,
        config=config,
        messages=messages,
        tools=tools,
        recursion_level=recursion_level + 1,
    )
