from typing import cast

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from draive.metrics import ArgumentsTrace, ResultTrace, TokenUsage
from draive.openai.chat_tools import (
    _execute_chat_tool_calls,  # pyright: ignore[reportPrivateUsage]
)
from draive.openai.client import OpenAIClient
from draive.openai.config import OpenAIChatConfig
from draive.openai.errors import OpenAIException
from draive.scope import ctx
from draive.tools import Toolbox

__all__ = [
    "_chat_response",
]


async def _chat_response(
    *,
    client: OpenAIClient,
    config: OpenAIChatConfig,
    messages: list[ChatCompletionMessageParam],
    tools: Toolbox,
    recursion_level: int = 0,
) -> str:
    with ctx.nested(
        "chat_response",
        metrics=[ArgumentsTrace.of(messages=messages.copy())],
    ):
        completion: ChatCompletion
        if recursion_level == config.recursion_limit:
            ctx.log_warning("Reaching limit of recursive OpenAI calls, ignoring tools...")
            completion = await client.chat_completion(
                config=config,
                messages=messages,
            )

        elif recursion_level != 0:  # suggest/require tool call only initially
            completion = await client.chat_completion(
                config=config,
                messages=messages,
                tools=cast(
                    list[ChatCompletionToolParam],
                    tools.available_tools,
                ),
            )

        elif suggested_tool_name := tools.suggested_tool_name:
            completion = await client.chat_completion(
                config=config,
                messages=messages,
                tools=cast(
                    list[ChatCompletionToolParam],
                    tools.available_tools,
                ),
                tools_suggestion={
                    "type": "function",
                    "function": {
                        "name": suggested_tool_name,
                    },
                },
            )

        else:
            completion = await client.chat_completion(
                config=config,
                messages=messages,
                tools=cast(
                    list[ChatCompletionToolParam],
                    tools.available_tools,
                ),
                tools_suggestion=tools.suggest_tools,
            )

        if usage := completion.usage:
            ctx.record(
                TokenUsage.for_model(
                    config.model,
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                ),
            )

        if not completion.choices:
            raise OpenAIException("Invalid OpenAI completion - missing messages!", completion)

        completion_message: ChatCompletionMessage = completion.choices[0].message

        if (tool_calls := completion_message.tool_calls) and (tools := tools):
            ctx.record(ResultTrace.of(tool_calls))

            tools_result: list[ChatCompletionMessageParam] | str = await _execute_chat_tool_calls(
                tool_calls=tool_calls,
                tools=tools,
            )

            if isinstance(tools_result, str):
                return tools_result
            else:
                messages.extend(tools_result)

        elif message := completion_message.content:
            ctx.record(ResultTrace.of(message))
            return message

        else:
            raise OpenAIException("Invalid OpenAI completion", completion)

    # recursion outside of context
    if recursion_level >= config.recursion_limit:
        raise OpenAIException("Reached limit of recursive calls of %d", config.recursion_limit)

    return await _chat_response(
        client=client,
        config=config,
        messages=messages,
        tools=tools,
        recursion_level=recursion_level + 1,
    )
