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
from draive.scope import (
    ArgumentsTrace,
    ResultTrace,
    TokenUsage,
    ctx,
)
from draive.tools import Toolbox, ToolException

__all__ = [
    "_chat_response",
]


async def _chat_response(
    *,
    client: OpenAIClient,
    config: OpenAIChatConfig,
    messages: list[ChatCompletionMessageParam],
    tools: Toolbox | None,
) -> str:
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
        )

        if usage := completion.usage:
            ctx.record(
                TokenUsage(
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                ),
            )

        if not completion.choices:
            raise ToolException("Invalid OpenAI completion - missing messages!", completion)

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
            raise ToolException("Invalid OpenAI completion", completion)

    # recursion outside of context
    return await _chat_response(
        client=client,
        config=config,
        messages=messages,
        tools=tools,
    )
