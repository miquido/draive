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
from draive.tools import ToolException
from draive.types import Toolset

__all__ = [
    "_chat_completion",
]


async def _chat_completion(
    *,
    client: OpenAIClient,
    config: OpenAIChatConfig,
    messages: list[ChatCompletionMessageParam],
    toolset: Toolset | None,
) -> str:
    async with ctx.nested(
        "chat_response",
        ArgumentsTrace(messages=messages),
    ):
        completion: ChatCompletion = await client.chat_completion(
            config=config,
            messages=messages,
            tools=cast(
                list[ChatCompletionToolParam],
                toolset.available_tools if toolset else [],
            ),
        )

        if usage := completion.usage:
            await ctx.record(
                TokenUsage(
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                ),
            )

        if not completion.choices:
            raise ToolException("Invalid OpenAI completion - missing messages!", completion)

        completion_message: ChatCompletionMessage = completion.choices[0].message

        if (tool_calls := completion_message.tool_calls) and (toolset := toolset):
            messages.extend(
                await _execute_chat_tool_calls(
                    tool_calls=tool_calls,
                    toolset=toolset,
                )
            )

        elif message := completion_message.content:
            await ctx.record(ResultTrace(message))
            return message

        else:
            raise ToolException("Invalid OpenAI completion", completion)

    # recursion outside of context
    return await _chat_completion(
        client=client,
        config=config,
        messages=messages,
        toolset=toolset,
    )
