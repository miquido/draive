from typing import cast

from draive.metrics import ArgumentsTrace, ResultTrace, TokenUsage
from draive.mistral.chat_tools import (
    _execute_chat_tool_calls,  # pyright: ignore[reportPrivateUsage]
)
from draive.mistral.client import MistralClient
from draive.mistral.config import MistralChatConfig
from draive.mistral.errors import MistralException
from draive.mistral.models import ChatCompletionResponse, ChatMessage, ChatMessageResponse
from draive.scope import ctx
from draive.tools import Toolbox

__all__ = [
    "_chat_response",
]


async def _chat_response(
    *,
    client: MistralClient,
    config: MistralChatConfig,
    messages: list[ChatMessage],
    tools: Toolbox,
    recursion_level: int = 0,
) -> str:
    if recursion_level > config.recursion_limit:
        raise MistralException("Reached limit of recursive calls of %d", config.recursion_limit)

    with ctx.nested(
        "chat_response",
        metrics=[ArgumentsTrace.of(messages=messages.copy())],
    ):
        suggest_tools: bool
        available_tools: list[dict[str, object]]
        if recursion_level == 0 and (suggested := tools.suggested_tool):
            # suggest/require tool call only initially
            suggest_tools = True
            available_tools = cast(list[dict[str, object]], [suggested])
        else:
            suggest_tools = False
            available_tools = cast(
                list[dict[str, object]],
                tools.available_tools if tools else [],
            )

        completion: ChatCompletionResponse = await client.chat_completion(
            config=config,
            messages=messages,
            tools=available_tools,
            suggest_tools=suggest_tools,
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
            raise MistralException("Invalid Mistral completion - missing messages!", completion)

        completion_message: ChatMessageResponse = completion.choices[0].message

        if (tool_calls := completion_message.tool_calls) and (tools := tools):
            ctx.record(ResultTrace.of(tool_calls))

            tools_result: list[ChatMessage] | str = await _execute_chat_tool_calls(
                tool_calls=tool_calls,
                tools=tools,
            )

            if isinstance(tools_result, str):
                return tools_result
            else:
                messages.extend(tools_result)

        elif message := completion_message.content:
            ctx.record(ResultTrace.of(message))
            match message:
                case str(content):
                    return content

                # API docs say that it can be only a string in response
                # however library model allows list as well
                case other:
                    return str(other)

        else:
            raise MistralException("Invalid Mistral completion", completion)

    # recursion outside of context
    return await _chat_response(
        client=client,
        config=config,
        messages=messages,
        tools=tools,
        recursion_level=recursion_level + 1,
    )
