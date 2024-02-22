from asyncio import gather
from typing import cast

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from draive.openai.client import OpenAIClient
from draive.openai.config import OpenAIChatConfig
from draive.scope import ArgumentsTrace, ResultTrace, TokenUsage, ctx
from draive.tools import ToolException
from draive.types import ConversationMessage, StringConvertible, Toolset

__all__ = [
    "openai_chat_completion",
]


async def openai_chat_completion(
    *,
    instruction: str,
    input: StringConvertible,  # noqa: A002
    history: list[ConversationMessage] | None = None,
    toolset: Toolset | None = None,
) -> str:
    config: OpenAIChatConfig = ctx.state(OpenAIChatConfig)
    async with ctx.nested(
        "openai_chat_completion",
        ArgumentsTrace(message=input, history=history),
        config,
    ):
        result: str = await _chat_completion(
            client=ctx.dependency(OpenAIClient),
            config=config,
            messages=_prepare_messages(
                instruction=instruction,
                history=history or [],
                input=input,
            ),
            toolset=toolset,
        )
        await ctx.record(ResultTrace(result))
        return result


def _prepare_messages(
    instruction: str,
    history: list[ConversationMessage],
    input: StringConvertible,  # noqa: A002
) -> list[ChatCompletionMessageParam]:
    input_message: ChatCompletionMessageParam = {
        "role": "user",
        "content": str(input),
    }

    openai_messages: list[ChatCompletionMessageParam] = []
    for message in history:
        match message.author:
            case "user":
                openai_messages.append(
                    {
                        "role": "user",
                        "content": str(message.content),
                    },
                )

            case "assistant":
                openai_messages.append(
                    {
                        "role": "assistant",
                        "content": str(message.content),
                    },
                )

            case other:
                ctx.log_error(
                    "Invalid message author: %s Ignoring conversation memory.",
                    other,
                )
                return [
                    {
                        "role": "system",
                        "content": instruction,
                    },
                    input_message,
                ]

    return [
        {
            "role": "system",
            "content": instruction,
        },
        *openai_messages,
        input_message,
    ]


async def _chat_completion(
    *,
    client: OpenAIClient,
    config: OpenAIChatConfig,
    messages: list[ChatCompletionMessageParam],
    toolset: Toolset | None,
) -> str:
    async with ctx.nested("chat_completion"):
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
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                        for call in tool_calls
                    ],
                }
            )
            messages.extend(
                await gather(
                    *[
                        _chat_tool_call(
                            call_id=call.id,
                            name=call.function.name,
                            arguments=call.function.arguments,
                            toolset=toolset,
                        )
                        for call in tool_calls
                    ],
                    return_exceptions=False,
                ),
            )

        elif message := completion_message.content:
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


async def _chat_tool_call(
    *,
    call_id: str,
    name: str,
    arguments: str,
    toolset: Toolset,
) -> ChatCompletionMessageParam:
    try:  # make sure that tool error won't blow up whole chain
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": str(
                await toolset.call_tool(
                    name,
                    arguments=arguments,
                )
            ),
        }

    # error should be already logged by ScopeContext
    except BaseException:
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": "Error",  # TODO: refine error result message
        }
