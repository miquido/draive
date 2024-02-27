from typing import Literal, overload

from openai.types.chat import ChatCompletionMessageParam

from draive.openai.chat_response import _chat_completion  # pyright: ignore[reportPrivateUsage]
from draive.openai.chat_stream import (
    OpenAIChatStream,
    _chat_stream,  # pyright: ignore[reportPrivateUsage]
)
from draive.openai.client import OpenAIClient
from draive.openai.config import OpenAIChatConfig
from draive.scope import ctx
from draive.types import ConversationMessage, StringConvertible, Toolset

__all__ = [
    "openai_chat_completion",
]


@overload
async def openai_chat_completion(
    *,
    instruction: str,
    input: StringConvertible,  # noqa: A002
    history: list[ConversationMessage] | None = None,
    toolset: Toolset | None = None,
    stream: Literal[True],
) -> OpenAIChatStream:
    ...


@overload
async def openai_chat_completion(
    *,
    instruction: str,
    input: StringConvertible,  # noqa: A002
    history: list[ConversationMessage] | None = None,
    toolset: Toolset | None = None,
) -> str:
    ...


async def openai_chat_completion(
    *,
    instruction: str,
    input: StringConvertible,  # noqa: A002
    history: list[ConversationMessage] | None = None,
    toolset: Toolset | None = None,
    stream: bool = False,
) -> OpenAIChatStream | str:
    config: OpenAIChatConfig = ctx.state(OpenAIChatConfig)
    async with ctx.nested("openai_chat_completion"):
        client: OpenAIClient = ctx.dependency(OpenAIClient)
        if stream:
            return await _chat_stream(
                client=client,
                config=config,
                messages=_prepare_messages(
                    instruction=instruction,
                    history=history or [],
                    input=input,
                ),
                toolset=toolset,
            )

        else:
            return await _chat_completion(
                client=client,
                config=config,
                messages=_prepare_messages(
                    instruction=instruction,
                    history=history or [],
                    input=input,
                ),
                toolset=toolset,
            )


def _prepare_messages(
    instruction: str,
    history: list[ConversationMessage],
    input: StringConvertible,  # noqa: A002
) -> list[ChatCompletionMessageParam]:
    input_message: ChatCompletionMessageParam = {
        "role": "user",
        "content": str(input),
    }

    messages: list[ChatCompletionMessageParam] = []
    for message in history:
        match message.author:
            case "user":
                messages.append(
                    {
                        "role": "user",
                        "content": str(message.content),
                    },
                )

            case "assistant":
                messages.append(
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
        *messages,
        input_message,
    ]
