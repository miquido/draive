from typing import Literal, overload

from openai.types.chat import ChatCompletionMessageParam

# from draive.helpers import AsyncStream
from draive.openai.chat_response import _chat_response  # pyright: ignore[reportPrivateUsage]
from draive.openai.chat_stream import (
    OpenAIChatStream,
    OpenAIChatStreamingPart,
    _chat_stream,  # pyright: ignore[reportPrivateUsage]
)
from draive.openai.client import OpenAIClient
from draive.openai.config import OpenAIChatConfig
from draive.scope import ctx
from draive.types import ConversationMessage, StreamingProgressUpdate, StringConvertible, Toolset
from draive.utils import AsyncStreamTask

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
    stream: StreamingProgressUpdate[OpenAIChatStreamingPart],
) -> str:
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
    stream: StreamingProgressUpdate[OpenAIChatStreamingPart] | bool = False,
) -> OpenAIChatStream | str | None:
    config: OpenAIChatConfig = ctx.state(OpenAIChatConfig)
    async with ctx.nested("openai_chat_completion", config):
        client: OpenAIClient = ctx.dependency(OpenAIClient)
        match stream:
            case False:
                return await _chat_response(
                    client=client,
                    config=config,
                    messages=_prepare_messages(
                        instruction=instruction,
                        history=history or [],
                        input=input,
                        limit=config.context_messages_limit,
                    ),
                    toolset=toolset,
                )

            case True:

                @ctx.with_current
                async def stream_task(
                    progress: StreamingProgressUpdate[OpenAIChatStreamingPart],
                ) -> None:
                    await _chat_stream(
                        client=client,
                        config=config,
                        messages=_prepare_messages(
                            instruction=instruction,
                            history=history or [],
                            input=input,
                            limit=config.context_messages_limit,
                        ),
                        toolset=toolset,
                        progress=progress,
                    )

                return AsyncStreamTask(
                    job=stream_task,
                    task_spawn=ctx.spawn_task,
                )

            case progress:
                return await _chat_stream(
                    client=client,
                    config=config,
                    messages=_prepare_messages(
                        instruction=instruction,
                        history=history or [],
                        input=input,
                        limit=config.context_messages_limit,
                    ),
                    toolset=toolset,
                    progress=progress,
                )


def _prepare_messages(
    instruction: str,
    history: list[ConversationMessage],
    input: StringConvertible,  # noqa: A002
    limit: int,
) -> list[ChatCompletionMessageParam]:
    input_message: ChatCompletionMessageParam
    if isinstance(input, ConversationMessage):
        input_message = {
            "role": "user",
            "content": f"{input.author}:\n{input.content}" if input.author else input.content,
        }
    else:
        input_message = {
            "role": "user",
            "content": str(input),
        }

    messages: list[ChatCompletionMessageParam] = []
    for message in history:
        match message.role:
            case "user":
                messages.append(
                    {
                        "role": "user",
                        "content": f"{message.author}:\n{message.content}"
                        if message.author
                        else message.content,
                    },
                )

            case "assistant":
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"{message.author}:\n{message.content}"
                        if message.author
                        else message.content,
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
        *messages[-limit:],
        input_message,
    ]
