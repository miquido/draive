from typing import Literal, overload

from openai.types.chat import ChatCompletionContentPartParam, ChatCompletionMessageParam

from draive.openai.chat_response import _chat_response  # pyright: ignore[reportPrivateUsage]
from draive.openai.chat_stream import _chat_stream  # pyright: ignore[reportPrivateUsage]
from draive.openai.client import OpenAIClient
from draive.openai.config import OpenAIChatConfig
from draive.scope import ctx
from draive.tools import ToolsProgressContext
from draive.types import (
    ConversationMessage,
    ConversationMessageContent,
    ConversationResponseStream,
    ConversationStreamingUpdate,
    ImageURLContent,
    ProgressUpdate,
    Toolset,
)
from draive.utils import AsyncStreamTask

__all__ = [
    "openai_chat_completion",
]


@overload
async def openai_chat_completion(
    *,
    config: OpenAIChatConfig,
    instruction: str,
    input: ConversationMessage | ConversationMessageContent,  # noqa: A002
    history: list[ConversationMessage] | None = None,
    toolset: Toolset | None = None,
    stream: Literal[True],
) -> ConversationResponseStream:
    ...


@overload
async def openai_chat_completion(
    *,
    config: OpenAIChatConfig,
    instruction: str,
    input: ConversationMessage | ConversationMessageContent,  # noqa: A002
    history: list[ConversationMessage] | None = None,
    toolset: Toolset | None = None,
    stream: ProgressUpdate[ConversationStreamingUpdate],
) -> str:
    ...


@overload
async def openai_chat_completion(
    *,
    config: OpenAIChatConfig,
    instruction: str,
    input: ConversationMessage | ConversationMessageContent,  # noqa: A002
    history: list[ConversationMessage] | None = None,
    toolset: Toolset | None = None,
) -> str:
    ...


async def openai_chat_completion(  # noqa: PLR0913
    *,
    config: OpenAIChatConfig,
    instruction: str,
    input: ConversationMessage | ConversationMessageContent,  # noqa: A002
    history: list[ConversationMessage] | None = None,
    toolset: Toolset | None = None,
    stream: ProgressUpdate[ConversationStreamingUpdate] | bool = False,
) -> ConversationResponseStream | str:
    with ctx.nested("openai_chat_completion", config):
        client: OpenAIClient = ctx.dependency(OpenAIClient)
        messages: list[ChatCompletionMessageParam] = _prepare_messages(
            config=config,
            instruction=instruction,
            history=history or [],
            input=input,
            limit=config.context_messages_limit,
        )
        match stream:
            case False:
                return await _chat_response(
                    client=client,
                    config=config,
                    messages=messages,
                    toolset=toolset,
                )

            case True:

                async def stream_task(
                    progress: ProgressUpdate[ConversationStreamingUpdate],
                ) -> None:
                    with ctx.updated(
                        ToolsProgressContext(
                            progress=progress or (lambda update: None),
                        ),
                    ):
                        await _chat_stream(
                            client=client,
                            config=config,
                            messages=messages,
                            toolset=toolset,
                            progress=progress,
                        )

                return AsyncStreamTask(job=stream_task)

            case progress:
                with ctx.updated(
                    ToolsProgressContext(
                        progress=progress or (lambda update: None),
                    ),
                ):
                    return await _chat_stream(
                        client=client,
                        config=config,
                        messages=messages,
                        toolset=toolset,
                        progress=progress,
                    )


def _prepare_messages(
    config: OpenAIChatConfig,
    instruction: str,
    history: list[ConversationMessage],
    input: ConversationMessage | ConversationMessageContent,  # noqa: A002
    limit: int,
) -> list[ChatCompletionMessageParam]:
    assert limit > 0, "Messages limit has to be greater than zero"  # nosec: B101
    input_message: ChatCompletionMessageParam
    if isinstance(input, ConversationMessage):
        input_message = _convert_message(
            config=config,
            message=input,
        )
    else:
        input_message = _convert_message(
            config=config,
            message=ConversationMessage(
                role="user",
                content=input,
            ),
        )

    messages: list[ChatCompletionMessageParam] = []
    for message in history:
        try:
            messages.append(
                _convert_message(
                    config=config,
                    message=message,
                )
            )
        except ValueError:
            ctx.log_error(
                "Invalid message: %s Ignoring memory.",
                message,
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


def _convert_message(
    config: OpenAIChatConfig,
    message: ConversationMessage,
) -> ChatCompletionMessageParam:
    match message.role:
        case "user":
            if isinstance(message.content, str):
                return {
                    "role": "user",
                    "content": message.content,
                }
            elif isinstance(message.content, ImageURLContent):
                return {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": message.content.url,
                                "detail": config.vision_details,
                            },
                        }
                    ],
                }
            elif isinstance(message.content, list):
                content_parts: list[ChatCompletionContentPartParam] = []
                for part in message.content:
                    if isinstance(part, str):
                        return {
                            "role": "user",
                            "content": part,
                        }
                    elif isinstance(part, ImageURLContent):
                        return {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": part.url,
                                        "detail": config.vision_details,
                                    },
                                }
                            ],
                        }
                    else:
                        raise ValueError("Unsupported message content", message)
                return {
                    "role": "user",
                    "content": content_parts,
                }
            else:
                raise ValueError("Unsupported message content", message)

        case "assistant":
            if isinstance(message.content, str):
                return {
                    "role": "assistant",
                    "content": message.content,
                }
            else:
                raise ValueError("Invalid assistant message", message)

        case other:
            raise ValueError("Invalid message role", other)
