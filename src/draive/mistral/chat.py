from typing import Literal, overload

from mistralai.models.chat_completion import ChatMessage

from draive.mistral.chat_response import _chat_response  # pyright: ignore[reportPrivateUsage]
from draive.mistral.chat_stream import _chat_stream  # pyright: ignore[reportPrivateUsage]
from draive.mistral.client import MistralClient
from draive.mistral.config import MistralChatConfig
from draive.scope import ctx
from draive.tools import ToolsProgressContext
from draive.types import (
    ConversationMessage,
    ConversationMessageContent,
    ConversationResponseStream,
    ConversationStreamingUpdate,
    ProgressUpdate,
    Toolset,
)
from draive.utils import AsyncStreamTask

__all__ = [
    "mistral_chat_completion",
]


@overload
async def mistral_chat_completion(
    *,
    config: MistralChatConfig,
    instruction: str,
    input: ConversationMessage | ConversationMessageContent,  # noqa: A002
    history: list[ConversationMessage] | None = None,
    toolset: Toolset | None = None,
    stream: Literal[True],
) -> ConversationResponseStream:
    ...


@overload
async def mistral_chat_completion(
    *,
    config: MistralChatConfig,
    instruction: str,
    input: ConversationMessage | ConversationMessageContent,  # noqa: A002
    history: list[ConversationMessage] | None = None,
    toolset: Toolset | None = None,
    stream: ProgressUpdate[ConversationStreamingUpdate],
) -> str:
    ...


@overload
async def mistral_chat_completion(
    *,
    config: MistralChatConfig,
    instruction: str,
    input: ConversationMessage | ConversationMessageContent,  # noqa: A002
    history: list[ConversationMessage] | None = None,
    toolset: Toolset | None = None,
) -> str:
    ...


async def mistral_chat_completion(  # noqa: PLR0913
    *,
    config: MistralChatConfig,
    instruction: str,
    input: ConversationMessage | ConversationMessageContent,  # noqa: A002
    history: list[ConversationMessage] | None = None,
    toolset: Toolset | None = None,
    stream: ProgressUpdate[ConversationStreamingUpdate] | bool = False,
) -> ConversationResponseStream | str:
    with ctx.nested("mistral_chat_completion", config):
        client: MistralClient = ctx.dependency(MistralClient)
        messages: list[ChatMessage] = _prepare_messages(
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
    instruction: str,
    history: list[ConversationMessage],
    input: ConversationMessage | ConversationMessageContent,  # noqa: A002
    limit: int,
) -> list[ChatMessage]:
    assert limit > 0, "Messages limit has to be greater than zero"  # nosec: B101
    input_message: ChatMessage
    if isinstance(input, ConversationMessage):
        input_message = _convert_message(message=input)
    else:
        input_message = _convert_message(
            message=ConversationMessage(
                role="user",
                content=input,
            )
        )

    messages: list[ChatMessage] = []
    for message in history:
        try:
            messages.append(_convert_message(message=message))
        except ValueError:
            ctx.log_error(
                "Invalid message: %s Ignoring memory.",
                message,
            )
            return [
                ChatMessage(
                    role="system",
                    content=instruction,
                ),
                input_message,
            ]
    return [
        ChatMessage(
            role="system",
            content=instruction,
        ),
        *messages[-limit:],
        input_message,
    ]


def _convert_message(
    message: ConversationMessage,
) -> ChatMessage:
    match message.role:
        case "user":
            if isinstance(message.content, str):
                return ChatMessage(
                    role="user",
                    content=message.content,
                )
            elif isinstance(message.content, list):
                content_parts: list[str] = []
                for part in message.content:
                    if isinstance(part, str):
                        content_parts.append(part)
                    else:
                        raise ValueError("Unsupported message content", message)
                return ChatMessage(
                    role="user",
                    content="\n".join(content_parts),
                )
            else:
                raise ValueError("Unsupported message content", message)

        case "assistant":
            if isinstance(message.content, str):
                return ChatMessage(
                    role="assistant",
                    content=message.content,
                )
            else:
                raise ValueError("Invalid assistant message", message)

        case other:
            raise ValueError("Invalid message role", other)
