from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Literal, overload

from draive.mistral.chat import mistral_chat_completion
from draive.mistral.config import MistralChatConfig
from draive.scope import ArgumentsTrace, ResultTrace, ctx
from draive.types import (
    ConversationMessage,
    ConversationMessageContent,
    ConversationResponseStream,
    ConversationStreamingUpdate,
    Memory,
    ProgressUpdate,
    Toolset,
)
from draive.utils import AsyncStreamTask

__all__ = [
    "mistral_conversation_completion",
]


@overload
async def mistral_conversation_completion(
    *,
    instruction: str,
    input: ConversationMessage | ConversationMessageContent,  # noqa: A002
    memory: Memory[ConversationMessage] | None = None,
    toolset: Toolset | None = None,
    stream: Literal[True],
) -> ConversationResponseStream:
    ...


@overload
async def mistral_conversation_completion(
    *,
    instruction: str,
    input: ConversationMessage | ConversationMessageContent,  # noqa: A002
    memory: Memory[ConversationMessage] | None = None,
    toolset: Toolset | None = None,
    stream: ProgressUpdate[ConversationStreamingUpdate],
) -> ConversationMessage:
    ...


@overload
async def mistral_conversation_completion(
    *,
    instruction: str,
    input: ConversationMessage | ConversationMessageContent,  # noqa: A002
    memory: Memory[ConversationMessage] | None = None,
    toolset: Toolset | None = None,
) -> ConversationMessage:
    ...


async def mistral_conversation_completion(
    *,
    instruction: str,
    input: ConversationMessage | ConversationMessageContent,  # noqa: A002
    memory: Memory[ConversationMessage] | None = None,
    toolset: Toolset | None = None,
    stream: ProgressUpdate[ConversationStreamingUpdate] | bool = False,
) -> ConversationResponseStream | ConversationMessage:
    with ctx.nested(
        "mistral_conversation",
        ArgumentsTrace(message=input),
    ):
        user_message: ConversationMessage
        if isinstance(input, ConversationMessage):
            user_message = input

        else:
            user_message = ConversationMessage(
                timestamp=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S%z"),
                role="user",
                content=input,
            )

        history: list[ConversationMessage]

        if memory:
            history = await memory.recall()
        else:
            history = []

        match stream:
            case True:

                async def stream_task(
                    progress: ProgressUpdate[ConversationStreamingUpdate],
                ) -> None:
                    response: ConversationMessage = await _mistral_conversation_stream(
                        instruction=instruction,
                        user_message=user_message,
                        history=history,
                        toolset=toolset,
                        remember=memory.remember if memory else None,
                        progress=progress,
                    )
                    ctx.record(ResultTrace(response))

                return AsyncStreamTask(job=stream_task)

            case False:
                response: ConversationMessage = await _mistral_conversation_response(
                    instruction=instruction,
                    user_message=user_message,
                    history=history,
                    toolset=toolset,
                    remember=memory.remember if memory else None,
                )
                ctx.record(ResultTrace(response))
                return response

            case progress:
                response: ConversationMessage = await _mistral_conversation_stream(
                    instruction=instruction,
                    user_message=user_message,
                    history=history,
                    toolset=toolset,
                    remember=memory.remember if memory else None,
                    progress=progress,
                )
                ctx.record(ResultTrace(response))
                return response


async def _mistral_conversation_response(
    *,
    instruction: str,
    user_message: ConversationMessage,
    history: list[ConversationMessage],
    toolset: Toolset | None = None,
    remember: Callable[[list[ConversationMessage]], Awaitable[None]] | None,
) -> ConversationMessage:
    with ctx.nested("mistral_conversation_response"):
        response_content: str = await mistral_chat_completion(
            config=ctx.state(MistralChatConfig),
            instruction=instruction,
            input=user_message,
            history=history,
            toolset=toolset,
        )
        response: ConversationMessage = ConversationMessage(
            role="assistant",
            author=ctx.state(MistralChatConfig).model,
            content=response_content,
            timestamp=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S%z"),
        )

        if remember := remember:
            await remember(
                [
                    user_message,
                    response,
                ],
            )

        return response


async def _mistral_conversation_stream(  # noqa: PLR0913
    *,
    instruction: str,
    user_message: ConversationMessage,
    history: list[ConversationMessage],
    toolset: Toolset | None = None,
    remember: Callable[[list[ConversationMessage]], Awaitable[None]] | None,
    progress: ProgressUpdate[ConversationStreamingUpdate],
) -> ConversationMessage:
    with ctx.nested("mistral_conversation_stream"):
        response_content: str = await mistral_chat_completion(
            config=ctx.state(MistralChatConfig),
            instruction=instruction,
            input=user_message,
            history=history,
            toolset=toolset,
            stream=progress,
        )

        response: ConversationMessage = ConversationMessage(
            role="assistant",
            author=ctx.state(MistralChatConfig).model,
            content=response_content,
            timestamp=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S%z"),
        )

        if remember := remember:
            await remember(
                [
                    user_message,
                    response,
                ],
            )

        return response
