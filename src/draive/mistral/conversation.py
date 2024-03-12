from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Literal, overload

from draive.mistral.chat import mistral_chat_completion
from draive.mistral.chat_stream import MistralChatStreamingMessagePart, MistralChatStreamingPart
from draive.mistral.chat_tools import MistralChatStreamingToolStatus
from draive.scope import ArgumentsTrace, ResultTrace, ctx
from draive.types import (
    ConversationMessage,
    ConversationResponseStream,
    ConversationStreamingAction,
    ConversationStreamingPart,
    Memory,
    StreamingProgressUpdate,
    StringConvertible,
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
    input: ConversationMessage | StringConvertible,  # noqa: A002
    memory: Memory[ConversationMessage] | None = None,
    toolset: Toolset | None = None,
    stream: Literal[True],
) -> ConversationResponseStream:
    ...


@overload
async def mistral_conversation_completion(
    *,
    instruction: str,
    input: ConversationMessage | StringConvertible,  # noqa: A002
    memory: Memory[ConversationMessage] | None = None,
    toolset: Toolset | None = None,
    stream: StreamingProgressUpdate[ConversationStreamingPart],
) -> ConversationMessage:
    ...


@overload
async def mistral_conversation_completion(
    *,
    instruction: str,
    input: ConversationMessage | StringConvertible,  # noqa: A002
    memory: Memory[ConversationMessage] | None = None,
    toolset: Toolset | None = None,
) -> ConversationMessage:
    ...


async def mistral_conversation_completion(
    *,
    instruction: str,
    input: ConversationMessage | StringConvertible,  # noqa: A002
    memory: Memory[ConversationMessage] | None = None,
    toolset: Toolset | None = None,
    stream: StreamingProgressUpdate[ConversationStreamingPart] | bool = False,
) -> ConversationResponseStream | ConversationMessage:
    async with ctx.nested(
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
                content=str(input),
            )

        history: list[ConversationMessage]

        if memory:
            history = await memory.recall()
        else:
            history = []

        match stream:
            case True:

                @ctx.with_current
                async def stream_task(
                    progress: StreamingProgressUpdate[ConversationStreamingPart],
                ) -> None:
                    response: ConversationMessage = await _mistral_conversation_stream(
                        instruction=instruction,
                        user_message=user_message,
                        history=history,
                        toolset=toolset,
                        remember=memory.remember if memory else None,
                        progress=progress,
                    )
                    await ctx.record(ResultTrace(response))

                return AsyncStreamTask(
                    job=stream_task,
                    task_spawn=ctx.spawn_task,
                )

            case False:
                response: ConversationMessage = await _mistral_conversation_response(
                    instruction=instruction,
                    user_message=user_message,
                    history=history,
                    toolset=toolset,
                    remember=memory.remember if memory else None,
                )
                await ctx.record(ResultTrace(response))
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
                await ctx.record(ResultTrace(response))
                return response


async def _mistral_conversation_response(
    *,
    instruction: str,
    user_message: ConversationMessage,
    history: list[ConversationMessage],
    toolset: Toolset | None = None,
    remember: Callable[[list[ConversationMessage]], Awaitable[None]] | None,
) -> ConversationMessage:
    async with ctx.nested("mistral_conversation_response"):
        response: ConversationMessage = ConversationMessage(
            timestamp=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S%z"),
            role="assistant",
            content=await mistral_chat_completion(
                instruction=instruction,
                input=user_message,
                history=history,
                toolset=toolset,
            ),
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
    progress: StreamingProgressUpdate[ConversationStreamingPart],
) -> ConversationMessage:
    async with ctx.nested("mistral_conversation_stream"):
        response: ConversationMessage = ConversationMessage(
            role="assistant",
            content="",
            timestamp=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
        actions: dict[str, ConversationStreamingAction] = {}

        def progress_update(update: MistralChatStreamingPart) -> None:
            nonlocal response  # capture response state
            nonlocal actions  # capture actions state
            match update:
                case MistralChatStreamingMessagePart(content=content):
                    response = response.updated(content=response.content + content)

                case MistralChatStreamingToolStatus(
                    id=tool_id,
                    name=tool_name,
                    status=status,
                    data=data,
                ):
                    actions[tool_id] = ConversationStreamingAction(
                        id=tool_id,
                        action="TOOL_CALL",
                        name=tool_name,
                        status=status,
                        data=data,
                    )

            progress(
                ConversationStreamingPart(
                    actions=list(actions.values()),
                    message=response,
                )
            )

        await mistral_chat_completion(
            instruction=instruction,
            input=user_message,
            history=history,
            toolset=toolset,
            stream=progress_update,
        )

        if remember := remember:
            await remember(
                [
                    user_message,
                    response,
                ],
            )

        return response
