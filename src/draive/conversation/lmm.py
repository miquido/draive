from datetime import UTC, datetime
from typing import Literal, overload

from draive.conversation.completion import ConversationCompletionStream
from draive.conversation.message import (
    ConversationMessage,
    ConversationMessageContent,
    ConversationStreamingUpdate,
)
from draive.lmm import LMMCompletionMessage, lmm_completion
from draive.tools import Toolbox
from draive.types import Memory, UpdateSend
from draive.utils import AsyncStreamTask

__all__: list[str] = [
    "lmm_conversation_completion",
]


@overload
async def lmm_conversation_completion(
    *,
    instruction: str,
    input: ConversationMessage | ConversationMessageContent,  # noqa: A002
    memory: Memory[ConversationMessage] | None = None,
    tools: Toolbox | None = None,
    stream: Literal[True],
) -> ConversationCompletionStream:
    ...


@overload
async def lmm_conversation_completion(
    *,
    instruction: str,
    input: ConversationMessage | ConversationMessageContent,  # noqa: A002
    memory: Memory[ConversationMessage] | None = None,
    tools: Toolbox | None = None,
    stream: UpdateSend[ConversationStreamingUpdate],
) -> ConversationMessage:
    ...


@overload
async def lmm_conversation_completion(
    *,
    instruction: str,
    input: ConversationMessage | ConversationMessageContent,  # noqa: A002
    memory: Memory[ConversationMessage] | None = None,
    tools: Toolbox | None = None,
) -> ConversationMessage:
    ...


async def lmm_conversation_completion(
    *,
    instruction: str,
    input: ConversationMessage | ConversationMessageContent,  # noqa: A002
    memory: Memory[ConversationMessage] | None = None,
    tools: Toolbox | None = None,
    stream: UpdateSend[ConversationStreamingUpdate] | bool = False,
) -> ConversationCompletionStream | ConversationMessage:
    system_message: LMMCompletionMessage = LMMCompletionMessage(
        role="system",
        content=instruction,
    )
    user_message: ConversationMessage
    if isinstance(input, ConversationMessage):
        user_message = input

    else:
        user_message = ConversationMessage(
            timestamp=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S%z"),
            role="user",
            content=input,
        )

    context: list[LMMCompletionMessage]

    if memory:
        context = [
            system_message,
            *await memory.recall(),
            user_message,
        ]

    else:
        context = [
            system_message,
            user_message,
        ]

    match stream:
        case True:

            async def stream_task(
                update: UpdateSend[ConversationStreamingUpdate],
            ) -> None:
                nonlocal memory
                completion: LMMCompletionMessage = await lmm_completion(
                    context=context,
                    tools=tools,
                    stream=update,
                )
                response_message: ConversationMessage = ConversationMessage(
                    timestamp=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S%z"),
                    role=completion.role,
                    content=completion.content,
                )
                if memory := memory:
                    await memory.remember(
                        [
                            user_message,
                            response_message,
                        ],
                    )

            return AsyncStreamTask(job=stream_task)

        case False:
            completion: LMMCompletionMessage = await lmm_completion(
                context=context,
                tools=tools,
            )
            response_message: ConversationMessage = ConversationMessage(
                timestamp=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S%z"),
                role=completion.role,
                content=completion.content,
            )
            if memory := memory:
                await memory.remember(
                    [
                        user_message,
                        response_message,
                    ],
                )

            return response_message

        case update:
            completion: LMMCompletionMessage = await lmm_completion(
                context=context,
                tools=tools,
                stream=update,
            )
            response_message: ConversationMessage = ConversationMessage(
                timestamp=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S%z"),
                role=completion.role,
                content=completion.content,
            )
            if memory := memory:
                await memory.remember(
                    [
                        user_message,
                        response_message,
                    ],
                )

            return response_message
