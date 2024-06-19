from collections.abc import Sequence
from typing import Literal, overload

from draive.conversation.model import ConversationMessage, ConversationResponseStream
from draive.conversation.state import Conversation
from draive.lmm import AnyTool, Toolbox
from draive.scope import ctx
from draive.types import Instruction, Memory, MultimodalContent, MultimodalContentConvertible

__all__ = [
    "conversation_completion",
]


@overload
async def conversation_completion(
    *,
    instruction: Instruction | str,
    input: ConversationMessage | MultimodalContent | MultimodalContentConvertible,  # noqa: A002
    memory: Memory[Sequence[ConversationMessage], ConversationMessage]
    | Sequence[ConversationMessage]
    | None = None,
    tools: Toolbox | Sequence[AnyTool] | None = None,
    stream: Literal[True],
) -> ConversationResponseStream: ...


@overload
async def conversation_completion(
    *,
    instruction: Instruction | str,
    input: ConversationMessage | MultimodalContent | MultimodalContentConvertible,  # noqa: A002
    memory: Memory[Sequence[ConversationMessage], ConversationMessage]
    | Sequence[ConversationMessage]
    | None = None,
    tools: Toolbox | Sequence[AnyTool] | None = None,
    stream: Literal[False] = False,
) -> ConversationMessage: ...


@overload
async def conversation_completion(
    *,
    instruction: Instruction | str,
    input: ConversationMessage | MultimodalContent | MultimodalContentConvertible,  # noqa: A002
    memory: Memory[Sequence[ConversationMessage], ConversationMessage]
    | Sequence[ConversationMessage]
    | None = None,
    tools: Toolbox | Sequence[AnyTool] | None = None,
    stream: bool,
) -> ConversationResponseStream | ConversationMessage: ...


async def conversation_completion(
    *,
    instruction: Instruction | str,
    input: ConversationMessage | MultimodalContent | MultimodalContentConvertible,  # noqa: A002
    memory: Memory[Sequence[ConversationMessage], ConversationMessage]
    | Sequence[ConversationMessage]
    | None = None,
    tools: Toolbox | Sequence[AnyTool] | None = None,
    stream: bool = False,
) -> ConversationResponseStream | ConversationMessage:
    conversation: Conversation = ctx.state(Conversation)
    return await conversation.completion(
        instruction=instruction,
        input=input,
        memory=memory or conversation.memory,
        tools=tools,
        stream=stream,
    )
