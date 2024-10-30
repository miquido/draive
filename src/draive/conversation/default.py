from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from haiway import Missing, ctx, not_missing

from draive.conversation.types import ConversationMessage
from draive.instructions import Instruction
from draive.lmm import Toolbox, lmm_invocation
from draive.types import (
    LMMCompletion,
    LMMContextElement,
    LMMToolRequests,
    LMMToolResponse,
    Memory,
    MultimodalContent,
)

__all__: list[str] = [
    "default_conversation_completion",
]


async def default_conversation_completion(
    *,
    instruction: Instruction | str | None,
    message: ConversationMessage,
    memory: Memory[Iterable[ConversationMessage], ConversationMessage],
    toolbox: Toolbox,
    **extra: Any,
) -> ConversationMessage:
    with ctx.scope("conversation_completion"):
        context: list[LMMContextElement]

        messages: Iterable[ConversationMessage] | Missing = await memory.recall()
        if not_missing(messages):
            context = [
                *(message.as_lmm_context_element() for message in messages),
                message.as_lmm_context_element(),
            ]

        else:
            context = [message.as_lmm_context_element()]

        return await _conversation_completion(
            instruction=instruction,
            message=message,
            memory=memory,
            context=context,
            toolbox=toolbox,
            **extra,
        )


async def _conversation_completion(
    instruction: Instruction | str | None,
    message: ConversationMessage,
    memory: Memory[Iterable[ConversationMessage], ConversationMessage],
    context: list[LMMContextElement],
    toolbox: Toolbox,
    **extra: Any,
) -> ConversationMessage:
    recursion_level: int = 0
    while recursion_level <= toolbox.recursion_limit:
        match await lmm_invocation(
            instruction=instruction,
            context=context,
            tools=toolbox.available_tools(),
            tool_selection=toolbox.tool_selection(recursion_level=recursion_level),
            output="text",
            **extra,
        ):
            case LMMCompletion() as completion:
                ctx.log_debug("Received conversation result")
                response_message: ConversationMessage = ConversationMessage(
                    role="model",
                    created=datetime.now(UTC),
                    content=completion.content,
                )
                await memory.remember(
                    message,
                    response_message,
                )
                return response_message

            case LMMToolRequests() as tool_requests:
                ctx.log_debug("Received conversation tool calls")

                responses: list[LMMToolResponse] = await toolbox.respond(tool_requests)

                if direct_content := [
                    response.content for response in responses if response.direct
                ]:
                    response_message: ConversationMessage = ConversationMessage(
                        role="model",
                        created=datetime.now(UTC),
                        content=MultimodalContent.of(*direct_content),
                    )
                    await memory.remember(
                        message,
                        response_message,
                    )
                    return response_message

                else:
                    context.extend([tool_requests, *responses])

        recursion_level += 1  # continue with next recursion level

    raise RuntimeError("LMM exceeded limit of recursive calls")
