from collections.abc import AsyncGenerator, AsyncIterable
from datetime import UTC, datetime
from typing import Any, Literal, overload

from haiway import ctx

from draive.conversation.types import ConversationMessage, ConversationOutputChunk
from draive.models import (
    GenerativeModel,
    ModelContextElement,
    ModelInput,
    ModelInstructions,
    ModelMemory,
    ModelMemoryRecall,
    ModelOutput,
    ModelReasoning,
    ModelToolRequest,
    Toolbox,
)
from draive.multimodal import ArtifactContent

__all__ = ("conversation_completion",)


@overload
async def conversation_completion(
    *,
    instructions: ModelInstructions,
    toolbox: Toolbox,
    memory: ModelMemory,
    input: ConversationMessage,
    stream: Literal[False] = False,
    **extra: Any,
) -> ConversationMessage: ...


@overload
async def conversation_completion(
    *,
    instructions: ModelInstructions,
    toolbox: Toolbox,
    memory: ModelMemory,
    input: ConversationMessage,
    stream: Literal[True],
    **extra: Any,
) -> AsyncIterable[ConversationOutputChunk]: ...


async def conversation_completion(
    *,
    instructions: ModelInstructions,
    toolbox: Toolbox,
    memory: ModelMemory,
    input: ConversationMessage,  # noqa: A002
    stream: bool = False,
    **extra: Any,
) -> AsyncIterable[ConversationOutputChunk] | ConversationMessage:
    """Default conversation completion using ``GenerativeModel.loop``.

    Normalizes memory, resolves instructions, and runs a tool-enabled loop over the
    generative model, returning a message or streaming output chunks.
    """
    if stream:
        return ctx.stream(
            _conversation_completion_stream,
            instructions=instructions,
            toolbox=toolbox,
            memory=memory,
            input=input,
            **extra,
        )

    else:
        return await _conversation_completion(
            instructions=instructions,
            toolbox=toolbox,
            memory=memory,
            input=input,
            **extra,
        )


async def _conversation_completion(
    instructions: ModelInstructions,
    toolbox: Toolbox,
    memory: ModelMemory,
    input: ConversationMessage,  # noqa: A002
    **extra: Any,
) -> ConversationMessage:
    # relying on memory recall correctness
    memory_recall: ModelMemoryRecall = await memory.recall()
    context: list[ModelContextElement] = [
        *memory_recall.context,
        ModelInput.of(input.content),
    ]

    # run input moderation in parallel - TODO: should we use sanitized input?
    result: ModelOutput = await GenerativeModel.loop(
        instructions=instructions,
        toolbox=toolbox,
        context=context,
        stream=False,
        **extra,
    )

    ctx.log_debug("...finalizing message...")
    response_message: ConversationMessage = ConversationMessage.model(
        created=datetime.now(UTC),
        content=result.content_with_reasoning,
    )

    ctx.log_debug("...remembering...")
    try:
        await memory.remember(*context[len(memory_recall.context) :])

    except Exception as exc:
        ctx.log_error(
            "Failed to remember conversation context",
            exception=exc,
        )
        raise exc

    ctx.log_debug("... response message finished!")
    return response_message


async def _conversation_completion_stream(
    instructions: ModelInstructions,
    toolbox: Toolbox,
    memory: ModelMemory,
    input: ConversationMessage,  # noqa: A002
    **extra: Any,
) -> AsyncGenerator[ConversationOutputChunk]:
    memory_recall: ModelMemoryRecall = await memory.recall()
    context: list[ModelContextElement] = [
        *memory_recall.context,
        ModelInput.of(input.content),
    ]

    async for chunk in await GenerativeModel.loop(
        instructions=instructions,
        toolbox=toolbox,
        context=context,
        stream=True,
        **extra,
    ):
        if isinstance(chunk, ModelReasoning):
            # Wrap reasoning as an artifact to stream alongside content
            yield ConversationOutputChunk.of(
                ArtifactContent.of(
                    chunk,
                    category="reasoning",
                    hidden=True,
                )
            )

        else:
            assert not isinstance(chunk, ModelToolRequest)  # nosec: B101
            yield ConversationOutputChunk.of(chunk)

    ctx.log_debug("...remembering...")
    try:
        await memory.remember(*context[len(memory_recall.context) :])

    except Exception as exc:
        ctx.log_error(
            "Failed to remember conversation context",
            exception=exc,
        )
        raise exc

    ctx.log_debug("... streaming message finished!")
