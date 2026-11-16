import asyncio
from collections.abc import Sequence
from typing import Any

import pytest
from haiway import Meta, Pagination, ctx

from draive.conversation import Conversation
from draive.conversation.realtime.default import realtime_conversation_preparing
from draive.conversation.state import ConversationMemory
from draive.conversation.types import (
    ConversationAssistantTurn,
    ConversationEvent,
    ConversationTurn,
    ConversationUserTurn,
)
from draive.models import (
    GenerativeModel,
    ModelContext,
    ModelContextElement,
    ModelInput,
    ModelOutput,
    ModelOutputSelection,
    ModelToolRequest,
    ModelToolResponse,
    ModelTools,
)
from draive.models.types import ModelOutputStream
from draive.multimodal import MultimodalContent, TextContent
from draive.tools import Toolbox

from .test_conversation_realtime_memory import _Harness, _tool_response  # pyright: ignore


def _user(text: str, **meta: str) -> ConversationUserTurn:
    return ConversationUserTurn.of(MultimodalContent.of(text), meta=meta)


def _kinds(context: Sequence[ModelContextElement]) -> list[str]:
    return [element.kind for element in context]


async def _recall(*turns: ConversationTurn) -> ModelContext:
    return await ConversationMemory.constant(turns).recall()


@pytest.mark.asyncio
async def test_recall_merges_user_turn_following_tool_response() -> None:
    interrupted = ConversationAssistantTurn.of(
        MultimodalContent.of("checking"),
        ConversationEvent.tool_request(ModelToolRequest.of("t1", tool="lookup")),
        ConversationEvent.tool_response(_tool_response("t1")),
    )
    context = await _recall(_user("first"), interrupted, _user("never mind"))

    assert _kinds(context) == ["input", "output", "input"]
    last = context[-1]
    assert isinstance(last, ModelInput)
    assert [response.identifier for response in last.tool_responses] == ["t1"]
    assert last.content.to_str() == "never mind"
    # tool responses keep leading position, content follows in order
    assert isinstance(last.input[0], ModelToolResponse)


@pytest.mark.asyncio
async def test_recall_merges_inputs_around_empty_assistant_reply() -> None:
    context = await _recall(
        _user("one", a="1"),
        ConversationAssistantTurn.of(),
        _user("two", b="2"),
    )

    assert _kinds(context) == ["input"]
    merged = context[0]
    assert isinstance(merged, ModelInput)
    assert [block.to_str() for block in merged.input if isinstance(block, MultimodalContent)] == [
        "one",
        "two",
    ]
    assert merged.meta == Meta.of({"a": "1", "b": "2"})


@pytest.mark.asyncio
async def test_recall_merges_consecutive_outputs() -> None:
    context = await _recall(
        _user("q"),
        ConversationAssistantTurn.of(MultimodalContent.of("a")),
        ConversationAssistantTurn.of(MultimodalContent.of("b")),
    )

    assert _kinds(context) == ["input", "output"]
    merged = context[1]
    assert isinstance(merged, ModelOutput)
    assert [block.to_str() for block in merged.output if isinstance(block, MultimodalContent)] == [
        "a",
        "b",
    ]


@pytest.mark.asyncio
async def test_recall_keeps_alternating_context_untouched() -> None:
    turns = (
        _user("q1"),
        ConversationAssistantTurn.of(MultimodalContent.of("a1")),
        _user("q2"),
        ConversationAssistantTurn.of(MultimodalContent.of("a2")),
    )
    expected = tuple(element for turn in turns for element in turn.to_model_context())

    context = await _recall(*turns)

    assert tuple(context) == expected


@pytest.mark.asyncio
async def test_recall_merges_custom_recalling_output() -> None:
    async def recalling(pagination: Pagination | None = None, **extra: Any) -> ModelContext:
        _ = (pagination, extra)
        return (ModelInput.of(MultimodalContent.of("a")), ModelInput.of(MultimodalContent.of("b")))

    async def unused(*args: Any, **extra: Any) -> Any:
        raise NotImplementedError

    memory = ConversationMemory(fetching=unused, recalling=recalling, remembering=unused)

    assert _kinds(await memory.recall()) == ["input"]


@pytest.mark.asyncio
async def test_realtime_interrupted_after_tool_response_replays_alternating() -> None:
    from draive.models import ModelSessionEvent

    harness = _Harness(
        [
            ModelSessionEvent.turn_completed(
                ModelOutput.of(
                    MultimodalContent.of("checking"),
                    ModelToolRequest.of("t1", tool="lookup"),
                )
            ),
            ModelSessionEvent.turn_completed(ModelInput.of(_tool_response("t1"))),
            ModelSessionEvent.turn_completed(ModelInput.of(MultimodalContent.of("never mind"))),
            TextContent.of("end"),
        ]
    )
    async with ctx.scope("test", harness.model()):
        async with realtime_conversation_preparing(
            instructions="", toolbox=Toolbox.empty, memory=harness.memory(), output="text"
        ) as session:
            await asyncio.wait_for(session.read(), 1)

    assert [turn.turn for turn in harness.remembered] == ["assistant", "user"]
    context = await _recall(*harness.remembered)
    assert _kinds(context) == ["output", "input"]
    last = context[1]
    assert isinstance(last, ModelInput)
    assert [response.identifier for response in last.tool_responses] == ["t1"]
    assert last.content.to_str() == "never mind"


@pytest.mark.asyncio
async def test_realtime_empty_reply_replays_alternating() -> None:
    context = await _recall(_user("hello"), ConversationAssistantTurn.of(), _user("again"))

    assert _kinds(context) == ["input"]


@pytest.mark.asyncio
async def test_completion_merges_message_into_trailing_input() -> None:
    captured: list[Sequence[ModelContextElement]] = []

    async def _stream() -> ModelOutputStream:
        yield TextContent.of("ok")

    def generating(
        *,
        instructions: str,
        tools: ModelTools,
        context: Sequence[ModelContextElement],
        output: ModelOutputSelection,
        **extra: Any,
    ) -> ModelOutputStream:
        _ = (instructions, tools, output, extra)
        captured.append(tuple(context))
        return _stream()

    interrupted = ConversationAssistantTurn.of(
        MultimodalContent.of("checking"),
        ConversationEvent.tool_request(ModelToolRequest.of("t1", tool="lookup")),
        ConversationEvent.tool_response(_tool_response("t1")),
    )
    memory = ConversationMemory.constant((_user("first"), interrupted))
    async with ctx.scope("test", GenerativeModel(generating=generating), Conversation()):
        _ = [chunk async for chunk in Conversation.completion(message="next", memory=memory)]

    context = captured[0]
    assert _kinds(context) == ["input", "output", "input"]
    last = context[-1]
    assert isinstance(last, ModelInput)
    assert [response.identifier for response in last.tool_responses] == ["t1"]
    assert last.content.to_str() == "next"
