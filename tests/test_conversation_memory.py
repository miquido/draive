from collections.abc import Sequence

import pytest
from haiway import Paginated, Pagination

from draive.conversation.state import ConversationMemory
from draive.conversation.types import (
    ConversationAssistantTurn,
    ConversationEvent,
    ConversationUserTurn,
)
from draive.models import ModelInput, ModelOutput, ModelToolRequest, ModelToolResponse
from draive.multimodal import MultimodalContent


@pytest.mark.asyncio
async def test_volatile_memory_recall_converts_turns_to_model_context() -> None:
    memory = ConversationMemory.volatile()
    await memory.remember(
        ConversationUserTurn.of(MultimodalContent.of("hello")),
        ConversationAssistantTurn.of(MultimodalContent.of("hi there")),
    )

    context = await memory.recall()

    assert len(context) == 2
    assert isinstance(context[0], ModelInput)
    assert context[0].content.to_str() == "hello"
    assert isinstance(context[1], ModelOutput)
    assert context[1].content.to_str() == "hi there"


@pytest.mark.asyncio
async def test_volatile_memory_recall_extracts_tool_blocks_from_turn_events() -> None:
    tool_request = ModelToolRequest.of(
        "call_1",
        tool="search",
        arguments={"query": "draive"},
    )
    tool_response = ModelToolResponse.of(
        "call_1",
        tool="search",
        content=MultimodalContent.of("result"),
    )

    memory = ConversationMemory.volatile()
    await memory.remember(
        ConversationUserTurn.of(MultimodalContent.of("user input")),
        ConversationAssistantTurn.of(
            ConversationEvent.tool_request(tool_request),
            MultimodalContent.of("assistant output"),
            ConversationEvent.tool_response(tool_response),
        ),
    )

    context: Sequence[ModelInput | ModelOutput] = await memory.recall()

    assert len(context) == 3
    assert isinstance(context[0], ModelInput)
    assert context[0].content.to_str() == "user input"
    assert isinstance(context[1], ModelOutput)
    assert context[1].tool_requests[0].identifier == "call_1"
    assert isinstance(context[2], ModelInput)
    assert context[2].tool_responses[0].identifier == "call_1"


@pytest.mark.asyncio
async def test_volatile_memory_fetch_returns_immutable_snapshot() -> None:
    memory = ConversationMemory.volatile()
    await memory.remember(ConversationUserTurn.of(MultimodalContent.of("hello")))

    fetched = await memory.fetch()
    assert isinstance(fetched, Paginated)
    assert len(fetched) == 1

    fetched_list = list(fetched)
    fetched_list.append(ConversationUserTurn.of(MultimodalContent.of("intruder")))

    fetched_again = await memory.fetch()
    assert len(fetched_again) == 1
    assert fetched_again[0].content[0].to_str() == "hello"


@pytest.mark.asyncio
async def test_volatile_memory_recall_uses_pagination_limit() -> None:
    memory = ConversationMemory.volatile()
    await memory.remember(
        ConversationUserTurn.of(MultimodalContent.of("first")),
        ConversationAssistantTurn.of(MultimodalContent.of("second")),
        ConversationUserTurn.of(MultimodalContent.of("third")),
    )

    context = await memory.recall(Pagination.of(limit=2))

    assert len(context) == 2
    assert isinstance(context[0], ModelOutput)
    assert context[0].content.to_str() == "second"
    assert isinstance(context[1], ModelInput)
    assert context[1].content.to_str() == "third"


def test_conversation_turn_preserves_content_by_default() -> None:
    user_turn = ConversationUserTurn.of(MultimodalContent.of("hello"))
    assistant_turn = ConversationAssistantTurn.of(MultimodalContent.of("hi"))

    assert user_turn.content[0].to_str() == "hello"
    assert assistant_turn.content[0].to_str() == "hi"


@pytest.mark.asyncio
async def test_volatile_memory_preserves_turn_order_in_remember() -> None:
    memory = ConversationMemory.volatile()
    existing_turn = ConversationUserTurn.of(MultimodalContent.of("earlier"))
    await memory.remember(existing_turn)

    user_turn = ConversationUserTurn.of(MultimodalContent.of("hello"))
    assistant_turn = ConversationAssistantTurn.of(MultimodalContent.of("hi"))
    await memory.remember(user_turn, assistant_turn)

    remembered_turns = await memory.fetch()
    assert len(remembered_turns) == 3
    assert remembered_turns[0].identifier == existing_turn.identifier
    assert remembered_turns[1].identifier == user_turn.identifier
    assert remembered_turns[2].identifier == assistant_turn.identifier
