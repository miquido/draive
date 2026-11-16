from collections.abc import Sequence
from typing import Any

from draive.mistral.completions import _context_messages  # pyright: ignore[reportPrivateUsage]
from draive.models import (
    ModelInput,
    ModelOutput,
    ModelReasoning,
    ModelToolRequest,
    ModelToolResponse,
)
from draive.multimodal import MultimodalContent, TextContent


def _messages(
    *elements: ModelInput | ModelOutput,
) -> Sequence[Any]:
    return tuple(_context_messages(elements))


def _response(identifier: str) -> ModelToolResponse:
    return ModelToolResponse.of(
        identifier,
        tool="echo",
        content=MultimodalContent.of(TextContent.of("done")),
    )


def test_tool_messages_directly_follow_the_tool_calls_and_user_content_comes_after() -> None:
    messages = _messages(
        ModelOutput.of(ModelToolRequest.of("call-1", tool="echo", arguments={})),
        ModelInput.of(
            MultimodalContent.of(TextContent.of("question")),
            _response("call-1"),
        ),
    )

    assert [message["role"] for message in messages] == ["assistant", "tool", "user"]


def test_parallel_tool_calls_share_a_single_assistant_message() -> None:
    messages = _messages(
        ModelOutput.of(
            MultimodalContent.of(TextContent.of("checking")),
            ModelToolRequest.of("call-1", tool="echo", arguments={}),
            ModelToolRequest.of("call-2", tool="echo", arguments={}),
        ),
        ModelInput.of(_response("call-1"), _response("call-2")),
    )

    assert [message["role"] for message in messages] == ["assistant", "tool", "tool"]
    assert [call["id"] for call in messages[0]["tool_calls"]] == ["call-1", "call-2"]
    assert [chunk["text"] for chunk in messages[0]["content"]] == ["checking"]


def test_output_with_reasoning_and_text_is_a_single_assistant_message() -> None:
    messages = _messages(
        ModelOutput.of(
            ModelReasoning.of(TextContent.of("thinking"), meta={"kind": "thinking"}),
            MultimodalContent.of(TextContent.of("answer")),
        ),
    )

    assert [message["role"] for message in messages] == ["assistant"]
    assert [chunk["type"] for chunk in messages[0]["content"]] == ["thinking", "text"]


def test_tool_calls_without_content_carry_empty_content() -> None:
    messages = _messages(
        ModelOutput.of(ModelToolRequest.of("call-1", tool="echo", arguments={})),
    )

    assert messages[0]["content"] == ""
