from collections.abc import Sequence
from typing import Any

from haiway import MISSING

from draive.models import (
    ModelInput,
    ModelOutput,
    ModelToolRequest,
    ModelToolResponse,
)
from draive.multimodal import MultimodalContent, TextContent
from draive.vllm.messages import _context_messages


def _messages(
    *elements: ModelInput | ModelOutput,
) -> Sequence[Any]:
    return tuple(
        _context_messages(
            instructions="instructions",
            context=elements,
            vision_details=MISSING,
        )
    )


def test_context_messages_include_tool_responses() -> None:
    messages = _messages(
        ModelOutput.of(
            ModelToolRequest.of(
                "call-1",
                tool="echo",
                arguments={"value": "hello"},
            ),
        ),
        ModelInput.of(
            ModelToolResponse.of(
                "call-1",
                tool="echo",
                content=MultimodalContent.of(TextContent.of("hello")),
            ),
        ),
    )

    # system, assistant with the tool call and the matching tool result
    assert [message["role"] for message in messages] == ["system", "assistant", "tool"]

    tool_message = messages[2]
    assert tool_message["tool_call_id"] == "call-1"
    assert [part["text"] for part in tool_message["content"]] == ["hello"]


def test_context_messages_skip_empty_user_content() -> None:
    messages = _messages(
        ModelInput.of(
            ModelToolResponse.of(
                "call-1",
                tool="echo",
                content=MultimodalContent.of(TextContent.of("hello")),
            ),
        ),
    )

    # a tool only turn must not send an empty user message
    assert [message["role"] for message in messages] == ["system", "tool"]


def test_context_messages_keep_user_content_with_tool_responses() -> None:
    messages = _messages(
        ModelInput.of(
            MultimodalContent.of(TextContent.of("question")),
            ModelToolResponse.of(
                "call-1",
                tool="echo",
                content=MultimodalContent.of(TextContent.of("hello")),
            ),
        ),
    )

    assert [message["role"] for message in messages] == ["system", "user", "tool"]


def test_context_messages_omit_empty_assistant_content() -> None:
    messages = _messages(
        ModelOutput.of(
            ModelToolRequest.of(
                "call-1",
                tool="echo",
                arguments={"value": "hello"},
            ),
        ),
    )

    assistant_message = messages[1]
    assert assistant_message["role"] == "assistant"
    # an empty content array is rejected by servers validating message content
    assert "content" not in assistant_message
    assert [call["id"] for call in assistant_message["tool_calls"]] == ["call-1"]


def test_context_messages_keep_assistant_content_with_tool_calls() -> None:
    messages = _messages(
        ModelOutput.of(
            MultimodalContent.of(TextContent.of("checking")),
            ModelToolRequest.of(
                "call-1",
                tool="echo",
                arguments={"value": "hello"},
            ),
        ),
    )

    assistant_message = messages[1]
    assert [part["text"] for part in assistant_message["content"]] == ["checking"]
    assert [call["id"] for call in assistant_message["tool_calls"]] == ["call-1"]
