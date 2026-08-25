"""Bedrock generation streams through ConverseStream instead of a single response."""

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import pytest
from haiway import ctx

from draive.bedrock.config import BedrockChatConfig
from draive.bedrock.converse import BedrockConverse, _request_parameters
from draive.models import (
    ModelInputInvalid,
    ModelOutputFailed,
    ModelOutputInvalid,
    ModelOutputLimit,
    ModelRateLimit,
    ModelReasoning,
    ModelReasoningChunk,
    ModelToolRequest,
    ModelTools,
)
from draive.multimodal import TextContent


class _FakeEventStream:
    """Mirrors botocore's EventStream, iterated and closed through the sdk object."""

    def __init__(self, events: Sequence[Mapping[str, Any]]) -> None:
        self._events = events
        self.closed: bool = False

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        return iter(self._events)

    def close(self) -> None:
        self.closed = True


def _model(events: Sequence[Mapping[str, Any]] | _FakeEventStream) -> Any:
    model = object.__new__(BedrockConverse)
    stream = events if isinstance(events, _FakeEventStream) else _FakeEventStream(events)

    def _converse_stream(**_: Any) -> Mapping[str, Any]:
        return {"stream": stream}

    model._client = type("_Client", (), {"converse_stream": staticmethod(_converse_stream)})()

    return model


async def _collect(
    events: Sequence[Mapping[str, Any]] | _FakeEventStream,
    **overrides: Any,
) -> list[Any]:
    arguments: dict[str, Any] = {
        "instructions": "system",
        "tools": ModelTools.none,
        "context": (),
        "output": "text",
        "config": BedrockChatConfig(model="bedrock-test"),
    }
    arguments.update(overrides)
    async with ctx.scope("test"):
        return [chunk async for chunk in _model(events).completion(**arguments)]


@pytest.mark.asyncio
async def test_text_deltas_stream_as_they_arrive() -> None:
    chunks = await _collect(
        [
            {"contentBlockDelta": {"delta": {"text": "hello "}, "contentBlockIndex": 0}},
            {"contentBlockDelta": {"delta": {"text": "world"}, "contentBlockIndex": 0}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "end_turn"}},
        ]
    )

    assert [chunk.text for chunk in chunks if isinstance(chunk, TextContent)] == [
        "hello ",
        "world",
    ]


@pytest.mark.asyncio
async def test_tool_call_accumulates_across_deltas() -> None:
    chunks = await _collect(
        [
            {
                "contentBlockStart": {
                    "start": {"toolUse": {"toolUseId": "call-1", "name": "echo"}},
                    "contentBlockIndex": 0,
                }
            },
            {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"x":'}}}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": " 1}"}}}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "tool_use"}},
        ]
    )

    requests = [chunk for chunk in chunks if isinstance(chunk, ModelToolRequest)]
    assert len(requests) == 1
    assert requests[0].identifier == "call-1"
    assert requests[0].tool == "echo"
    assert requests[0].arguments == {"x": 1}


@pytest.mark.asyncio
async def test_a_tool_without_arguments_is_still_reported() -> None:
    chunks = await _collect(
        [
            {
                "contentBlockStart": {
                    "start": {"toolUse": {"toolUseId": "call-1", "name": "ping"}},
                    "contentBlockIndex": 0,
                }
            },
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "tool_use"}},
        ]
    )

    requests = [chunk for chunk in chunks if isinstance(chunk, ModelToolRequest)]
    assert len(requests) == 1
    assert requests[0].arguments == {}


@pytest.mark.asyncio
async def test_malformed_tool_arguments_are_reported_as_invalid_output() -> None:
    with pytest.raises(ModelOutputInvalid):
        await _collect(
            [
                {
                    "contentBlockStart": {
                        "start": {"toolUse": {"toolUseId": "call-1", "name": "echo"}},
                        "contentBlockIndex": 0,
                    }
                },
                {"contentBlockDelta": {"delta": {"toolUse": {"input": "{not json"}}}},
                {"contentBlockStop": {"contentBlockIndex": 0}},
            ]
        )


@pytest.mark.asyncio
async def test_reasoning_signature_closes_its_block() -> None:
    chunks = await _collect(
        [
            {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "thinking"}}}},
            {"contentBlockDelta": {"delta": {"reasoningContent": {"signature": "sig-1"}}}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "end_turn"}},
        ]
    )

    reasoning = [chunk for chunk in chunks if isinstance(chunk, ModelReasoningChunk)]
    assert [chunk.final for chunk in reasoning] == [False, True]

    blocks = ModelReasoning.blocks(reasoning)
    assert len(blocks) == 1
    assert blocks[0].reasoning.to_str() == "thinking"
    assert blocks[0].meta.get_str("signature") == "sig-1"


@pytest.mark.asyncio
async def test_max_tokens_stop_reason_reports_the_output_limit() -> None:
    with pytest.raises(ModelOutputLimit):
        await _collect(
            [
                {"contentBlockDelta": {"delta": {"text": "partial"}}},
                {"messageStop": {"stopReason": "max_tokens"}},
            ]
        )


@pytest.mark.asyncio
async def test_event_stream_is_released_when_iteration_ends_early() -> None:
    stream = _FakeEventStream(
        [
            {"contentBlockDelta": {"delta": {"text": "partial"}}},
            {"messageStop": {"stopReason": "max_tokens"}},
        ]
    )

    with pytest.raises(ModelOutputLimit):
        await _collect(stream)

    # the response is released although the iteration ended before its completion
    assert stream.closed


@pytest.mark.asyncio
async def test_event_stream_is_released_when_iteration_completes() -> None:
    stream = _FakeEventStream([{"messageStop": {"stopReason": "end_turn"}}])

    await _collect(stream)

    assert stream.closed


@pytest.mark.asyncio
async def test_content_filtered_stop_reason_reports_a_failure() -> None:
    with pytest.raises(ModelOutputFailed):
        await _collect([{"messageStop": {"stopReason": "content_filtered"}}])


@pytest.mark.asyncio
async def test_a_streamed_throttling_member_reports_a_rate_limit() -> None:
    with pytest.raises(ModelRateLimit):
        await _collect([{"throttlingException": {"message": "slow down"}}])


@pytest.mark.asyncio
async def test_a_streamed_validation_member_reports_invalid_input() -> None:
    with pytest.raises(ModelInputInvalid):
        await _collect([{"validationException": {"message": "bad request"}}])


@pytest.mark.asyncio
async def test_usage_metadata_does_not_terminate_the_stream() -> None:
    chunks = await _collect(
        [
            {
                "metadata": {
                    "usage": {
                        "inputTokens": 10,
                        "outputTokens": 3,
                        "cacheReadInputTokens": 4,
                    }
                }
            },
            {"contentBlockDelta": {"delta": {"text": "after usage"}}},
            {"messageStop": {"stopReason": "end_turn"}},
        ]
    )

    assert [chunk.text for chunk in chunks if isinstance(chunk, TextContent)] == ["after usage"]


def test_an_inline_guardrail_reaches_the_request() -> None:
    parameters = _request_parameters(
        instructions="system",
        messages=[],
        tools=ModelTools.none,
        config=BedrockChatConfig(
            model="bedrock-test",
            guardrail_identifier="guard-1",
            guardrail_version="2",
        ),
    )

    assert parameters["guardrailConfig"] == {
        "guardrailIdentifier": "guard-1",
        "guardrailVersion": "2",
        "streamProcessingMode": "sync",
    }


def test_no_guardrail_is_configured_by_default() -> None:
    parameters = _request_parameters(
        instructions="system",
        messages=[],
        tools=ModelTools.none,
        config=BedrockChatConfig(model="bedrock-test"),
    )

    assert "guardrailConfig" not in parameters
