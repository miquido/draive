from collections.abc import AsyncIterator, Sequence
from types import SimpleNamespace
from typing import Any

import pytest
from anthropic.types import (
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
)
from haiway import ctx

from draive.anthropic.config import AnthropicConfig
from draive.anthropic.messages import AnthropicMessages
from draive.models import ModelOutputFailed, ModelOutputLimit, ModelTools


def _iter_async[T](items: Sequence[T]) -> AsyncIterator[T]:
    async def _iterator() -> AsyncIterator[T]:
        for item in items:
            yield item

    return _iterator()


class _FakeAnthropicStream:
    def __init__(self, events: Sequence[Any]) -> None:
        self._events = events

    async def __aenter__(self) -> _FakeAnthropicStream:
        return self

    async def __aexit__(
        self,
        _type: Any,
        _value: Any,
        _tb: Any,
    ) -> None:
        return None

    def __aiter__(self) -> AsyncIterator[Any]:
        return _iter_async(self._events)


@pytest.mark.asyncio
async def test_anthropic_stream_reports_context_window_exceeded_as_output_failed() -> None:
    events = [
        RawMessageStartEvent(
            type="message_start",
            message={
                "id": "msg-1",
                "content": [],
                "model": "claude-test",
                "role": "assistant",
                "stop_reason": None,
                "stop_sequence": None,
                "type": "message",
                "usage": {
                    "input_tokens": 3,
                    "output_tokens": 0,
                },
            },
        ),
        RawMessageDeltaEvent(
            type="message_delta",
            delta={
                "stop_reason": "model_context_window_exceeded",
                "stop_sequence": None,
            },
            usage={"output_tokens": 2},
        ),
        RawMessageStopEvent(type="message_stop"),
    ]

    def _stream(**_: Any) -> _FakeAnthropicStream:
        return _FakeAnthropicStream(events)

    fake_client = SimpleNamespace(
        messages=SimpleNamespace(
            stream=_stream,
        )
    )

    model = object.__new__(AnthropicMessages)
    model._provider = "anthropic"
    model._client = fake_client

    async with ctx.scope("test"):
        stream = model.completion(
            instructions="system",
            tools=ModelTools.none,
            context=(),
            output="text",
            config=AnthropicConfig(model="claude-test", max_output_tokens=64),
        )
        # an exceeded context window is an input side limit, not a continuable truncation,
        # so it has to stay distinct from `max_tokens` - matching the bedrock provider
        with pytest.raises(ModelOutputFailed):
            _ = [chunk async for chunk in stream]


@pytest.mark.asyncio
async def test_anthropic_stream_reports_max_tokens_as_output_limit() -> None:
    events = [
        RawMessageStartEvent(
            type="message_start",
            message={
                "id": "msg-1",
                "content": [],
                "model": "claude-test",
                "role": "assistant",
                "stop_reason": None,
                "stop_sequence": None,
                "type": "message",
                "usage": {
                    "input_tokens": 3,
                    "output_tokens": 0,
                },
            },
        ),
        RawMessageDeltaEvent(
            type="message_delta",
            delta={
                "stop_reason": "max_tokens",
                "stop_sequence": None,
            },
            usage={"output_tokens": 2},
        ),
        RawMessageStopEvent(type="message_stop"),
    ]

    def _stream(**_: Any) -> _FakeAnthropicStream:
        return _FakeAnthropicStream(events)

    fake_client = SimpleNamespace(
        messages=SimpleNamespace(
            stream=_stream,
        )
    )

    model = object.__new__(AnthropicMessages)
    model._provider = "anthropic"
    model._client = fake_client

    async with ctx.scope("test"):
        stream = model.completion(
            instructions="system",
            tools=ModelTools.none,
            context=(),
            output="text",
            config=AnthropicConfig(model="claude-test", max_output_tokens=64),
        )
        with pytest.raises(ModelOutputLimit):
            _ = [chunk async for chunk in stream]
