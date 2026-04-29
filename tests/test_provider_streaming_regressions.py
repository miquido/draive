from collections.abc import AsyncIterator, Sequence
from types import SimpleNamespace
from typing import Any

import pytest
from anthropic.types import (
    InputJSONDelta,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    TextBlock,
    TextDelta,
    ToolUseBlock,
)
from google.genai.types import FinishReason
from haiway import ctx

from draive.anthropic.config import AnthropicConfig
from draive.anthropic.messages import AnthropicMessages
from draive.gemini.config import GeminiConfig
from draive.gemini.generating import GeminiGenerating
from draive.mistral.completions import MistralCompletions
from draive.mistral.config import MistralChatConfig
from draive.models import (
    ModelOutputFailed,
    ModelOutputLimit,
    ModelToolRequest,
    ModelTools,
)
from draive.multimodal import TextContent
from draive.vllm.config import VLLMChatConfig
from draive.vllm.messages import VLLMMessages


def _iter_async[T](items: Sequence[T]) -> AsyncIterator[T]:
    async def _iterator() -> AsyncIterator[T]:
        for item in items:
            yield item

    return _iterator()


def _vllm_chunk(
    *,
    content: str | None = None,
    tool_calls: Sequence[Any] | None = None,
    finish_reason: str | None = None,
) -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=content,
                    tool_calls=list(tool_calls) if tool_calls else None,
                ),
                finish_reason=finish_reason,
            )
        ]
    )


def _tool_call(
    *,
    index: int,
    identifier: str | None = None,
    name: str | None = None,
    arguments: str | dict[str, Any] | None = None,
) -> Any:
    return SimpleNamespace(
        index=index,
        id=identifier,
        function=SimpleNamespace(
            name=name,
            arguments=arguments,
        ),
    )


@pytest.mark.asyncio
async def test_vllm_stream_accumulates_tool_calls_and_emits_on_stream_end() -> None:
    stream_chunks = [
        _vllm_chunk(content="hello"),
        _vllm_chunk(
            tool_calls=[
                _tool_call(
                    index=0,
                    identifier="call-1",
                    name="ec",
                    arguments='{"x":',
                )
            ]
        ),
        _vllm_chunk(
            tool_calls=[
                _tool_call(
                    index=0,
                    name="ho",
                    arguments="1}",
                )
            ]
        ),
        _vllm_chunk(
            finish_reason="tool_calls",
        ),
    ]

    async def _create_stream(**_: Any) -> AsyncIterator[Any]:
        return _iter_async(stream_chunks)

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=_create_stream,
            )
        )
    )

    model = object.__new__(VLLMMessages)
    model._base_url = "http://localhost"
    model._client = fake_client

    async with ctx.scope("test"):
        stream = model.completion(
            instructions="system",
            tools=ModelTools.none,
            context=(),
            output="text",
            config=VLLMChatConfig(model="vllm-test"),
        )
        chunks = [chunk async for chunk in stream]

    assert any(isinstance(chunk, TextContent) and chunk.text == "hello" for chunk in chunks)
    tool_requests = [chunk for chunk in chunks if isinstance(chunk, ModelToolRequest)]
    assert len(tool_requests) == 1
    assert tool_requests[0].identifier == "call-1"
    assert tool_requests[0].tool == "echo"
    assert tool_requests[0].arguments == {"x": 1}


@pytest.mark.asyncio
async def test_gemini_stream_preserves_model_output_limit_exception() -> None:
    chunk = SimpleNamespace(
        usage_metadata=None,
        candidates=[
            SimpleNamespace(
                finish_reason=FinishReason.MAX_TOKENS,
                finish_message="Reached max tokens",
                safety_ratings=None,
                content=None,
            )
        ],
    )

    async def _generate_content_stream(**_: Any) -> AsyncIterator[Any]:
        return _iter_async([chunk])

    fake_client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(
                generate_content_stream=_generate_content_stream,
            )
        )
    )

    model = object.__new__(GeminiGenerating)
    model._client = fake_client

    async with ctx.scope("test"):
        stream = model.completion(
            instructions="system",
            context=(),
            tools=ModelTools.none,
            output="text",
            config=GeminiConfig(model="gemini-test", max_output_tokens=64),
        )
        with pytest.raises(ModelOutputLimit):
            _ = [chunk async for chunk in stream]


@pytest.mark.asyncio
async def test_mistral_stream_handles_usage_and_emits_tool_request_once() -> None:
    start_call = _tool_call(
        index=0,
        identifier="call-1",
        name="echo",
        arguments='{"x":',
    )
    continuation_call = _tool_call(
        index=0,
        name="",
        arguments="1}",
    )
    event_with_usage_and_choice = SimpleNamespace(
        data=SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="hello", tool_calls=[start_call]),
                    finish_reason=None,
                )
            ],
            model="mistral-test",
        )
    )
    final_event = SimpleNamespace(
        data=SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None, tool_calls=[continuation_call]),
                    finish_reason="stop",
                )
            ],
            model="mistral-test",
        )
    )

    async def _stream_async(**_: Any) -> AsyncIterator[Any]:
        return _iter_async([event_with_usage_and_choice, final_event])

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            stream_async=_stream_async,
        )
    )

    model = object.__new__(MistralCompletions)
    model._client = fake_client

    async with ctx.scope("test"):
        stream = model.completion(
            instructions="system",
            tools=ModelTools.none,
            context=(),
            output="text",
            config=MistralChatConfig(model="mistral-test"),
        )
        chunks = [chunk async for chunk in stream]

    assert any(isinstance(chunk, TextContent) and chunk.text == "hello" for chunk in chunks)
    tool_requests = [chunk for chunk in chunks if isinstance(chunk, ModelToolRequest)]
    assert len(tool_requests) == 1
    assert tool_requests[0].identifier == "call-1"
    assert tool_requests[0].tool == "echo"
    assert tool_requests[0].arguments == {"x": 1}


@pytest.mark.asyncio
async def test_mistral_stream_raises_on_missing_tool_call_index() -> None:
    start_call = _tool_call(
        index=0,
        identifier="call-1",
        name="echo",
        arguments='{"x":',
    )
    invalid_continuation_call = _tool_call(
        index=None,
        name="",
        arguments="1}",
    )
    first_event = SimpleNamespace(
        data=SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None, tool_calls=[start_call]),
                    finish_reason=None,
                )
            ],
            model="mistral-test",
        )
    )
    invalid_event = SimpleNamespace(
        data=SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None, tool_calls=[invalid_continuation_call]),
                    finish_reason=None,
                )
            ],
            model="mistral-test",
        )
    )

    async def _stream_async(**_: Any) -> AsyncIterator[Any]:
        return _iter_async([first_event, invalid_event])

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            stream_async=_stream_async,
        )
    )

    model = object.__new__(MistralCompletions)
    model._client = fake_client

    async with ctx.scope("test"):
        stream = model.completion(
            instructions="system",
            tools=ModelTools.none,
            context=(),
            output="text",
            config=MistralChatConfig(model="mistral-test"),
        )
        with pytest.raises(ModelOutputFailed, match="missing tool call index"):
            _ = [chunk async for chunk in stream]


class _FakeAnthropicStream:
    def __init__(self, events: Sequence[Any]) -> None:
        self._events = events

    async def __aenter__(self) -> _FakeAnthropicStream:
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: Any,
    ) -> None:
        return None

    def __aiter__(self) -> AsyncIterator[Any]:
        return _iter_async(self._events)


@pytest.mark.asyncio
async def test_anthropic_stream_emits_text_and_tool_request() -> None:
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
                    "cache_read_input_tokens": 1,
                    "output_tokens": 0,
                },
            },
        ),
        RawContentBlockStartEvent(
            type="content_block_start",
            index=0,
            content_block=TextBlock(type="text", text=""),
        ),
        RawContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=TextDelta(type="text_delta", text="hello"),
        ),
        RawContentBlockStopEvent(type="content_block_stop", index=0),
        RawContentBlockStartEvent(
            type="content_block_start",
            index=1,
            content_block=ToolUseBlock(
                type="tool_use",
                id="call-1",
                name="echo",
                input={},
            ),
        ),
        RawContentBlockDeltaEvent(
            type="content_block_delta",
            index=1,
            delta=InputJSONDelta(type="input_json_delta", partial_json='{"x":'),
        ),
        RawContentBlockDeltaEvent(
            type="content_block_delta",
            index=1,
            delta=InputJSONDelta(type="input_json_delta", partial_json="1}"),
        ),
        RawContentBlockStopEvent(type="content_block_stop", index=1),
        RawMessageDeltaEvent(
            type="message_delta",
            delta={
                "stop_reason": "tool_use",
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
            config=AnthropicConfig(model="claude-test"),
        )
        chunks = [chunk async for chunk in stream]

    assert any(isinstance(chunk, TextContent) and chunk.text == "hello" for chunk in chunks)
    tool_requests = [chunk for chunk in chunks if isinstance(chunk, ModelToolRequest)]
    assert len(tool_requests) == 1
    assert tool_requests[0].identifier == "call-1"
    assert tool_requests[0].tool == "echo"
    assert tool_requests[0].arguments == {"x": 1}


@pytest.mark.asyncio
async def test_anthropic_stream_preserves_model_output_limit_exception() -> None:
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


@pytest.mark.asyncio
async def test_anthropic_stream_treats_pause_turn_as_turn_end() -> None:
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
                "stop_reason": "pause_turn",
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
            config=AnthropicConfig(model="claude-test"),
        )
        chunks = [chunk async for chunk in stream]

    assert chunks == []
