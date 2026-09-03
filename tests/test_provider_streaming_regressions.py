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
from mistralai.client.models import TextChunk, ThinkChunk
from mistralai.client.models.contentchunk import UnknownContentChunk
from ollama import ChatResponse, Message

from draive.anthropic.config import AnthropicConfig
from draive.anthropic.messages import AnthropicMessages
from draive.gemini.config import GeminiConfig
from draive.gemini.generating import GeminiGenerating
from draive.mistral.completions import MistralCompletions
from draive.mistral.config import MistralChatConfig
from draive.models import (
    ModelInput,
    ModelOutputLimit,
    ModelReasoningChunk,
    ModelToolRequest,
    ModelTools,
)
from draive.multimodal import MultimodalContent, TextContent
from draive.ollama.chat import OllamaChat
from draive.ollama.config import OllamaChatConfig
from draive.vllm.config import VLLMChatConfig
from draive.vllm.messages import VLLMMessages


def _iter_async[T](items: Sequence[T]) -> AsyncIterator[T]:
    async def _iterator() -> AsyncIterator[T]:
        for item in items:
            yield item

    return _iterator()


class _FakeResponseStream:
    """Mirrors openai's AsyncStream, which is released through its own close."""

    def __init__(self, items: Sequence[Any]) -> None:
        self._items = items
        self.closed: bool = False

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self) -> AsyncIterator[Any]:
        return _iter_async(self._items)


class _FakeContentStream:
    """Mirrors gemini's response stream, which is an async generator."""

    def __init__(self, items: Sequence[Any]) -> None:
        self._items = items
        self.closed: bool = False

    async def aclose(self) -> None:
        self.closed = True

    def __aiter__(self) -> AsyncIterator[Any]:
        return _iter_async(self._items)


class _FakeEventStream:
    """Mirrors mistral's EventStreamAsync, which is an async context manager."""

    def __init__(self, items: Sequence[Any]) -> None:
        self._items = items
        self.closed: bool = False

    async def __aenter__(self) -> _FakeEventStream:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.closed = True

    def __aiter__(self) -> AsyncIterator[Any]:
        return _iter_async(self._items)


def _vllm_chunk(
    *,
    content: str | None = None,
    tool_calls: Sequence[Any] | None = None,
    finish_reason: str | None = None,
    usage: Any = None,
) -> Any:
    return SimpleNamespace(
        usage=usage,
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=content,
                    tool_calls=list(tool_calls) if tool_calls else None,
                ),
                finish_reason=finish_reason,
            )
        ],
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

    async def _create_stream(**_: Any) -> _FakeResponseStream:
        return _FakeResponseStream(stream_chunks)

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

    event_stream = _FakeEventStream([event_with_usage_and_choice, final_event])

    async def _stream_async(**_: Any) -> _FakeEventStream:
        return event_stream

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
    assert event_stream.closed


@pytest.mark.asyncio
async def test_mistral_stream_splits_parallel_tool_calls_sharing_an_index() -> None:
    # `ToolCall.index` defaults to 0 rather than being absent, so two complete calls
    # delivered in one delta must be separated by their identifiers
    first_call = _tool_call(
        index=0,
        identifier="call-1",
        name="echo",
        arguments='{"x": 1}',
    )
    second_call = _tool_call(
        index=0,
        identifier="call-2",
        name="ping",
        arguments='{"y": 2}',
    )
    event = SimpleNamespace(
        data=SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None, tool_calls=[first_call, second_call]),
                    finish_reason="stop",
                )
            ],
            model="mistral-test",
        )
    )

    event_stream = _FakeEventStream([event])

    async def _stream_async(**_: Any) -> _FakeEventStream:
        return event_stream

    fake_client = SimpleNamespace(chat=SimpleNamespace(stream_async=_stream_async))

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

    tool_requests = [chunk for chunk in chunks if isinstance(chunk, ModelToolRequest)]
    assert [(request.identifier, request.tool) for request in tool_requests] == [
        ("call-1", "echo"),
        ("call-2", "ping"),
    ]
    assert [request.arguments for request in tool_requests] == [{"x": 1}, {"y": 2}]


@pytest.mark.asyncio
async def test_mistral_stream_is_closed_when_output_limit_interrupts() -> None:
    event = SimpleNamespace(
        data=SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="partial", tool_calls=None),
                    finish_reason="length",
                )
            ],
            model="mistral-test",
        )
    )

    event_stream = _FakeEventStream([event])

    async def _stream_async(**_: Any) -> _FakeEventStream:
        return event_stream

    fake_client = SimpleNamespace(chat=SimpleNamespace(stream_async=_stream_async))

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
        with pytest.raises(ModelOutputLimit):
            _ = [chunk async for chunk in stream]

    # the response must be released even though iteration was interrupted
    assert event_stream.closed


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


@pytest.mark.asyncio
async def test_vllm_stream_requests_usage_reporting() -> None:
    recorded: dict[str, Any] = {}

    async def _create_stream(**kwargs: Any) -> _FakeResponseStream:
        recorded.update(kwargs)
        return _FakeResponseStream(
            [
                _vllm_chunk(content="hello"),
                _vllm_chunk(
                    finish_reason="stop",
                    usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
                ),
            ]
        )

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

    # usage is only reported within the stream when explicitly requested
    assert recorded["stream_options"] == {"include_usage": True}
    assert any(isinstance(chunk, TextContent) and chunk.text == "hello" for chunk in chunks)


@pytest.mark.asyncio
async def test_mistral_stream_skips_tool_calls_without_an_identifier() -> None:
    # `ToolCall.id` defaults to the literal "null" - the api validates the identifier
    # echoed back within the following turn, so a locally generated one is not an option
    first_call = _tool_call(
        index=0,
        identifier="null",
        name="echo",
        arguments='{"x": 1}',
    )
    second_call = _tool_call(
        index=1,
        identifier="tool-2",
        name="ping",
        arguments='{"y": 2}',
    )
    event = SimpleNamespace(
        data=SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None, tool_calls=[first_call, second_call]),
                    finish_reason="stop",
                )
            ],
            model="mistral-test",
        )
    )

    async def _stream_async(**_: Any) -> _FakeEventStream:
        return _FakeEventStream([event])

    model = object.__new__(MistralCompletions)
    model._client = SimpleNamespace(chat=SimpleNamespace(stream_async=_stream_async))

    async with ctx.scope("test"):
        stream = model.completion(
            instructions="system",
            tools=ModelTools.none,
            context=(),
            output="text",
            config=MistralChatConfig(model="mistral-test"),
        )
        chunks = [chunk async for chunk in stream]

    tool_requests = [chunk for chunk in chunks if isinstance(chunk, ModelToolRequest)]
    # the placeholder identifier is dropped, only the identified call remains
    assert [request.tool for request in tool_requests] == ["ping"]
    assert [request.identifier for request in tool_requests] == ["tool-2"]


@pytest.mark.asyncio
async def test_mistral_stream_emits_reasoning_chunks_and_skips_unknown_chunks() -> None:
    event = SimpleNamespace(
        data=SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=[
                            ThinkChunk(thinking=[TextChunk(text="pondering")]),
                            UnknownContentChunk(raw={"type": "future"}),
                            TextChunk(text="answer"),
                        ],
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ],
            model="mistral-test",
        )
    )

    async def _stream_async(**_: Any) -> _FakeEventStream:
        return _FakeEventStream([event])

    model = object.__new__(MistralCompletions)
    model._client = SimpleNamespace(chat=SimpleNamespace(stream_async=_stream_async))

    async with ctx.scope("test"):
        stream = model.completion(
            instructions="system",
            tools=ModelTools.none,
            context=(),
            output="text",
            config=MistralChatConfig(model="mistral-test"),
        )
        chunks = [chunk async for chunk in stream]

    reasoning = [chunk for chunk in chunks if isinstance(chunk, ModelReasoningChunk)]
    assert [chunk.reasoning_chunk.text for chunk in reasoning] == ["pondering"]
    text = [chunk for chunk in chunks if isinstance(chunk, TextContent)]
    assert [chunk.text for chunk in text] == ["answer"]


@pytest.mark.asyncio
async def test_anthropic_stream_emits_tool_request_without_arguments() -> None:
    # a tool without parameters still delivers a single empty json fragment
    events = [
        RawContentBlockStartEvent(
            type="content_block_start",
            index=0,
            content_block=ToolUseBlock(
                type="tool_use",
                id="call-1",
                name="ping",
                input={},
            ),
        ),
        RawContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=InputJSONDelta(type="input_json_delta", partial_json=""),
        ),
        RawContentBlockStopEvent(type="content_block_stop", index=0),
        RawMessageDeltaEvent(
            type="message_delta",
            delta={"stop_reason": "tool_use", "stop_sequence": None},
            usage={"output_tokens": 2},
        ),
        RawMessageStopEvent(type="message_stop"),
    ]

    def _stream(**_: Any) -> _FakeAnthropicStream:
        return _FakeAnthropicStream(events)

    model = object.__new__(AnthropicMessages)
    model._provider = "anthropic"
    model._client = SimpleNamespace(messages=SimpleNamespace(stream=_stream))

    async with ctx.scope("test"):
        chunks = [
            chunk
            async for chunk in model.completion(
                instructions="",
                tools=ModelTools.none,
                context=(),
                output="text",
                config=AnthropicConfig(model="claude-test"),
            )
        ]

    tool_requests = [chunk for chunk in chunks if isinstance(chunk, ModelToolRequest)]
    assert len(tool_requests) == 1
    assert tool_requests[0].tool == "ping"
    assert tool_requests[0].arguments == {}


@pytest.mark.asyncio
async def test_ollama_stream_is_closed_when_output_limit_interrupts() -> None:
    closed: list[bool] = []

    async def _chunks() -> AsyncIterator[Any]:
        try:
            yield ChatResponse(
                model="test",
                done=False,
                message=Message(role="assistant", content="partial"),
            )
            yield ChatResponse(
                model="test",
                done=True,
                done_reason="length",
                message=Message(role="assistant", content=""),
            )
            yield ChatResponse(  # must never be reached
                model="test",
                done=True,
                message=Message(role="assistant", content="unreachable"),
            )

        finally:
            closed.append(True)

    stream = _chunks()

    async def _chat(**_: Any) -> AsyncIterator[Any]:
        return stream

    model = object.__new__(OllamaChat)
    model._client = SimpleNamespace(chat=_chat)

    async with ctx.scope("test"):
        with pytest.raises(ModelOutputLimit):
            async for _ in model.completion(
                instructions="",
                tools=ModelTools.none,
                context=(),
                output="text",
                config=OllamaChatConfig(model="test", max_output_tokens=4),
            ):
                pass

    assert closed == [True]


@pytest.mark.asyncio
async def test_anthropic_json_output_leaves_the_context_untouched() -> None:
    requests: list[dict[str, Any]] = []

    events = [
        RawContentBlockStartEvent(
            type="content_block_start",
            index=0,
            content_block=TextBlock(type="text", text=""),
        ),
        RawContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=TextDelta(type="text_delta", text='{"a": 1}'),
        ),
        RawContentBlockStopEvent(type="content_block_stop", index=0),
        RawMessageDeltaEvent(
            type="message_delta",
            delta={"stop_reason": "end_turn", "stop_sequence": None},
            usage={"output_tokens": 4},
        ),
        RawMessageStopEvent(type="message_stop"),
    ]

    def _stream(**kwargs: Any) -> Any:
        requests.append(kwargs)
        return _FakeAnthropicStream(events)

    model = object.__new__(AnthropicMessages)
    model._provider = "anthropic"
    model._client = SimpleNamespace(messages=SimpleNamespace(stream=_stream))

    async with ctx.scope("test"):
        chunks = [
            chunk
            async for chunk in model.completion(
                instructions="",
                tools=ModelTools.none,
                context=(ModelInput.of(MultimodalContent.of("Describe Paris.")),),
                output="json",
                config=AnthropicConfig(model="claude-test"),
            )
        ]

    decoded = "".join(chunk.text for chunk in chunks if isinstance(chunk, TextContent))
    assert decoded == '{"a": 1}'  # nothing is injected into the produced content
    # a schema-less json request has no dedicated API mode, requesting it must
    # not append content of its own to the context either
    assert requests[0]["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Describe Paris."}],
        }
    ]


@pytest.mark.asyncio
async def test_vllm_stream_releases_response_when_output_limit_ends_iteration() -> None:
    response_stream = _FakeResponseStream(
        [
            _vllm_chunk(content="hello"),
            _vllm_chunk(finish_reason="length"),
        ]
    )

    async def _create_stream(**_: Any) -> _FakeResponseStream:
        return response_stream

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
            config=VLLMChatConfig(model="vllm-test", max_output_tokens=8),
        )
        with pytest.raises(ModelOutputLimit):
            _ = [chunk async for chunk in stream]

    # the response is released although the iteration ended before its completion
    assert response_stream.closed


@pytest.mark.asyncio
async def test_gemini_stream_releases_response_when_output_limit_ends_iteration() -> None:
    content_stream = _FakeContentStream(
        [
            SimpleNamespace(
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
        ]
    )

    async def _generate_content_stream(**_: Any) -> _FakeContentStream:
        return content_stream

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

    # the response is released although the iteration ended before its completion
    assert content_stream.closed


@pytest.mark.asyncio
async def test_gemini_stream_releases_response_when_consumer_stops_early() -> None:
    def _text_chunk(text: str) -> Any:
        return SimpleNamespace(
            usage_metadata=None,
            prompt_feedback=None,
            candidates=[
                SimpleNamespace(
                    finish_reason=None,
                    finish_message=None,
                    safety_ratings=None,
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(
                                text=text,
                                thought=None,
                                thought_signature=None,
                                function_call=None,
                                inline_data=None,
                                file_data=None,
                            )
                        ]
                    ),
                )
            ],
        )

    content_stream = _FakeContentStream(
        [_text_chunk("first"), _text_chunk("second"), _text_chunk("third")]
    )

    async def _generate_content_stream(**_: Any) -> _FakeContentStream:
        return content_stream

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
        received: int = 0
        async for _ in stream:
            received += 1
            break  # leave the iteration while the provider stream is still open

        assert received == 1
        assert not content_stream.closed  # nothing released while suspended at the yield

        await stream.aclose()

    # the provider stream is released where the consumer stopped, not by the collector
    assert content_stream.closed
