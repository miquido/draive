from collections.abc import AsyncIterator, Sequence
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest
from anthropic.types import (
    InputJSONDelta,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RedactedThinkingBlock,
    ServerToolUseBlock,
    SignatureDelta,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolUseBlock,
)
from haiway import Missing, ctx

from draive.anthropic.config import AnthropicConfig
from draive.anthropic.messages import (
    AnthropicMessages,
    _context_messages,  # pyright: ignore[reportPrivateUsage]
    _system,  # pyright: ignore[reportPrivateUsage]
)
from draive.models import (
    ModelInput,
    ModelOutput,
    ModelOutputFailed,
    ModelOutputInvalid,
    ModelQuotaLimit,
    ModelRateLimit,
    ModelReasoning,
    ModelReasoningChunk,
    ModelToolRequest,
    ModelToolResponse,
    ModelTools,
)
from draive.multimodal import ArtifactContent, MultimodalContent, TextContent
from draive.resources import ResourceContent, ResourceReference


class _Stream:
    def __init__(
        self,
        events: Sequence[Any],
        *,
        failure: Exception | None = None,
    ) -> None:
        self._events = events
        self._failure = failure

    async def __aenter__(self) -> _Stream:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    def __aiter__(self) -> AsyncIterator[Any]:
        async def _iterator() -> AsyncIterator[Any]:
            for event in self._events:
                yield event

            if self._failure is not None:
                raise self._failure

        return _iterator()


async def _run(
    events: Sequence[Any],
    *,
    failure: Exception | None = None,
    requests: list[dict[str, Any]] | None = None,
) -> list[Any]:
    def _stream(**kwargs: Any) -> _Stream:
        if requests is not None:
            requests.append(kwargs)

        return _Stream(events, failure=failure)

    model = object.__new__(AnthropicMessages)
    model._provider = "anthropic"  # pyright: ignore[reportAttributeAccessIssue]
    model._client = SimpleNamespace(messages=SimpleNamespace(stream=_stream))  # pyright: ignore[reportAttributeAccessIssue]
    async with ctx.scope("test"):
        return [
            chunk
            async for chunk in model.completion(
                instructions="",
                tools=ModelTools.none,
                context=(ModelInput.of(MultimodalContent.of("hi")),),
                output="text",
                config=AnthropicConfig(model="claude-test"),
            )
        ]


def _start(block: Any, index: int = 0) -> Any:
    return RawContentBlockStartEvent(type="content_block_start", index=index, content_block=block)


def _delta(delta: Any, index: int = 0) -> Any:
    return RawContentBlockDeltaEvent(type="content_block_delta", index=index, delta=delta)


def _stop(index: int = 0) -> Any:
    return RawContentBlockStopEvent(type="content_block_stop", index=index)


def _message_delta(stop_reason: str | None = "end_turn") -> Any:
    # unvalidated, a stop reason newer than the sdk has to be handled as well
    return RawMessageDeltaEvent.model_construct(
        type="message_delta",
        delta=SimpleNamespace(stop_reason=stop_reason, stop_sequence=None),
        usage=SimpleNamespace(
            input_tokens=None,
            cache_read_input_tokens=None,
            output_tokens=1,
        ),
    )


def _tool_events(*fragments: str) -> list[Any]:
    return [
        _start(ToolUseBlock(type="tool_use", id="call-1", name="echo", input={})),
        *(
            _delta(InputJSONDelta(type="input_json_delta", partial_json=piece))
            for piece in fragments
        ),
        _stop(),
        _message_delta("tool_use"),
    ]


@pytest.mark.asyncio
async def test_thinking_block_is_closed_with_its_signature() -> None:
    chunks = await _run(
        [
            _start(ThinkingBlock(type="thinking", thinking="", signature="")),
            _delta(ThinkingDelta(type="thinking_delta", thinking="hmm")),
            _delta(SignatureDelta(type="signature_delta", signature="sig-1")),
            _stop(),
            _start(TextBlock(type="text", text="")),
            _delta(TextDelta(type="text_delta", text="answer")),
            _stop(1),
            _message_delta(),
        ]
    )

    reasoning = [chunk for chunk in chunks if isinstance(chunk, ModelReasoningChunk)]
    assert reasoning[0].reasoning_chunk.text == "hmm"
    assert reasoning[1].meta.get_str("signature") == "sig-1"
    assert reasoning[-1].final is True
    assert reasoning[-1].meta.get_str("kind") == "thinking"
    blocks = ModelReasoning.blocks(reasoning)
    assert len(blocks) == 1
    assert blocks[0].reasoning.to_str() == "hmm"
    assert blocks[0].meta.get_str("signature") == "sig-1"
    assert chunks[-1] == TextContent.of("answer")


@pytest.mark.asyncio
async def test_redacted_thinking_keeps_its_data() -> None:
    chunks = await _run(
        [
            _start(RedactedThinkingBlock(type="redacted_thinking", data="opaque")),
            _stop(),
            _message_delta(),
        ]
    )

    blocks = ModelReasoning.blocks(
        [chunk for chunk in chunks if isinstance(chunk, ModelReasoningChunk)]
    )
    assert len(blocks) == 1
    assert blocks[0].meta.get_str("kind") == "redacted_thinking"
    assert blocks[0].meta.get_str("data") == "opaque"


@pytest.mark.asyncio
async def test_two_thinking_blocks_do_not_merge() -> None:
    chunks = await _run(
        [
            _start(ThinkingBlock(type="thinking", thinking="", signature="")),
            _delta(ThinkingDelta(type="thinking_delta", thinking="one")),
            _delta(SignatureDelta(type="signature_delta", signature="sig-a")),
            _stop(),
            _start(ThinkingBlock(type="thinking", thinking="", signature=""), 1),
            _delta(ThinkingDelta(type="thinking_delta", thinking="two"), 1),
            _delta(SignatureDelta(type="signature_delta", signature="sig-b"), 1),
            _stop(1),
            _message_delta(),
        ]
    )

    blocks = ModelReasoning.blocks(
        [chunk for chunk in chunks if isinstance(chunk, ModelReasoningChunk)]
    )
    assert [(block.reasoning.to_str(), block.meta.get_str("signature")) for block in blocks] == [
        ("one", "sig-a"),
        ("two", "sig-b"),
    ]


@pytest.mark.asyncio
async def test_tool_arguments_are_assembled_from_fragments() -> None:
    chunks = await _run(_tool_events('{"a"', ': [1, 2], "b": ', '"c"}'))

    request = chunks[0]
    assert isinstance(request, ModelToolRequest)
    assert request.arguments == {"a": (1, 2), "b": "c"}


@pytest.mark.asyncio
async def test_truncated_tool_arguments_are_invalid_output() -> None:
    with pytest.raises(ModelOutputInvalid, match="decoding"):
        await _run(_tool_events('{"a": [1,'))


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", ("[1]", '"text"', "3"))
async def test_non_object_tool_arguments_are_invalid_output(arguments: str) -> None:
    with pytest.raises(ModelOutputInvalid, match="not an object"):
        await _run(_tool_events(arguments))


@pytest.mark.asyncio
async def test_unsupported_content_block_is_invalid_output() -> None:
    block = ServerToolUseBlock(type="server_tool_use", id="srv-1", name="web_search", input={})
    with pytest.raises(ModelOutputInvalid, match="server_tool_use"):
        await _run([_start(block)])


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_reason", ("refusal", "something_new"))
async def test_failing_stop_reasons_are_output_failures(stop_reason: str) -> None:
    with pytest.raises(ModelOutputFailed, match=stop_reason):
        await _run([_message_delta(stop_reason)])


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_reason", ("end_turn", "stop_sequence", "tool_use"))
async def test_regular_stop_reasons_complete_the_output(stop_reason: str) -> None:
    assert await _run([_message_delta(stop_reason)]) == []


def _status_error[Error: anthropic.APIStatusError](
    error_type: type[Error],
    status: int,
    headers: dict[str, str] | None = None,
) -> Error:
    response = httpx.Response(
        status,
        headers=headers,
        request=httpx.Request("POST", "https://api.anthropic.test/v1/messages"),
    )

    return error_type("failure", response=response, body=None)


@pytest.mark.asyncio
async def test_rate_limit_is_translated_with_retry_delay() -> None:
    error = _status_error(anthropic.RateLimitError, 429, {"Retry-After": "7"})
    with pytest.raises(ModelRateLimit) as exc_info:
        await _run([], failure=error)

    assert exc_info.value.retry_after == 7


@pytest.mark.asyncio
async def test_zero_capacity_rate_limit_is_a_quota_failure() -> None:
    error = _status_error(
        anthropic.RateLimitError,
        429,
        {"anthropic-ratelimit-tokens-limit": "0"},
    )
    with pytest.raises(ModelQuotaLimit):
        await _run([], failure=error)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_type",
    (anthropic.OverloadedError, anthropic.ServiceUnavailableError),
)
async def test_overload_is_retryable(error_type: type[anthropic.APIStatusError]) -> None:
    status = 529 if error_type is anthropic.OverloadedError else 503
    with pytest.raises(ModelRateLimit):
        await _run([], failure=_status_error(error_type, status))


@pytest.mark.asyncio
async def test_unexpected_errors_are_output_failures() -> None:
    with pytest.raises(ModelOutputFailed, match="connection dropped"):
        await _run([], failure=ConnectionError("connection dropped"))


@pytest.mark.asyncio
async def test_request_carries_the_prepared_parameters() -> None:
    requests: list[dict[str, Any]] = []
    await _run([_message_delta()], requests=requests)

    request = requests[0]
    assert request["model"] == "claude-test"
    assert request["max_tokens"] == 16384
    assert request["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


def _png() -> ResourceContent:
    return ResourceContent.of(b"\x89PNG", mime_type="image/png")


def test_tool_response_becomes_tool_result_with_error_flag() -> None:
    messages = list(
        _context_messages(
            (
                ModelInput.of(MultimodalContent.of("go")),
                ModelOutput.of(ModelToolRequest.of("call-1", tool="echo", arguments={"x": 1})),
                ModelInput.of(
                    ModelToolResponse.of(
                        "call-1",
                        tool="echo",
                        status="error",
                        content=MultimodalContent.of("boom"),
                    )
                ),
            )
        )
    )

    assert messages[1]["content"] == [
        {"id": "call-1", "type": "tool_use", "name": "echo", "input": {"x": 1}}
    ]
    result: Any = messages[2]["content"][0]
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "call-1"
    assert result["is_error"] is True
    assert list(result["content"]) == [{"type": "text", "text": "boom"}]


def test_cache_marker_lands_on_the_last_visible_element() -> None:
    messages = list(
        _context_messages(
            (
                ModelInput.of(
                    MultimodalContent.of(
                        "first",
                        ResourceContent.of(b"\x89PNG", mime_type="image/png"),
                        ArtifactContent.of({"a": 1}, category="data", hidden=True),
                    ),
                    meta={"cache": "ephemeral"},
                ),
            )
        )
    )

    content: Any = messages[0]["content"]
    assert len(content) == 2
    assert content[0]["type"] == "text"
    assert "cache_control" not in content[0]
    assert content[1]["type"] == "image"
    assert content[1]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.parametrize(
    ("instructions", "cache_ttl", "expected"),
    (
        ("", Missing, anthropic.omit),
        ("", "1h", anthropic.omit),
        ("be brief", Missing, "be brief"),
        (
            "be brief",
            "1h",
            [
                {
                    "type": "text",
                    "text": "be brief",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ],
        ),
    ),
)
def test_instructions_carry_cache_marker_only_when_configured(
    instructions: str,
    cache_ttl: Any,
    expected: Any,
) -> None:
    assert _system(instructions, cache_ttl=cache_ttl) == expected


@pytest.mark.parametrize(
    ("part", "expected"),
    (
        (
            ResourceContent.of(b"%PDF", mime_type="application/pdf"),
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf"}},
        ),
        (
            ResourceContent.of(b"plain", mime_type="text/plain"),
            {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": "plain"},
            },
        ),
        (
            ResourceReference.of("https://example.com/a.pdf", mime_type="application/pdf"),
            {"type": "document", "source": {"type": "url", "url": "https://example.com/a.pdf"}},
        ),
        (
            ResourceReference.of("https://example.com/a.png", mime_type="image/png"),
            {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}},
        ),
    ),
)
def test_resources_are_delivered_as_documents_or_images(part: Any, expected: Any) -> None:
    messages = list(_context_messages((ModelInput.of(MultimodalContent.of(part)),)))

    block: Any = messages[0]["content"][0]
    assert block["type"] == expected["type"]
    for key, value in expected["source"].items():
        assert block["source"][key] == value


@pytest.mark.parametrize(
    "part",
    (
        ResourceContent.of(b"abc", mime_type="audio/wav"),
        ResourceReference.of("https://example.com/a.mp3", mime_type="audio/mpeg"),
    ),
)
def test_unsupported_media_is_rejected(part: Any) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        list(_context_messages((ModelInput.of(MultimodalContent.of(part)),)))


def test_thinking_and_redacted_thinking_are_replayed() -> None:
    messages = list(
        _context_messages(
            (
                ModelInput.of(MultimodalContent.of("go")),
                ModelOutput.of(
                    ModelReasoning.of(
                        TextContent.of("hmm"),
                        meta={"kind": "thinking", "signature": "sig"},
                    ),
                    ModelReasoning.of(
                        TextContent.empty,
                        meta={"kind": "redacted_thinking", "data": "opaque"},
                    ),
                    MultimodalContent.of("answer"),
                ),
                ModelInput.of(MultimodalContent.of("more")),
            )
        )
    )

    assert messages[1]["content"] == [
        {"type": "thinking", "thinking": "hmm", "signature": "sig"},
        {"type": "redacted_thinking", "data": "opaque"},
        {"type": "text", "text": "answer"},
    ]


def test_foreign_reasoning_kind_is_dropped() -> None:
    messages = list(
        _context_messages(
            (
                ModelInput.of(MultimodalContent.of("go")),
                ModelOutput.of(
                    ModelReasoning.of(TextContent.of("x"), meta={"kind": "thought"}),
                    MultimodalContent.of("answer"),
                ),
                ModelInput.of(MultimodalContent.of("more")),
            )
        )
    )

    assert messages[1]["content"] == [{"type": "text", "text": "answer"}]
