from collections.abc import AsyncIterator, Sequence
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
from haiway import ctx
from openai.types.responses import (
    ResponseErrorEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemDoneEvent,
    ResponseReasoningItem,
    ResponseReasoningSummaryPartAddedEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseRefusalDoneEvent,
    ResponseTextDeltaEvent,
)

from draive.models import (
    ModelInput,
    ModelOutput,
    ModelOutputFailed,
    ModelOutputInvalid,
    ModelOutputLimit,
    ModelQuotaLimit,
    ModelRateLimit,
    ModelReasoning,
    ModelReasoningChunk,
    ModelToolRequest,
    ModelToolResponse,
    ModelTools,
)
from draive.multimodal import MultimodalContent, TextContent
from draive.openai.config import OpenAIResponsesConfig
from draive.openai.responses import OpenAIResponses, _context_to_params  # pyright: ignore[reportPrivateUsage]
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
        self.closed: bool = False

    async def __aenter__(self) -> _Stream:
        return self

    async def __aexit__(self, *_: Any) -> None:
        self.closed = True

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
    max_output_tokens: int | None = None,
) -> list[Any]:
    model = object.__new__(OpenAIResponses)
    model._base_url = None  # pyright: ignore[reportAttributeAccessIssue]
    model._client = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        responses=SimpleNamespace(stream=lambda **_: _Stream(events, failure=failure))
    )
    config = (
        OpenAIResponsesConfig(model="gpt-6.2-sol")
        if max_output_tokens is None
        else OpenAIResponsesConfig(model="gpt-6.2-sol", max_output_tokens=max_output_tokens)
    )
    async with ctx.scope("test"):
        return [
            chunk
            async for chunk in model.completion(
                instructions="",
                tools=ModelTools.none,
                context=(ModelInput.of(MultimodalContent.of("hi")),),
                output="text",
                config=config,
            )
        ]


def _text_delta(text: str) -> Any:
    return ResponseTextDeltaEvent.model_construct(type="response.output_text.delta", delta=text)


def _summary_delta(text: str) -> Any:
    return ResponseReasoningSummaryTextDeltaEvent.model_construct(
        type="response.reasoning_summary_text.delta",
        delta=text,
    )


def _item_done(item: Any) -> Any:
    return ResponseOutputItemDoneEvent.model_construct(
        type="response.output_item.done",
        item=item,
    )


def _function_call(arguments: str) -> Any:
    return _item_done(
        ResponseFunctionToolCall(
            type="function_call",
            call_id="call-1",
            name="echo",
            arguments=arguments,
        )
    )


def _terminal(
    kind: str = "response.completed",
    *,
    error: Any = None,
    incomplete: str | None = None,
    usage: Any = None,
) -> Any:
    return SimpleNamespace(
        type=kind,
        response=SimpleNamespace(
            usage=usage,
            error=error,
            incomplete_details=SimpleNamespace(reason=incomplete) if incomplete else None,
        ),
    )


@pytest.mark.asyncio
async def test_text_deltas_are_streamed() -> None:
    chunks = await _run([_text_delta("Hel"), _text_delta("lo"), _terminal()])

    assert chunks == [TextContent.of("Hel"), TextContent.of("lo")]


@pytest.mark.asyncio
async def test_reasoning_summary_is_closed_by_the_encrypted_item() -> None:
    chunks = await _run(
        [
            _summary_delta("thinking"),
            _item_done(
                ResponseReasoningItem(
                    type="reasoning",
                    id="rs-1",
                    summary=[],
                    encrypted_content="enc",
                )
            ),
            _text_delta("answer"),
            _terminal(),
        ]
    )

    reasoning = [chunk for chunk in chunks if isinstance(chunk, ModelReasoningChunk)]
    assert reasoning[-1].final is True
    blocks = ModelReasoning.blocks(reasoning)
    assert len(blocks) == 1
    assert blocks[0].reasoning.to_str() == "thinking"
    assert blocks[0].meta.get_str("id") == "rs-1"
    assert blocks[0].meta.get_str("encrypted") == "enc"


@pytest.mark.asyncio
async def test_function_call_becomes_tool_request() -> None:
    chunks = await _run([_function_call('{"x": 1}'), _terminal()])

    assert len(chunks) == 1
    assert isinstance(chunks[0], ModelToolRequest)
    assert (chunks[0].identifier, chunks[0].tool, chunks[0].arguments) == (
        "call-1",
        "echo",
        {"x": 1},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", ('{"x":', "[1]", '"text"'))
async def test_malformed_tool_arguments_are_invalid_output(arguments: str) -> None:
    with pytest.raises(ModelOutputInvalid, match="Tool arguments"):
        await _run([_function_call(arguments)])


@pytest.mark.asyncio
async def test_refusal_is_invalid_output() -> None:
    with pytest.raises(ModelOutputInvalid, match="cannot help"):
        await _run(
            [
                ResponseRefusalDoneEvent.model_construct(
                    type="response.refusal.done",
                    refusal="cannot help",
                )
            ]
        )


@pytest.mark.asyncio
async def test_error_event_is_output_failure() -> None:
    with pytest.raises(ModelOutputFailed, match="server_error: oops"):
        await _run(
            [
                ResponseErrorEvent.model_construct(
                    type="error",
                    code="server_error",
                    message="oops",
                )
            ]
        )


@pytest.mark.asyncio
async def test_response_error_is_output_failure() -> None:
    with pytest.raises(ModelOutputFailed, match="rate: slow"):
        await _run(
            [_terminal("response.failed", error=SimpleNamespace(code="rate", message="slow"))]
        )


@pytest.mark.asyncio
async def test_other_incomplete_reason_is_invalid_output() -> None:
    with pytest.raises(ModelOutputInvalid, match="content_filter"):
        await _run([_terminal("response.incomplete", incomplete="content_filter")])


@pytest.mark.asyncio
async def test_output_limit_reports_the_configured_budget() -> None:
    with pytest.raises(ModelOutputLimit) as exc_info:
        await _run(
            [_terminal("response.incomplete", incomplete="max_output_tokens")],
            max_output_tokens=32,
        )

    assert exc_info.value.max_output_tokens == 32


@pytest.mark.asyncio
async def test_usage_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "draive.openai.responses.record_usage_metrics",
        lambda **kwargs: recorded.append(kwargs),  # pyright: ignore[reportUnknownLambdaType, reportUnknownMemberType, reportUnknownArgumentType]
    )
    await _run(
        [
            _terminal(
                usage=SimpleNamespace(
                    input_tokens=5,
                    input_tokens_details=SimpleNamespace(cached_tokens=2),
                    output_tokens=7,
                    output_tokens_details=SimpleNamespace(reasoning_tokens=3),
                )
            )
        ]
    )

    assert recorded == [
        {
            "provider": "openai",
            "model": "gpt-6.2-sol",
            "input_tokens": 5,
            "cached_input_tokens": 2,
            "output_tokens": 7,
            "reasoning_output_tokens": 3,
        }
    ]


def _rate_limit(
    headers: dict[str, str],
    body: Any = None,
) -> openai.RateLimitError:
    response = httpx.Response(
        429,
        headers=headers,
        request=httpx.Request("POST", "https://api.openai.test/v1/responses"),
    )

    return openai.RateLimitError("limited", response=response, body=body)


@pytest.mark.asyncio
async def test_rate_limit_is_translated_with_retry_delay() -> None:
    with pytest.raises(ModelRateLimit) as exc_info:
        await _run([], failure=_rate_limit({"Retry-After": "3"}))

    assert exc_info.value.retry_after == 3


@pytest.mark.asyncio
async def test_zero_capacity_rate_limit_is_a_quota_failure() -> None:
    with pytest.raises(ModelQuotaLimit):
        await _run([], failure=_rate_limit({"x-ratelimit-limit-tokens": "0"}))


@pytest.mark.asyncio
async def test_exhausted_quota_code_is_a_quota_failure() -> None:
    error = _rate_limit({}, body={"code": "insufficient_quota", "message": "out of credit"})
    with pytest.raises(ModelQuotaLimit):
        await _run([], failure=error)


@pytest.mark.asyncio
async def test_unexpected_errors_are_output_failures() -> None:
    with pytest.raises(ModelOutputFailed, match="connection dropped"):
        await _run([], failure=ConnectionError("connection dropped"))


@pytest.mark.asyncio
async def test_unsupported_context_is_reported_as_input_failure() -> None:
    from draive.models import ModelInputInvalid

    model = object.__new__(OpenAIResponses)
    model._base_url = None  # pyright: ignore[reportAttributeAccessIssue]
    model._client = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        responses=SimpleNamespace(stream=lambda **_: _Stream([]))
    )
    async with ctx.scope("test"):
        with pytest.raises(ModelInputInvalid, match="Unsupported media"):
            _ = [
                chunk
                async for chunk in model.completion(
                    instructions="",
                    tools=ModelTools.none,
                    context=(
                        ModelInput.of(
                            MultimodalContent.of(ResourceContent.of(b"abc", mime_type="audio/wav"))
                        ),
                    ),
                    output="text",
                    config=OpenAIResponsesConfig(model="gpt-6.2-sol"),
                )
            ]


def _params(*context: ModelInput | ModelOutput) -> list[dict[str, Any]]:
    return [dict(element) for element in _context_to_params(context, vision_details="auto")]


def test_tool_exchange_is_replayed_as_function_items() -> None:
    params = _params(
        ModelInput.of(MultimodalContent.of("go")),
        ModelOutput.of(ModelToolRequest.of("call-1", tool="echo", arguments={"x": 1})),
        ModelInput.of(
            ModelToolResponse.of(
                "call-1",
                tool="echo",
                content=MultimodalContent.of("done"),
            )
        ),
    )

    assert params[1] == {
        "type": "function_call",
        "call_id": "call-1",
        "name": "echo",
        "arguments": '{"x": 1}',
        "status": "completed",
    }
    assert params[2] == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": [{"type": "input_text", "text": "done"}],
    }


def test_reasoning_is_replayed_only_with_its_identity() -> None:
    params = _params(
        ModelInput.of(MultimodalContent.of("go")),
        ModelOutput.of(
            ModelReasoning.of(
                TextContent.of("kept"),
                meta={"kind": "reasoning", "id": "rs-1", "encrypted": "enc"},
            ),
            ModelReasoning.of(TextContent.of("no payload"), meta={"kind": "reasoning"}),
            ModelReasoning.of(
                TextContent.of("no id"),
                meta={"kind": "reasoning", "encrypted": "enc-2"},
            ),
            MultimodalContent.of("answer"),
        ),
    )

    reasoning = [param for param in params if param["type"] == "reasoning"]
    assert reasoning == [
        {
            "id": "rs-1",
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "kept"}],
            "encrypted_content": "enc",
        }
    ]


def test_reasoning_of_another_provider_is_dropped_instead_of_failing() -> None:
    params = _params(
        ModelInput.of(MultimodalContent.of("go")),
        ModelOutput.of(
            ModelReasoning.of(
                TextContent.of("foreign"),
                meta={"kind": "gemini_thought", "id": "x", "encrypted": "enc"},
            ),
            MultimodalContent.of("answer"),
        ),
    )

    assert [param for param in params if param["type"] == "reasoning"] == []
    assert params[-1]["type"] == "message"


@pytest.mark.parametrize(
    ("part", "expected"),
    (
        (
            ResourceContent.of(b"%PDF", mime_type="application/pdf"),
            {"type": "input_file", "filename": "file.pdf"},
        ),
        (
            ResourceReference.of("https://example.com/a.pdf", mime_type="application/pdf"),
            {"type": "input_file", "file_url": "https://example.com/a.pdf"},
        ),
        (
            ResourceReference.of("https://example.com/a.png", mime_type="image/png"),
            {
                "type": "input_image",
                "detail": "auto",
                "image_url": "https://example.com/a.png",
            },
        ),
    ),
)
def test_documents_and_images_are_delivered_by_kind(part: Any, expected: Any) -> None:
    params = _params(ModelInput.of(MultimodalContent.of(part)))

    content: Any = params[0]["content"][0]
    for key, value in expected.items():
        assert content[key] == value


@pytest.mark.parametrize(
    "part",
    (
        ResourceContent.of(b"abc", mime_type="audio/wav"),
        ResourceReference.of("https://example.com/a.mp3", mime_type="audio/mpeg"),
    ),
)
def test_unsupported_input_media_is_rejected(part: Any) -> None:
    with pytest.raises(ValueError, match="Unsupported media"):
        _params(ModelInput.of(MultimodalContent.of(part)))


def test_output_citations_are_replayed_as_annotations() -> None:
    params = _params(
        ModelOutput.of(
            MultimodalContent.of(
                TextContent.of(
                    "see source",
                    meta={
                        "citations": [
                            {
                                "kind": "url_citation",
                                "url": "https://example.com",
                                "title": "Example",
                                "start": 0,
                                "end": 3,
                            }
                        ]
                    },
                )
            )
        )
    )

    assert params[0]["content"][0]["annotations"] == [
        {
            "type": "url_citation",
            "url": "https://example.com",
            "title": "Example",
            "start_index": 0,
            "end_index": 3,
        }
    ]


def test_invalid_citation_metadata_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid citation"):
        _params(
            ModelOutput.of(
                MultimodalContent.of(TextContent.of("x", meta={"citations": [{"kind": "other"}]}))
            )
        )


@pytest.mark.asyncio
async def test_stream_error_event_with_rate_limit_code_is_a_rate_limit() -> None:
    with pytest.raises(ModelRateLimit):
        await _run(
            [
                ResponseErrorEvent.model_construct(
                    type="error",
                    code="rate_limit_exceeded",
                    message="Rate limit reached",
                )
            ]
        )


@pytest.mark.asyncio
async def test_failed_response_with_rate_limit_code_is_a_rate_limit() -> None:
    with pytest.raises(ModelRateLimit):
        await _run(
            [
                _terminal(
                    "response.failed",
                    error=SimpleNamespace(code="rate_limit_exceeded", message="slow down"),
                )
            ]
        )


@pytest.mark.asyncio
async def test_failed_response_with_exhausted_quota_is_a_quota_failure() -> None:
    with pytest.raises(ModelQuotaLimit):
        await _run(
            [
                _terminal(
                    "response.failed",
                    error=SimpleNamespace(code="insufficient_quota", message="no credit"),
                )
            ]
        )


@pytest.mark.asyncio
async def test_stream_without_terminal_event_is_a_failure() -> None:
    with pytest.raises(ModelOutputFailed, match="terminal"):
        await _run([_text_delta("partial")])


@pytest.mark.asyncio
async def test_reasoning_summary_parts_are_separated() -> None:
    chunks = await _run(
        [
            ResponseReasoningSummaryPartAddedEvent.model_construct(
                type="response.reasoning_summary_part.added", summary_index=0
            ),
            _summary_delta("first"),
            ResponseReasoningSummaryPartAddedEvent.model_construct(
                type="response.reasoning_summary_part.added", summary_index=1
            ),
            _summary_delta("second"),
            _item_done(
                ResponseReasoningItem(
                    type="reasoning",
                    id="rs-1",
                    summary=[],
                    encrypted_content="enc",
                )
            ),
            _terminal(),
        ]
    )

    blocks = ModelReasoning.blocks([c for c in chunks if isinstance(c, ModelReasoningChunk)])
    assert len(blocks) == 1
    assert blocks[0].reasoning.to_str() == "first\n\nsecond"


def test_reasoning_without_summary_is_replayed_with_empty_summary() -> None:
    params = _params(
        ModelInput.of(MultimodalContent.of("go")),
        ModelOutput.of(
            ModelReasoning.of(
                TextContent.empty,
                meta={"kind": "reasoning", "id": "rs-1", "encrypted": "enc"},
            ),
            MultimodalContent.of("answer"),
        ),
    )

    reasoning = [param for param in params if param["type"] == "reasoning"]
    assert [param["summary"] for param in reasoning] == [[]]
