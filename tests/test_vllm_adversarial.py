from collections.abc import AsyncIterator, Sequence
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
from haiway import Missing, State, ctx
from openai._utils import maybe_transform
from openai.types.chat.completion_create_params import CompletionCreateParamsStreaming

from draive.embedding import Embedded, TextEmbedding
from draive.models import (
    GenerativeModel,
    ModelInput,
    ModelOutput,
    ModelOutputFailed,
    ModelOutputInvalid,
    ModelOutputLimit,
    ModelQuotaLimit,
    ModelRateLimit,
    ModelToolRequest,
    ModelToolResponse,
    ModelTools,
)
from draive.multimodal import MultimodalContent, TextContent
from draive.vllm import VLLM
from draive.vllm.api import VLLMAPI, _provider_label
from draive.vllm.config import VLLMChatConfig, VLLMEmbeddingConfig
from draive.vllm.embedding import VLLMEmbedding
from draive.vllm.messages import VLLMMessages, _context_messages


class _Doc(State):
    text: str


class _Stream:
    def __init__(self, items: Sequence[Any]) -> None:
        self._items = items
        self.closed: bool = False

    def __aiter__(self) -> AsyncIterator[Any]:
        async def _iterate() -> AsyncIterator[Any]:
            for item in self._items:
                yield item

        return _iterate()

    async def close(self) -> None:
        self.closed = True


def _chunk(
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
                delta=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ],
    )


def _call(
    index: int,
    *,
    identifier: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> Any:
    return SimpleNamespace(
        index=index,
        id=identifier,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _model(
    chunks: Sequence[Any] = (),
    *,
    error: Exception | None = None,
) -> tuple[VLLMMessages, _Stream]:
    stream = _Stream(chunks)

    async def _create(**_: Any) -> _Stream:
        if error is not None:
            raise error

        return stream

    model = object.__new__(VLLMMessages)
    model._base_url = "http://localhost"
    model._client = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
    )
    return model, stream


async def _collect(model: VLLMMessages) -> list[Any]:
    async with ctx.scope("test"):
        return [
            chunk
            async for chunk in model.completion(
                instructions="",
                tools=ModelTools.none,
                context=(),
                output="text",
                config=VLLMChatConfig(model="test"),
            )
        ]


def _tool_stream(arguments: str | None) -> Sequence[Any]:
    return (
        _chunk(tool_calls=[_call(0, identifier="c", name="t", arguments=arguments)]),
        _chunk(finish_reason="tool_calls"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", ["[1, 2]", '"text"', "5", "true"])
async def test_non_object_tool_arguments_are_invalid_output(arguments: str) -> None:
    model, stream = _model(_tool_stream(arguments))
    with pytest.raises(ModelOutputInvalid):
        await _collect(model)

    assert stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", [None, "", "null"])
async def test_missing_or_null_tool_arguments_are_empty(arguments: str | None) -> None:
    model, _ = _model(_tool_stream(arguments))
    (request,) = await _collect(model)
    assert isinstance(request, ModelToolRequest)
    assert request.arguments == {}


@pytest.mark.asyncio
async def test_malformed_tool_arguments_are_invalid_output() -> None:
    model, _ = _model(_tool_stream('{"a":'))
    with pytest.raises(ModelOutputInvalid):
        await _collect(model)


@pytest.mark.asyncio
async def test_tool_call_without_identifier_is_invalid_output() -> None:
    model, _ = _model(
        (
            _chunk(tool_calls=[_call(0, name="t", arguments="{}")]),
            _chunk(finish_reason="tool_calls"),
        )
    )
    with pytest.raises(ModelOutputInvalid):
        await _collect(model)


@pytest.mark.asyncio
async def test_interleaved_parallel_tool_calls_are_ordered_by_index() -> None:
    model, _ = _model(
        (
            _chunk(
                tool_calls=[
                    _call(1, identifier="b", name="y", arguments="{"),
                    _call(0, identifier="a", name="x", arguments='{"p"'),
                ]
            ),
            _chunk(tool_calls=[_call(0, arguments=":1}"), _call(1, arguments="}")]),
            _chunk(finish_reason="tool_calls"),
        )
    )
    requests = await _collect(model)
    assert [(r.identifier, r.tool, r.arguments) for r in requests] == [
        ("a", "x", {"p": 1}),
        ("b", "y", {}),
    ]


@pytest.mark.asyncio
async def test_empty_deltas_and_usage_only_chunks_are_ignored() -> None:
    model, stream = _model(
        (
            _chunk(content=""),
            _chunk(content=None),
            SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4),
                choices=[],
            ),
            _chunk(content="ok", finish_reason="stop"),
        )
    )
    chunks = await _collect(model)
    assert [c.text for c in chunks] == ["ok"]
    assert stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["content_filter", "unknown"])
async def test_unsupported_finish_reason_fails(reason: str) -> None:
    model, stream = _model((_chunk(content="a"), _chunk(finish_reason=reason)))
    with pytest.raises(ModelOutputFailed, match=reason):
        await _collect(model)

    assert stream.closed


@pytest.mark.asyncio
async def test_length_finish_reason_raises_output_limit() -> None:
    model, _ = _model((_chunk(content="a"), _chunk(finish_reason="length")))
    with pytest.raises(ModelOutputLimit):
        await _collect(model)


@pytest.mark.asyncio
async def test_early_exit_releases_stream() -> None:
    model, stream = _model((_chunk(content="a"), _chunk(content="b"), _chunk(content="c")))
    async with ctx.scope("test"):
        iterator = model.completion(
            instructions="",
            tools=ModelTools.none,
            context=(),
            output="text",
            config=VLLMChatConfig(model="test"),
        )
        assert [chunk async for chunk in _first(iterator)] == ["a"]
        await iterator.aclose()  # pyright: ignore[reportAttributeAccessIssue]

    assert stream.closed


async def _first(iterator: AsyncIterator[Any]) -> AsyncIterator[str]:
    async for chunk in iterator:
        yield chunk.text
        break


_REQUEST = httpx.Request("POST", "http://localhost/v1/chat/completions")


@pytest.mark.asyncio
async def test_rate_limit_keeps_retry_after() -> None:
    model, _ = _model(
        error=openai.RateLimitError(
            "limited",
            response=httpx.Response(429, request=_REQUEST, headers={"Retry-After": "7"}),
            body=None,
        )
    )
    with pytest.raises(ModelRateLimit) as exc_info:
        await _collect(model)

    assert exc_info.value.retry_after == 7


@pytest.mark.asyncio
async def test_exhausted_quota_is_not_retryable() -> None:
    model, _ = _model(
        error=openai.RateLimitError(
            "quota",
            response=httpx.Response(429, request=_REQUEST),
            body={"code": "insufficient_quota"},
        )
    )
    with pytest.raises(ModelQuotaLimit):
        await _collect(model)


@pytest.mark.asyncio
async def test_rate_limit_during_streaming_is_translated() -> None:
    class _Failing(_Stream):
        def __aiter__(self) -> AsyncIterator[Any]:
            async def _iterate() -> AsyncIterator[Any]:
                yield _chunk(content="a")
                raise openai.RateLimitError(
                    "limited",
                    response=httpx.Response(429, request=_REQUEST, headers={"Retry-After": "2"}),
                    body=None,
                )

            return _iterate()

    stream = _Failing(())

    async def _create(**_: Any) -> _Stream:
        return stream

    model = object.__new__(VLLMMessages)
    model._base_url = "http://localhost"
    model._client = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
    )
    with pytest.raises(ModelRateLimit):
        await _collect(model)

    assert stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        openai.APITimeoutError(request=_REQUEST),
        openai.APIConnectionError(request=_REQUEST),
        openai.InternalServerError(
            "boom", response=httpx.Response(503, request=_REQUEST), body=None
        ),
    ],
)
async def test_transport_and_server_errors_are_typed(error: Exception) -> None:
    model, _ = _model(error=error)
    with pytest.raises(ModelOutputFailed):
        await _collect(model)


@pytest.mark.asyncio
async def test_unsupported_output_modality_is_not_an_invalid_input() -> None:
    model, _ = _model()
    with pytest.raises(NotImplementedError):
        async with ctx.scope("test"):
            async for _ in model.completion(
                instructions="",
                tools=ModelTools.none,
                context=(),
                output="image",
                config=VLLMChatConfig(model="test"),
            ):
                pass


@pytest.mark.asyncio
async def test_request_options_are_forwarded() -> None:
    captured: dict[str, Any] = {}

    async def _create(**kwargs: Any) -> _Stream:
        captured.update(kwargs)
        return _Stream(())

    model = object.__new__(VLLMMessages)
    model._base_url = "http://localhost"
    model._client = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
    )
    async with ctx.scope("test"):
        async for _ in model.completion(
            instructions="sys",
            tools=ModelTools.none,
            context=(),
            output="json",
            config=VLLMChatConfig(model="test", stop_sequences=("x",), max_output_tokens=5),
        ):
            pass

    assert captured["response_format"] == {"type": "json_object"}
    assert captured["stop"] == ["x"]
    assert captured["max_tokens"] == 5
    assert captured["stream_options"] == {"include_usage": True}


def test_tool_message_content_survives_request_serialization_retries() -> None:
    messages = list(
        _context_messages(
            instructions="",
            context=(
                ModelInput.of(
                    ModelToolResponse.of(
                        "c",
                        tool="t",
                        content=MultimodalContent.of(TextContent.of("result")),
                    ),
                ),
                ModelInput.of(MultimodalContent.of(TextContent.of("question"))),
            ),
            vision_details=Missing,
        )
    )
    # the sdk serializes the request on each attempt, retries included
    first = maybe_transform({"messages": messages, "model": "m"}, CompletionCreateParamsStreaming)
    second = maybe_transform({"messages": messages, "model": "m"}, CompletionCreateParamsStreaming)
    assert first == second
    assert second["messages"][0]["content"] == [{"type": "text", "text": "result"}]
    assert second["messages"][1]["content"] == [{"type": "text", "text": "question"}]


def test_empty_tool_response_uses_empty_string_content() -> None:
    (message,) = _context_messages(
        instructions="",
        context=(
            ModelInput.of(
                ModelToolResponse.of("c", tool="t", content=MultimodalContent.empty),
            ),
        ),
        vision_details=Missing,
    )
    assert message["content"] == ""


def test_empty_assistant_output_is_skipped() -> None:
    assert not list(
        _context_messages(
            instructions="",
            context=(ModelOutput.of(MultimodalContent.empty),),
            vision_details=Missing,
        )
    )


def test_assistant_tool_calls_precede_their_results() -> None:
    messages = list(
        _context_messages(
            instructions="",
            context=(
                ModelOutput.of(ModelToolRequest.of("c", tool="t", arguments={"a": 1})),
                ModelInput.of(
                    MultimodalContent.of(TextContent.of("next")),
                    ModelToolResponse.of(
                        "c",
                        tool="t",
                        content=MultimodalContent.of(TextContent.of("r")),
                    ),
                ),
            ),
            vision_details=Missing,
        )
    )
    assert [m["role"] for m in messages] == ["assistant", "tool", "user"]
    assert "content" not in messages[0]
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'


# api, client and embedding


@pytest.mark.parametrize(
    ("base_url", "label"),
    [
        ("http://user:secret@host:8000/v1?token=abc", "vllm@http://host/v1"),
        ("https://example.com", "vllm@https://example.com"),
    ],
)
def test_provider_label_hides_credentials(base_url: str, label: str) -> None:
    assert _provider_label(base_url) == label


def test_base_url_falls_back_to_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://openai-env/v1")
    assert VLLMAPI()._base_url == "http://openai-env/v1"  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm-env/v1")
    assert VLLMAPI()._base_url == "http://vllm-env/v1"  # pyright: ignore[reportPrivateUsage]
    assert VLLMAPI("http://explicit/v1")._base_url == "http://explicit/v1"  # pyright: ignore[reportPrivateUsage]


def test_prepare_client_never_uses_openai_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    api = VLLMAPI("http://localhost:1/v1", timeout=3, project="p")  # pyright: ignore[reportArgumentType]
    client = api._prepare_client()  # pyright: ignore[reportPrivateUsage]
    assert client.api_key == "vllm"
    assert client.organization == "vllm"
    assert client.project == "p"

    assert VLLMAPI("http://localhost:1/v1", api_key="k")._prepare_client().api_key == "k"  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("features", "expected"),
    [
        (None, {GenerativeModel, TextEmbedding}),
        ((GenerativeModel,), {GenerativeModel}),
        ((TextEmbedding,), {TextEmbedding}),
    ],
)
async def test_client_provides_requested_features(
    features: Sequence[Any] | None,
    expected: set[type],
) -> None:
    client = VLLM(base_url="http://localhost:1/v1", features=features)
    async with client as state:
        assert {type(element) for element in state} == expected
        assert hasattr(client, "_client")

    assert not hasattr(client, "_client")


def _embedding_model(
    create: Any,
) -> VLLMEmbedding:
    model = object.__new__(VLLMEmbedding)
    model._base_url = "http://localhost"
    model._client = SimpleNamespace(embeddings=SimpleNamespace(create=create))  # pyright: ignore[reportAttributeAccessIssue]
    return model


@pytest.mark.asyncio
async def test_embedding_batches_and_reorders_by_index() -> None:
    calls: list[Sequence[str]] = []

    async def _create(**kwargs: Any) -> Any:
        texts: Sequence[str] = kwargs["input"]
        calls.append(texts)
        # entries are returned in reverse, only the index pairs them with the input
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=[float(len(text))])
                for index, text in reversed(list(enumerate(texts)))
            ]
        )

    async with ctx.scope("test"):
        result = await _embedding_model(_create).create_texts_embedding(
            ["a", "bb", "ccc"],
            config=VLLMEmbeddingConfig(model="e", batch_size=2),
        )

    assert calls == [["a", "bb"], ["ccc"]]
    assert [item.vector for item in result] == [(1.0,), (2.0,), (3.0,)]
    assert all(isinstance(item, Embedded) for item in result)


@pytest.mark.asyncio
async def test_embedding_empty_input_makes_no_request() -> None:
    async def _create(**_: Any) -> Any:
        raise AssertionError("unexpected request")

    async with ctx.scope("test"):
        assert not await _embedding_model(_create).create_texts_embedding(
            [],
            config=VLLMEmbeddingConfig(model="e"),
        )


@pytest.mark.asyncio
async def test_embedding_uses_attribute_selector_and_dimensions() -> None:
    captured: list[dict[str, Any]] = []

    async def _create(**kwargs: Any) -> Any:
        captured.append(kwargs)
        return SimpleNamespace(
            data=[SimpleNamespace(index=i, embedding=[0.5]) for i, _ in enumerate(kwargs["input"])]
        )

    async with ctx.scope("test"):
        result = await _embedding_model(_create).create_texts_embedding(
            [_Doc(text="x")],
            attribute=lambda value: value.text,
            config=VLLMEmbeddingConfig(model="e", dimensions=8),
        )

    assert captured[0]["input"] == ["x"]
    assert captured[0]["dimensions"] == 8
    assert len(result) == 1


@pytest.mark.asyncio
async def test_embedding_errors_are_translated() -> None:
    async def _limited(**_: Any) -> Any:
        raise openai.RateLimitError(
            "limited",
            response=httpx.Response(429, request=_REQUEST, headers={"Retry-After": "3"}),
            body=None,
        )

    async def _broken(**_: Any) -> Any:
        raise openai.APIConnectionError(request=_REQUEST)

    async with ctx.scope("test"):
        with pytest.raises(ModelRateLimit):
            await _embedding_model(_limited).create_texts_embedding(
                ["a"], config=VLLMEmbeddingConfig(model="e")
            )

        with pytest.raises(ModelOutputFailed):
            await _embedding_model(_broken).create_texts_embedding(
                ["a"], config=VLLMEmbeddingConfig(model="e")
            )
