from collections.abc import AsyncIterator, Sequence
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from haiway import State, ctx
from ollama import ChatResponse, Message, ResponseError

from draive.embedding import TextEmbedding
from draive.models import (
    GenerativeModel,
    ModelInput,
    ModelOutput,
    ModelOutputFailed,
    ModelOutputLimit,
    ModelRateLimit,
    ModelReasoningChunk,
    ModelToolRequest,
    ModelToolResponse,
    ModelTools,
)
from draive.multimodal import MultimodalContent, TextContent
from draive.ollama import Ollama
from draive.ollama.api import OllamaAPI
from draive.ollama.chat import OllamaChat, _context_messages
from draive.ollama.config import OllamaChatConfig, OllamaEmbeddingConfig
from draive.ollama.embedding import OllamaEmbedding


class _Doc(State):
    text: str


def _response(
    *,
    content: str = "",
    thinking: str | None = None,
    tool_calls: Sequence[Message.ToolCall] | None = None,
    done: bool = False,
    done_reason: str | None = None,
    prompt_eval_count: int | None = None,
    eval_count: int | None = None,
) -> ChatResponse:
    return ChatResponse(
        model="test",
        done=done,
        done_reason=done_reason,
        prompt_eval_count=prompt_eval_count,
        eval_count=eval_count,
        message=Message(
            role="assistant",
            content=content,
            thinking=thinking,
            tool_calls=tool_calls,
        ),
    )


async def _run(
    items: Sequence[ChatResponse] = (),
    *,
    stream_error: Exception | None = None,
    request_error: Exception | None = None,
    closed: list[bool] | None = None,
) -> list[Any]:
    async def _stream() -> AsyncIterator[ChatResponse]:
        try:
            for item in items:
                yield item

            if stream_error is not None:
                raise stream_error

        finally:
            if closed is not None:
                closed.append(True)

    async def _chat(**_: Any) -> AsyncIterator[ChatResponse]:
        if request_error is not None:
            raise request_error

        return _stream()

    model = object.__new__(OllamaChat)
    model._client = SimpleNamespace(chat=_chat)  # pyright: ignore[reportAttributeAccessIssue]
    async with ctx.scope("test"):
        return [
            chunk
            async for chunk in model.completion(
                instructions="",
                tools=ModelTools.none,
                context=(),
                output="text",
                config=OllamaChatConfig(model="test"),
            )
        ]


@pytest.mark.asyncio
async def test_empty_stream_yields_nothing() -> None:
    assert not await _run([_response(done=True, done_reason="stop")])


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["load", "unload", "unexpected"])
async def test_other_done_reasons_are_tolerated(reason: str) -> None:
    chunks = await _run([_response(content="x", done=True, done_reason=reason)])
    assert [chunk.text for chunk in chunks] == ["x"]


@pytest.mark.asyncio
async def test_thinking_is_emitted_as_reasoning_chunk() -> None:
    chunks = await _run(
        [_response(thinking="hmm"), _response(content="answer", done=True, done_reason="stop")]
    )
    assert isinstance(chunks[0], ModelReasoningChunk)
    assert chunks[0].reasoning_chunk.text == "hmm"
    assert chunks[1].text == "answer"


@pytest.mark.asyncio
async def test_parallel_tool_calls_get_distinct_identifiers() -> None:
    call = Message.ToolCall(function=Message.ToolCall.Function(name="t", arguments={"a": 1}))
    chunks = await _run([_response(tool_calls=[call, call], done=True, done_reason="stop")])
    assert all(isinstance(chunk, ModelToolRequest) for chunk in chunks)
    assert len({chunk.identifier for chunk in chunks}) == 2
    assert [chunk.arguments for chunk in chunks] == [{"a": 1}, {"a": 1}]


@pytest.mark.asyncio
async def test_length_done_reason_raises_and_releases_stream() -> None:
    closed: list[bool] = []
    with pytest.raises(ModelOutputLimit):
        await _run([_response(done=True, done_reason="length")], closed=closed)

    assert closed == [True]


@pytest.mark.asyncio
async def test_rate_limit_status_is_translated() -> None:
    with pytest.raises(ModelRateLimit):
        await _run(request_error=ResponseError("busy", 429))


@pytest.mark.asyncio
async def test_rate_limit_during_streaming_is_translated_and_releases_stream() -> None:
    closed: list[bool] = []
    with pytest.raises(ModelRateLimit):
        await _run(
            [_response(content="a")],
            stream_error=ResponseError("busy", 429),
            closed=closed,
        )

    assert closed == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        ResponseError("down", 503),
        httpx.ConnectError("refused"),
        httpx.ReadTimeout("slow"),
        TimeoutError(),
    ],
)
async def test_other_errors_are_typed_with_a_reason(error: Exception) -> None:
    with pytest.raises(ModelOutputFailed) as exc_info:
        await _run(request_error=error)

    assert exc_info.value.reason  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.asyncio
async def test_early_exit_releases_stream() -> None:
    closed: list[bool] = []

    async def _stream() -> AsyncIterator[ChatResponse]:
        try:
            yield _response(content="a")
            yield _response(content="b")

        finally:
            closed.append(True)

    async def _chat(**_: Any) -> AsyncIterator[ChatResponse]:
        return _stream()

    model = object.__new__(OllamaChat)
    model._client = SimpleNamespace(chat=_chat)  # pyright: ignore[reportAttributeAccessIssue]
    async with ctx.scope("test"):
        iterator = model.completion(
            instructions="",
            tools=ModelTools.none,
            context=(),
            output="text",
            config=OllamaChatConfig(model="test"),
        )
        async for _ in iterator:
            break

        await iterator.aclose()  # pyright: ignore[reportAttributeAccessIssue]

    assert closed == [True]


def test_tool_responses_directly_follow_the_assistant_tool_calls() -> None:
    messages = list(
        _context_messages(
            instructions="",
            context=(
                ModelOutput.of(ModelToolRequest.of("c", tool="t", arguments={})),
                ModelInput.of(
                    MultimodalContent.of(TextContent.of("next")),
                    ModelToolResponse.of(
                        "c",
                        tool="t",
                        content=MultimodalContent.of(TextContent.of("result")),
                    ),
                ),
            ),
        )
    )
    assert [(m.role, m.content) for m in messages] == [
        ("assistant", ""),
        ("tool", "result"),
        ("user", "next"),
    ]


def test_empty_assistant_output_is_skipped() -> None:
    assert not list(
        _context_messages(
            instructions="",
            context=(ModelOutput.of(MultimodalContent.empty),),
        )
    )


def test_assistant_reasoning_without_content_is_kept() -> None:
    from draive.models import ModelReasoning

    messages = list(
        _context_messages(
            instructions="",
            context=(
                ModelOutput.of(
                    ModelReasoning.of(MultimodalContent.of(TextContent.of("thought"))),
                ),
            ),
        )
    )
    assert [(m.role, m.thinking) for m in messages] == [("assistant", "thought")]


# api, client and embedding


def test_server_url_falls_back_to_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_SERVER_URL", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "http://host-env:11434")
    assert OllamaAPI()._server_url == "http://host-env:11434"  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setenv("OLLAMA_SERVER_URL", "http://url-env:11434")
    assert OllamaAPI()._server_url == "http://url-env:11434"  # pyright: ignore[reportPrivateUsage]
    assert OllamaAPI("http://explicit")._server_url == "http://explicit"  # pyright: ignore[reportPrivateUsage]


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
    client = Ollama(server_url="http://localhost:1", features=features)
    async with client as state:
        assert {type(element) for element in state} == expected
        assert hasattr(client, "_client")

    assert not hasattr(client, "_client")


def _embedding_model(embed: Any) -> OllamaEmbedding:
    model = object.__new__(OllamaEmbedding)
    model._client = SimpleNamespace(embed=embed)  # pyright: ignore[reportAttributeAccessIssue]
    return model


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrent", [True, False])
async def test_embedding_batches_keep_request_order(concurrent: bool) -> None:
    calls: list[Sequence[str]] = []

    async def _embed(**kwargs: Any) -> Any:
        texts: Sequence[str] = kwargs["input"]
        calls.append(texts)
        return SimpleNamespace(embeddings=[[float(len(text))] for text in texts])

    async with ctx.scope("test"):
        result = await _embedding_model(_embed).create_texts_embedding(
            ["a", "bb", "ccc"],
            config=OllamaEmbeddingConfig(model="e", batch_size=2, concurrent=concurrent),
        )

    assert calls == [["a", "bb"], ["ccc"]]
    assert [item.vector for item in result] == [(1.0,), (2.0,), (3.0,)]


@pytest.mark.asyncio
async def test_embedding_empty_input_makes_no_request() -> None:
    async def _embed(**_: Any) -> Any:
        raise AssertionError("unexpected request")

    async with ctx.scope("test"):
        assert not await _embedding_model(_embed).create_texts_embedding(
            [],
            config=OllamaEmbeddingConfig(model="e"),
        )


@pytest.mark.asyncio
async def test_embedding_uses_attribute_selector() -> None:
    captured: list[Sequence[str]] = []

    async def _embed(**kwargs: Any) -> Any:
        captured.append(kwargs["input"])
        return SimpleNamespace(embeddings=[[0.5] for _ in kwargs["input"]])

    async with ctx.scope("test"):
        result = await _embedding_model(_embed).create_texts_embedding(
            [_Doc(text="x")],
            attribute=lambda value: value.text,
            config=OllamaEmbeddingConfig(model="e"),
        )

    assert captured == [["x"]]
    assert len(result) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrent", [True, False])
async def test_embedding_errors_are_translated(concurrent: bool) -> None:
    async def _limited(**_: Any) -> Any:
        raise ResponseError("busy", 429)

    async def _broken(**_: Any) -> Any:
        raise httpx.ConnectError("refused")

    config = OllamaEmbeddingConfig(model="e", concurrent=concurrent)
    async with ctx.scope("test"):
        with pytest.raises(ModelRateLimit):
            await _embedding_model(_limited).create_texts_embedding(["a"], config=config)

        with pytest.raises(ModelOutputFailed):
            await _embedding_model(_broken).create_texts_embedding(["a"], config=config)
