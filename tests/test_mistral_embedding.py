from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest
from haiway import State, ctx

from draive.mistral.config import MistralEmbeddingConfig
from draive.mistral.embedding import MistralEmbedding
from draive.models import ModelOutputFailed, ModelOutputInvalid, ModelRateLimit


class _Document(State):
    text: str


def _embedding(create: Any) -> MistralEmbedding:
    embedding = object.__new__(MistralEmbedding)
    embedding._client = SimpleNamespace(embeddings=SimpleNamespace(create_async=create))  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    return embedding


def _entries(vectors: Sequence[Sequence[float] | None], *, indexes: Sequence[int | None]) -> Any:
    return SimpleNamespace(
        data=[
            SimpleNamespace(index=index, embedding=vector)
            for index, vector in zip(vectors and indexes, vectors, strict=True)
        ],
        usage=SimpleNamespace(prompt_tokens=2),
    )


@pytest.mark.asyncio
async def test_inputs_are_split_into_batches_preserving_order() -> None:
    requests: list[Sequence[str]] = []

    async def create(*, model: str, inputs: Sequence[str]) -> Any:
        requests.append(inputs)
        return _entries([[float(len(text))] for text in inputs], indexes=range(len(inputs)))

    async with ctx.scope("test"):
        result = await _embedding(create).create_texts_embedding(
            ["a", "bb", "ccc"],
            config=MistralEmbeddingConfig(model="embed", batch_size=2),
        )

    assert requests == [["a", "bb"], ["ccc"]]
    assert [(item.value, item.vector) for item in result] == [
        ("a", (1.0,)),
        ("bb", (2.0,)),
        ("ccc", (3.0,)),
    ]


@pytest.mark.asyncio
async def test_entries_are_paired_with_inputs_by_their_index() -> None:
    async def create(*, model: str, inputs: Sequence[str]) -> Any:
        return _entries([[1.0], [0.0]], indexes=[1, 0])

    async with ctx.scope("test"):
        result = await _embedding(create).create_texts_embedding(
            ["first", "second"],
            config=MistralEmbeddingConfig(model="embed"),
        )

    assert [item.vector for item in result] == [(0.0,), (1.0,)]


@pytest.mark.asyncio
async def test_attribute_selects_the_embedded_text_of_state_values() -> None:
    sent: list[Sequence[str]] = []

    async def create(*, model: str, inputs: Sequence[str]) -> Any:
        sent.append(inputs)
        return _entries([[1.0]], indexes=[0])

    document = _Document(text="content")
    async with ctx.scope("test"):
        result = await _embedding(create).create_texts_embedding(
            [document],
            attribute=lambda value: value.text,
            config=MistralEmbeddingConfig(model="embed"),
        )

    assert sent == [["content"]]
    assert result[0].value == document


@pytest.mark.asyncio
async def test_empty_input_sends_no_request() -> None:
    async def create(**_: Any) -> Any:
        raise AssertionError("no request expected")

    async with ctx.scope("test"):
        result = await _embedding(create).create_texts_embedding(
            [],
            config=MistralEmbeddingConfig(model="embed"),
        )

    assert result == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "vectors",
    [
        pytest.param([[1.0], None], id="empty-vector"),
        pytest.param([[1.0]], id="fewer-vectors"),
        pytest.param([[1.0], [2.0], [3.0]], id="more-vectors"),
    ],
)
async def test_inconsistent_response_is_an_invalid_output(
    vectors: Sequence[Sequence[float] | None],
) -> None:
    async def create(**_: Any) -> Any:
        return _entries(vectors, indexes=range(len(vectors)))

    async with ctx.scope("test"):
        with pytest.raises(ModelOutputInvalid) as caught:
            await _embedding(create).create_texts_embedding(
                ["a", "b"],
                config=MistralEmbeddingConfig(model="embed"),
            )

    assert caught.value.provider == "mistral"


@pytest.mark.asyncio
async def test_request_failure_is_a_typed_exception() -> None:
    async def create(**_: Any) -> Any:
        raise RuntimeError("boom")

    async with ctx.scope("test"):
        with pytest.raises(ModelOutputFailed):
            await _embedding(create).create_texts_embedding(
                ["a"],
                config=MistralEmbeddingConfig(model="embed"),
            )


@pytest.mark.asyncio
async def test_rate_limit_keeps_retry_after() -> None:
    class _Limited(Exception):
        status_code = 429
        headers = {"Retry-After": "9"}  # noqa: RUF012

    async def create(**_: Any) -> Any:
        raise _Limited()

    async with ctx.scope("test"):
        with pytest.raises(ModelRateLimit) as caught:
            await _embedding(create).create_texts_embedding(
                ["a"],
                config=MistralEmbeddingConfig(model="embed"),
            )

    assert caught.value.retry_after == 9
