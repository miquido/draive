from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest
from cohere import EmbedByTypeResponse
from cohere.errors import TooManyRequestsError
from haiway import State, ctx

from draive.cohere.config import CohereImageEmbeddingConfig, CohereTextEmbeddingConfig
from draive.cohere.embedding import CohereEmbedding, _billed_input_tokens, _image_mime_type  # pyright: ignore[reportPrivateUsage]
from draive.models import ModelOutputFailed, ModelOutputInvalid, ModelRateLimit

_TEXT_CONFIG = CohereTextEmbeddingConfig(model="embed", purpose="search_document", batch_size=2)
_IMAGE_CONFIG = CohereImageEmbeddingConfig(model="embed", batch_size=2)


class _Document(State):
    text: str


def _response(
    vectors: Sequence[Sequence[float]] | None,
    *,
    billed: int | None = None,
) -> EmbedByTypeResponse:
    payload: dict[str, Any] = {
        "id": "embed-1",
        "embeddings": {"float": vectors} if vectors is not None else {},
        "texts": [],
    }
    if billed is not None:
        payload["meta"] = {"billed_units": {"input_tokens": billed}}

    return EmbedByTypeResponse.model_validate(payload)


def _embedding(embed: Any) -> CohereEmbedding:
    embedding = object.__new__(CohereEmbedding)
    embedding._client = SimpleNamespace(embed=embed)  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    return embedding


@pytest.mark.asyncio
async def test_texts_are_batched_and_requested_with_the_configured_purpose() -> None:
    requests: list[dict[str, Any]] = []

    async def embed(**request: Any) -> EmbedByTypeResponse:
        requests.append(request)
        return _response([[float(len(text))] for text in request["texts"]])

    async with ctx.scope("test"):
        result = await _embedding(embed).create_texts_embedding(
            ["a", "bb", "ccc"],
            config=_TEXT_CONFIG,
        )

    assert [request["texts"] for request in requests] == [["a", "bb"], ["ccc"]]
    assert all(request["input_type"] == "search_document" for request in requests)
    assert all(request["embedding_types"] == ["float"] for request in requests)
    assert [(item.value, item.vector) for item in result] == [
        ("a", (1.0,)),
        ("bb", (2.0,)),
        ("ccc", (3.0,)),
    ]


@pytest.mark.asyncio
async def test_attribute_selects_the_embedded_text_of_state_values() -> None:
    sent: list[Sequence[str]] = []

    async def embed(**request: Any) -> EmbedByTypeResponse:
        sent.append(request["texts"])
        return _response([[1.0]])

    document = _Document(text="content")
    async with ctx.scope("test"):
        result = await _embedding(embed).create_texts_embedding(
            [document],
            attribute=lambda value: value.text,
            config=_TEXT_CONFIG,
        )

    assert sent == [["content"]]
    assert result[0].value == document


@pytest.mark.asyncio
async def test_empty_input_sends_no_request() -> None:
    async def embed(**_: Any) -> EmbedByTypeResponse:
        raise AssertionError("no request expected")

    async with ctx.scope("test"):
        assert await _embedding(embed).create_texts_embedding([], config=_TEXT_CONFIG) == ()
        assert await _embedding(embed).create_images_embedding([], config=_IMAGE_CONFIG) == ()


@pytest.mark.asyncio
async def test_state_values_are_embedded_through_their_image_attribute() -> None:
    sent: list[Sequence[str]] = []

    class _Picture(State):
        data: bytes

    async def embed(**request: Any) -> EmbedByTypeResponse:
        sent.append(request["images"])
        assert request["input_type"] == "image"
        return _response([[1.0]])

    async with ctx.scope("test"):
        result = await _embedding(embed).create_images_embedding(
            [_Picture(data=b"\x89PNG\r\n\x1a\nrest")],
            attribute=lambda value: value.data,
            config=_IMAGE_CONFIG,
        )

    assert sent[0][0].startswith("data:image/png;base64,")
    assert len(result) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "vectors",
    [
        pytest.param(None, id="no-float-vectors"),
        pytest.param([], id="empty-vectors"),
        pytest.param([[1.0]], id="fewer-vectors"),
    ],
)
async def test_inconsistent_response_is_an_invalid_output(
    vectors: Sequence[Sequence[float]] | None,
) -> None:
    async def embed(**_: Any) -> EmbedByTypeResponse:
        return _response(vectors)

    async with ctx.scope("test"):
        with pytest.raises(ModelOutputInvalid) as caught:
            await _embedding(embed).create_texts_embedding(["a", "b"], config=_TEXT_CONFIG)

    assert caught.value.provider == "cohere"


@pytest.mark.asyncio
async def test_request_failure_is_a_typed_exception() -> None:
    async def embed(**_: Any) -> EmbedByTypeResponse:
        raise RuntimeError("boom")

    async with ctx.scope("test"):
        with pytest.raises(ModelOutputFailed):
            await _embedding(embed).create_texts_embedding(["a"], config=_TEXT_CONFIG)


@pytest.mark.asyncio
async def test_rate_limit_keeps_retry_after() -> None:
    async def embed(**_: Any) -> EmbedByTypeResponse:
        raise TooManyRequestsError(headers={"retry-after": "7"}, body={"message": "slow down"})

    async with ctx.scope("test"):
        with pytest.raises(ModelRateLimit) as caught:
            await _embedding(embed).create_texts_embedding(["a"], config=_TEXT_CONFIG)

    assert caught.value.retry_after == 7


def test_billed_tokens_are_summed_and_absent_when_unreported() -> None:
    assert _billed_input_tokens([_response([[1.0]], billed=3), _response([[1.0]], billed=4)]) == 7
    assert _billed_input_tokens([_response([[1.0]], billed=3), _response([[1.0]])]) == 3
    assert _billed_input_tokens([_response([[1.0]])]) is None


@pytest.mark.parametrize(
    ("data", "mime_type"),
    [
        (b"\x89PNG\r\n\x1a\n", "image/png"),
        (b"GIF87a", "image/gif"),
        (b"GIF89a", "image/gif"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp"),
        (b"RIFF\x00\x00\x00\x00WAVEfmt ", "image/jpeg"),
        (b"\xff\xd8\xff", "image/jpeg"),
        (b"", "image/jpeg"),
    ],
)
def test_image_mime_type_is_recognized_by_its_header(data: bytes, mime_type: str) -> None:
    assert _image_mime_type(data) == mime_type
