from types import SimpleNamespace
from typing import Any

import pytest
from haiway import ctx

from draive.mistral.config import MistralEmbeddingConfig
from draive.mistral.embedding import MistralEmbedding
from draive.models import ModelOutputFailed
from draive.openai.config import OpenAIEmbeddingConfig
from draive.openai.embedding import OpenAIEmbedding


class _Recorder:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> None:
        self.records.append(kwargs)

    def tokens(self) -> list[Any]:
        return [
            record["value"]
            for record in self.records
            if record.get("metric") == "embedding.input_tokens"
        ]


def _openai(create: Any) -> OpenAIEmbedding:
    model = object.__new__(OpenAIEmbedding)
    model._client = SimpleNamespace(embeddings=SimpleNamespace(create=create))  # pyright: ignore[reportAttributeAccessIssue]
    return model


@pytest.mark.asyncio
async def test_openai_embedding_pairs_vectors_and_records_token_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(ctx, "record_info", recorder)

    async def create(*, input: list[str], **_: Any) -> Any:  # noqa: A002
        return SimpleNamespace(
            # the api documents no ordering guarantee, entries are paired by index
            data=[
                SimpleNamespace(index=index, embedding=[float(len(input[index]))])
                for index in reversed(range(len(input)))
            ],
            usage=SimpleNamespace(prompt_tokens=len(input), total_tokens=len(input)),
        )

    async with ctx.scope("test"):
        embedded = await _openai(create).create_texts_embedding(
            ["a", "bbb", "cc"],
            config=OpenAIEmbeddingConfig(batch_size=2),
        )

    assert [(element.value, list(element.vector)) for element in embedded] == [
        ("a", [1.0]),
        ("bbb", [3.0]),
        ("cc", [2.0]),
    ]
    assert sum(recorder.tokens()) == 3  # two requests, two and one inputs


@pytest.mark.asyncio
async def test_openai_embedding_failure_is_translated() -> None:
    async def create(**_: Any) -> Any:
        raise RuntimeError("boom")

    async with ctx.scope("test"):
        with pytest.raises(ModelOutputFailed, match="embedding request failed"):
            await _openai(create).create_texts_embedding(
                ["a"],
                config=OpenAIEmbeddingConfig(),
            )


@pytest.mark.asyncio
async def test_mistral_embedding_records_token_usage_and_translates_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(ctx, "record_info", recorder)
    fail = False

    async def create_async(*, inputs: list[str], **_: Any) -> Any:
        if fail:
            raise RuntimeError("boom")

        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=[float(len(value))])
                for index, value in enumerate(inputs)
            ],
            usage=SimpleNamespace(prompt_tokens=7),
        )

    model = object.__new__(MistralEmbedding)
    model._client = SimpleNamespace(embeddings=SimpleNamespace(create_async=create_async))  # pyright: ignore[reportAttributeAccessIssue]

    async with ctx.scope("test"):
        embedded = await model.create_texts_embedding(
            ["a", "bb"],
            config=MistralEmbeddingConfig(model="mistral-embed"),
        )
        assert [list(element.vector) for element in embedded] == [[1.0], [2.0]]
        assert recorder.tokens() == [7]

        fail = True
        with pytest.raises(ModelOutputFailed):
            await model.create_texts_embedding(
                ["a"],
                config=MistralEmbeddingConfig(model="mistral-embed"),
            )


@pytest.mark.asyncio
async def test_openai_image_generation_records_invocation_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from draive.openai.config import OpenAIImageGenerationConfig
    from draive.openai.images import OpenAIImageGeneration

    recorder = _Recorder()
    monkeypatch.setattr(ctx, "record_info", recorder)

    async def generate(**_: Any) -> Any:
        return SimpleNamespace(
            data=[SimpleNamespace(b64_json="aGVsbG8=", url=None)],
            usage=SimpleNamespace(input_tokens=11, output_tokens=22),
        )

    model = object.__new__(OpenAIImageGeneration)
    model._client = SimpleNamespace(images=SimpleNamespace(generate=generate))  # pyright: ignore[reportAttributeAccessIssue]

    async with ctx.scope("test"):
        image = await model.generate_image(
            instructions="style",
            input="a cat",
            config=OpenAIImageGenerationConfig(size="1024x1024"),
        )

    assert image.mime_type == "image/png"
    assert {
        record["metric"]: record["value"]
        for record in recorder.records
        if record.get("metric") in ("model.input_tokens", "model.output_tokens")
    } == {"model.input_tokens": 11, "model.output_tokens": 22}
    invocations = [
        record for record in recorder.records if "attributes" in record and "metric" not in record
    ]
    assert invocations[0]["attributes"]["image.size"] == "1024x1024"


@pytest.mark.asyncio
async def test_openai_image_generation_failure_is_translated() -> None:
    from draive.openai.config import OpenAIImageGenerationConfig
    from draive.openai.images import OpenAIImageGeneration

    async def generate(**_: Any) -> Any:
        raise RuntimeError("boom")

    model = object.__new__(OpenAIImageGeneration)
    model._client = SimpleNamespace(images=SimpleNamespace(generate=generate))  # pyright: ignore[reportAttributeAccessIssue]

    async with ctx.scope("test"):
        with pytest.raises(ModelOutputFailed, match="image generation request failed"):
            await model.generate_image(
                instructions="",
                input="a cat",
                config=OpenAIImageGenerationConfig(),
            )


def test_cohere_billed_input_tokens_are_summed_when_reported() -> None:
    from draive.cohere.embedding import _billed_input_tokens  # pyright: ignore[reportPrivateUsage]

    def response(tokens: float | None, *, meta: bool = True) -> Any:
        return SimpleNamespace(
            meta=SimpleNamespace(billed_units=SimpleNamespace(input_tokens=tokens))
            if meta
            else None
        )

    assert _billed_input_tokens([response(3.0), response(4.0), response(None)]) == 7
    assert _billed_input_tokens([response(None), response(1.0, meta=False)]) is None
