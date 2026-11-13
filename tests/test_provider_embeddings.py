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
