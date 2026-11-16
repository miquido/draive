from collections.abc import AsyncIterable, Sequence
from typing import Any

import pytest
from haiway import ctx

from draive.embedding import Embedded, TextEmbedding
from draive.evaluators import (
    forbidden_keywords_context_evaluator,
    forbidden_keywords_evaluator,
    required_keywords_context_evaluator,
    required_keywords_evaluator,
    similarity_context_evaluator,
    similarity_evaluator,
    text_vector_similarity_evaluator,
)
from draive.models import GenerativeModel, ModelOutput, ModelOutputChunk
from draive.multimodal import MultimodalContent, TextContent


async def _stream(*chunks: ModelOutputChunk) -> AsyncIterable[ModelOutputChunk]:
    for chunk in chunks:
        yield chunk


def _judge(reply: str, captured: list[str] | None = None) -> GenerativeModel:
    def generating(
        *,
        instructions: str,
        context: Any,
        **extra: Any,
    ) -> AsyncIterable[ModelOutputChunk]:
        if captured is not None:
            captured.append(instructions)

        return _stream(TextContent.of(reply))

    return GenerativeModel(generating=generating)


def _embedding(vectors: dict[str, Sequence[float]]) -> TextEmbedding:
    async def embed(values: Sequence[Any], attribute: Any = None, **extra: Any) -> Any:
        return [Embedded(value=value, vector=vectors[value]) for value in values]

    return TextEmbedding(embedding=embed)


_OUTPUT = (ModelOutput.of(MultimodalContent.of("answer")),)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply", "score"),
    (
        ("<justification>j</justification><rating>GOOD</rating>", 0.6),
        ("<rating> Perfect \n</rating>", 1.0),
        ("<rating>none</rating>", 0.0),
    ),
)
async def test_similarity_evaluator_parses_judge_rating(reply: str, score: float) -> None:
    async with ctx.scope("test", _judge(reply)):
        result = await similarity_evaluator("a", reference="b")

    assert result.score == pytest.approx(score)
    assert "error" not in result.meta


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply",
    (
        "no tags at all",
        "<rating>great</rating>",
        "<rating>0.5</rating>",
        "<rating>good</rating><rating>poor</rating>",
    ),
)
async def test_similarity_evaluator_invalid_judge_output_is_failed_result(reply: str) -> None:
    async with ctx.scope("test", _judge(reply)):
        result = await similarity_evaluator("a", reference="b")

    assert result.score == 0.0
    assert not result.passed
    assert "error" in result.meta


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("evaluated", "reference", "comment"),
    (
        ("   ", "b", "Input was empty!"),
        ("a", "", "Reference was empty!"),
        ("a", " \n ", "Reference was empty!"),
    ),
)
async def test_similarity_evaluator_empty_inputs_skip_the_judge(
    evaluated: str,
    reference: str,
    comment: str,
) -> None:
    captured: list[str] = []
    async with ctx.scope("test", _judge("<rating>perfect</rating>", captured)):
        result = await similarity_evaluator(evaluated, reference=reference)

    assert result.score == 0.0
    assert result.meta["comment"] == comment
    assert captured == []


@pytest.mark.asyncio
async def test_similarity_evaluator_guidelines_with_braces_are_passed_through() -> None:
    captured: list[str] = []
    async with ctx.scope("test", _judge("<rating>good</rating>", captured)):
        await similarity_evaluator("a", reference="b", guidelines="use {braces} {x}")

    assert "use {braces} {x}" in captured[0]


@pytest.mark.asyncio
async def test_similarity_context_evaluator_chooses_instruction_by_reference() -> None:
    captured: list[str] = []
    async with ctx.scope("test", _judge("<rating>good</rating>", captured)):
        without_reference = await similarity_context_evaluator(_OUTPUT)
        blank_reference = await similarity_context_evaluator(_OUTPUT, reference="  ")
        with_reference = await similarity_context_evaluator(_OUTPUT, reference="ref")
        empty_context = await similarity_context_evaluator(())

    assert without_reference.score == pytest.approx(0.6)
    assert blank_reference.score == pytest.approx(0.6)
    assert with_reference.score == pytest.approx(0.6)
    assert captured[0] == captured[1] != captured[2]
    assert empty_context.score == 0.0
    assert empty_context.meta["comment"] == "Input context was empty!"
    assert len(captured) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("vectors", "expected"),
    (
        ({"a": [1.0, 0.0], "b": [-1.0, 0.0]}, 0.0),
        ({"a": [0.0, 0.0], "b": [1.0, 0.0]}, 0.0),
        ({"a": [1.0, 0.0], "b": [0.0, 1.0]}, 0.0),
        ({"a": [1.0, 1.0], "b": [2.0, 2.0]}, 1.0),
    ),
)
async def test_text_vector_similarity_is_clipped_to_unit_range(
    vectors: dict[str, Sequence[float]],
    expected: float,
) -> None:
    async with ctx.scope("test", _embedding(vectors)):
        result = await text_vector_similarity_evaluator("a", reference="b")

    assert result.score == pytest.approx(expected)


@pytest.mark.asyncio
@pytest.mark.parametrize(("evaluated", "reference"), (("", "b"), ("a", "")))
async def test_text_vector_similarity_empty_text_scores_zero(
    evaluated: str,
    reference: str,
) -> None:
    async with ctx.scope("test"):
        result = await text_vector_similarity_evaluator(evaluated, reference=reference)

    assert result.score == 0.0


@pytest.mark.asyncio
async def test_text_vector_similarity_identical_vectors_pass_default_threshold() -> None:
    async with ctx.scope("test", _embedding({"a": [1.0, 1.0], "b": [1.0, 1.0]})):
        result = await text_vector_similarity_evaluator("a", reference="b")

    assert result.passed


@pytest.mark.asyncio
async def test_text_vector_similarity_nan_is_not_a_perfect_score() -> None:
    vectors = {"a": [float("nan"), 1.0], "b": [1.0, 0.0]}
    async with ctx.scope("test", _embedding(vectors)):
        result = await text_vector_similarity_evaluator("a", reference="b")

    assert result.score < 1.0


# --- keywords -------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "keywords", "require_all", "expected"),
    (
        ("Alpha beta", ["alpha", "BETA"], True, 1.0),
        ("Alpha beta", ["alpha", "gamma"], True, 0.0),
        ("Alpha beta", ["alpha", "gamma"], False, 0.5),
        ("Alpha beta", ["alpha", "alpha", "gamma"], False, pytest.approx(2 / 3)),
        ("Alpha beta", [], True, 0.0),
        ("", ["alpha"], True, 0.0),
    ),
)
async def test_required_keywords_scores(
    text: str,
    keywords: list[str],
    require_all: bool,
    expected: float,
) -> None:
    async with ctx.scope("test"):
        result = await required_keywords_evaluator(text, keywords=keywords, require_all=require_all)

    assert result.score == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "keywords", "require_none", "expected"),
    (
        ("Alpha beta", ["gamma"], True, 1.0),
        ("Alpha beta", ["ALPHA"], True, 0.0),
        ("Alpha beta", ["alpha", "gamma"], False, 0.5),
        ("", ["alpha"], True, 1.0),
        ("Alpha", [], True, 1.0),
    ),
)
async def test_forbidden_keywords_scores(
    text: str,
    keywords: list[str],
    require_none: bool,
    expected: float,
) -> None:
    async with ctx.scope("test"):
        result = await forbidden_keywords_evaluator(
            text, keywords=keywords, require_none=require_none
        )

    assert result.score == expected


@pytest.mark.asyncio
async def test_keyword_normalization_applies_to_text_and_keywords() -> None:
    async with ctx.scope("test"):
        result = await required_keywords_evaluator(
            "Café",
            keywords=["CAFÉ"],
            normalization=lambda text: text.casefold(),
        )

    assert result.score == 1.0


@pytest.mark.asyncio
async def test_keyword_context_evaluators_inspect_model_context() -> None:
    async with ctx.scope("test"):
        required = await required_keywords_context_evaluator(_OUTPUT, keywords=["answer"])
        forbidden = await forbidden_keywords_context_evaluator(_OUTPUT, keywords=["answer"])
        empty = await required_keywords_context_evaluator((), keywords=["answer"])

    assert required.score == 1.0
    assert forbidden.score == 0.0
    assert empty.score == 0.0


@pytest.mark.asyncio
async def test_text_vector_similarity_keeps_real_differences() -> None:
    async with ctx.scope("test", _embedding({"a": [1.0, 0.0], "b": [1.0, 0.001]})):
        result = await text_vector_similarity_evaluator("a", reference="b")

    assert result.score < 1.0
