from collections.abc import AsyncIterable, Iterable, Sequence
from typing import Any

import pytest
from haiway import AttributeRequirement, State, ctx

from draive.embedding import Embedded, TextEmbedding, VectorIndex
from draive.evaluation import EvaluatorResult, EvaluatorSuiteCase, evaluator, evaluator_suite
from draive.generation import ModelGeneration
from draive.helpers import (
    InstructionPreparationAmbiguity,
    VolatileVectorIndex,
    prepare_instructions,
    refine_instructions,
)
from draive.helpers.evaluation_case_generation import generate_case_parameters
from draive.models import (
    GenerativeModel,
    ModelContextElement,
    ModelOutputChunk,
    ModelTools,
)
from draive.multimodal import (
    Template,
    TemplateDeclaration,
    TemplatesRepository,
    TextContent,
)


async def _stream_of(*chunks: ModelOutputChunk) -> AsyncIterable[ModelOutputChunk]:
    for chunk in chunks:
        yield chunk


def _replying_model(
    reply: str,
    captured: dict[str, Any] | None = None,
) -> GenerativeModel:
    def generating(
        *,
        instructions: str,
        tools: ModelTools,
        context: Sequence[ModelContextElement],
        output: Any,
        **extra: Any,
    ) -> AsyncIterable[ModelOutputChunk]:
        if captured is not None:
            captured["instructions"] = instructions
            captured["context"] = context

        return _stream_of(TextContent.of(reply))

    return GenerativeModel(generating=generating)


# --- instruction preparation ----------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_instructions_unwraps_result_tag() -> None:
    async with ctx.scope(
        "test",
        _replying_model("<RESULT_INSTRUCTION>prepared</RESULT_INSTRUCTION>"),
    ):
        assert await prepare_instructions("summarize the content") == "prepared"


@pytest.mark.asyncio
async def test_prepare_instructions_raises_ambiguity_on_questions() -> None:
    async with ctx.scope(
        "test",
        _replying_model("<QUESTIONS>which content?</QUESTIONS>"),
    ):
        with pytest.raises(InstructionPreparationAmbiguity) as error:
            await prepare_instructions("do something")

    assert error.value.questions == "which content?"


@pytest.mark.asyncio
async def test_prepare_instructions_raises_on_unrecognized_result() -> None:
    async with ctx.scope("test", _replying_model("Task not achievable")):
        with pytest.raises(ValueError):
            await prepare_instructions("do something impossible")


@pytest.mark.asyncio
async def test_prepare_instructions_passes_declaration_variables() -> None:
    captured: dict[str, Any] = {}
    declaration = TemplateDeclaration.of(
        "summary",
        description="summarize the topic",
        variables={"topic": "the topic to summarize"},
    )

    async with ctx.scope(
        "test",
        _replying_model("<RESULT_INSTRUCTION>ok</RESULT_INSTRUCTION>", captured),
    ):
        await prepare_instructions(declaration)

    input_text: str = captured["context"][0].content.to_str()
    assert "<USER_TASK>summarize the topic</USER_TASK>" in input_text
    assert "- topic: the topic to summarize" in input_text


@pytest.mark.asyncio
async def test_prepare_instructions_uses_na_without_variables() -> None:
    captured: dict[str, Any] = {}

    async with ctx.scope(
        "test",
        _replying_model("<RESULT_INSTRUCTION>ok</RESULT_INSTRUCTION>", captured),
    ):
        await prepare_instructions("summarize")

    assert "<TASK_VARIABLES>N/A</TASK_VARIABLES>" in captured["context"][0].content.to_str()


# --- evaluation case generation -------------------------------------------------------


class _GeneratedCase(State, serializable=True):
    value: str


@pytest.mark.asyncio
async def test_generate_case_parameters_feeds_results_back_as_examples() -> None:
    calls: list[Sequence[tuple[Any, Any]]] = []

    async def generating(
        generated: type[State],
        /,
        *,
        examples: Iterable[tuple[Any, State]],
        **extra: Any,
    ) -> State:
        calls.append(tuple(examples))
        return _GeneratedCase(value=f"case-{len(calls)}")

    async with ctx.scope("test", ModelGeneration(generating=generating)):
        results = await generate_case_parameters(
            _GeneratedCase,
            count=3,
            examples=(_GeneratedCase(value="seed"),),
        )

    assert tuple(result.value for result in results) == ("case-1", "case-2", "case-3")
    # each generated case joins the examples of the following generation
    assert [len(examples) for examples in calls] == [1, 2, 3]


@pytest.mark.asyncio
async def test_generate_case_parameters_allows_curly_braces_in_guidelines() -> None:
    # schema injection formats the instructions, guidelines braces have to stay escaped
    captured: dict[str, Any] = {}

    async def generating(
        generated: type[State],
        /,
        *,
        instructions: str,
        **extra: Any,
    ) -> State:
        captured["instructions"] = instructions
        return _GeneratedCase(value="generated")

    async with ctx.scope("test", ModelGeneration(generating=generating)):
        results = await generate_case_parameters(
            _GeneratedCase,
            count=1,
            examples=(),
            guidelines="use {topic} placeholders",
        )

    assert len(results) == 1
    assert "use {topic} placeholders" in captured["instructions"]
    # the schema placeholder is still resolved
    assert "{model_schema}" not in captured["instructions"]
    assert '"value"' in captured["instructions"]


# --- volatile vector index ------------------------------------------------------------


class _Document(State):
    text: str
    tag: str


_DOCUMENT_VECTORS: dict[str, Sequence[float]] = {
    "alpha": (1.0, 0.0),
    "beta": (0.0, 1.0),
    "gamma": (0.9, 0.1),
}


def _text_embedding() -> TextEmbedding:
    async def embedding(
        values: Iterable[str],
        /,
        **extra: Any,
    ) -> Sequence[Embedded[str]]:
        return [
            Embedded(value=value, vector=_DOCUMENT_VECTORS.get(value, (1.0, 0.0)))
            for value in values
        ]

    return TextEmbedding(embedding=embedding)


_DOCUMENTS = (
    _Document(text="alpha", tag="x"),
    _Document(text="beta", tag="y"),
    _Document(text="gamma", tag="x"),
)


@pytest.mark.asyncio
async def test_volatile_vector_index_searches_by_similarity() -> None:
    async with ctx.scope("test", VolatileVectorIndex(), _text_embedding()):
        await VectorIndex.index(_Document, attribute=_Document._.text, values=_DOCUMENTS)

        results = await VectorIndex.search(_Document, query="alpha")

    # gamma is closer to alpha than beta is
    assert tuple(result.text for result in results) == ("alpha", "gamma", "beta")


@pytest.mark.asyncio
async def test_volatile_vector_index_respects_limit_without_rerank() -> None:
    async with ctx.scope("test", VolatileVectorIndex(), _text_embedding()):
        await VectorIndex.index(_Document, attribute=_Document._.text, values=_DOCUMENTS)

        results = await VectorIndex.search(_Document, query="alpha", limit=2)

    assert tuple(result.text for result in results) == ("alpha", "gamma")


@pytest.mark.asyncio
async def test_volatile_vector_index_respects_limit_with_rerank() -> None:
    async with ctx.scope("test", VolatileVectorIndex(), _text_embedding()):
        await VectorIndex.index(_Document, attribute=_Document._.text, values=_DOCUMENTS)

        results = await VectorIndex.search(_Document, query="alpha", limit=2, rerank=True)

    assert len(results) == 2


@pytest.mark.asyncio
async def test_volatile_vector_index_lists_all_without_query() -> None:
    async with ctx.scope("test", VolatileVectorIndex(), _text_embedding()):
        await VectorIndex.index(_Document, attribute=_Document._.text, values=_DOCUMENTS)

        assert len(await VectorIndex.search(_Document, query=None)) == 3
        assert len(await VectorIndex.search(_Document, query=None, limit=2)) == 2


@pytest.mark.asyncio
async def test_volatile_vector_index_filters_by_requirements() -> None:
    async with ctx.scope("test", VolatileVectorIndex(), _text_embedding()):
        await VectorIndex.index(_Document, attribute=_Document._.text, values=_DOCUMENTS)

        results = await VectorIndex.search(
            _Document,
            query="alpha",
            requirements=AttributeRequirement[_Document].equal("x", _Document._.tag),
        )

    assert tuple(result.tag for result in results) == ("x", "x")


@pytest.mark.asyncio
async def test_volatile_vector_index_deletes_matching_and_all() -> None:
    async with ctx.scope("test", VolatileVectorIndex(), _text_embedding()):
        await VectorIndex.index(_Document, attribute=_Document._.text, values=_DOCUMENTS)

        await VectorIndex.delete(
            _Document,
            requirements=AttributeRequirement[_Document].equal("x", _Document._.tag),
        )
        remaining = await VectorIndex.search(_Document, query=None)
        assert tuple(result.text for result in remaining) == ("beta",)

        await VectorIndex.delete(_Document)
        assert await VectorIndex.search(_Document, query=None) == []


@pytest.mark.asyncio
async def test_volatile_vector_index_searches_empty_storage() -> None:
    async with ctx.scope("test", VolatileVectorIndex(), _text_embedding()):
        assert await VectorIndex.search(_Document, query="alpha") == []


# --- instruction refinement -----------------------------------------------------------


class _RefinementCase(State, serializable=True):
    value: str


_STRATEGIES_REPLY = """\
<strategy><name>Exemplification</name><approach>add examples</approach></strategy>
<strategy><name>Constraining</name><approach>add constraints</approach></strategy>
"""

_REFINED_INSTRUCTIONS = "REFINED INSTRUCTIONS PASSING"


def _refining_model(calls: list[str]) -> GenerativeModel:
    def generating(
        *,
        instructions: str,
        tools: ModelTools,
        context: Sequence[ModelContextElement],
        output: Any,
        **extra: Any,
    ) -> AsyncIterable[ModelOutputChunk]:
        if "EXACTLY 2 DIFFERENT refinement strategies" in instructions:
            calls.append("strategies")
            return _stream_of(TextContent.of(_STRATEGIES_REPLY))

        calls.append("refinement")
        return _stream_of(TextContent.of(_REFINED_INSTRUCTIONS))

    return GenerativeModel(generating=generating)


@evaluator(name="instructions_passing")
async def _passing_instructions(value: str) -> float:
    # score the instructions currently resolved from the repository, this is what
    # the refinement patches for each explored candidate
    loaded: str = await TemplatesRepository.resolve_str(Template.of("subject"))

    return 1.0 if "PASSING" in loaded else 0.0


@pytest.mark.asyncio
async def test_refine_instructions_returns_best_candidate(tmp_path) -> None:
    storage_path = tmp_path / "cases.json"
    storage_path.write_text("[]")

    async def definition(
        parameters: _RefinementCase,
    ) -> Sequence[EvaluatorResult]:
        return [await _passing_instructions(parameters.value)]

    suite = evaluator_suite(_RefinementCase, storage=storage_path)(definition)
    cases = (
        EvaluatorSuiteCase(identifier="first", parameters=_RefinementCase(value="a")),
        EvaluatorSuiteCase(identifier="second", parameters=_RefinementCase(value="b")),
    )
    calls: list[str] = []

    async with ctx.scope(
        "test",
        _refining_model(calls),
        TemplatesRepository.volatile(subject="ORIGINAL INSTRUCTIONS"),
    ):
        for case in cases:
            await suite.add_case(case.parameters, identifier=case.identifier)

        result = await refine_instructions(
            Template.of("subject"),
            evaluator_suite=suite.prepared(),
            evaluator_cases=cases,
            rounds_limit=1,
        )

    # two complementary strategies explored, each turned into refined instructions
    assert calls == ["strategies", "refinement", "refinement"]
    # a candidate improving over a non scoring root must not be pruned away
    assert result == _REFINED_INSTRUCTIONS


@pytest.mark.asyncio
async def test_refine_instructions_keeps_original_without_improvement(tmp_path) -> None:
    storage_path = tmp_path / "cases.json"
    storage_path.write_text("[]")

    async def definition(
        parameters: _RefinementCase,
    ) -> Sequence[EvaluatorResult]:
        return [await _passing_instructions(parameters.value)]

    suite = evaluator_suite(_RefinementCase, storage=storage_path)(definition)
    cases = (EvaluatorSuiteCase(identifier="only", parameters=_RefinementCase(value="a")),)

    def _stale_model() -> GenerativeModel:
        def generating(
            *,
            instructions: str,
            tools: ModelTools,
            context: Sequence[ModelContextElement],
            output: Any,
            **extra: Any,
        ) -> AsyncIterable[ModelOutputChunk]:
            if "EXACTLY 2 DIFFERENT refinement strategies" in instructions:
                return _stream_of(TextContent.of(_STRATEGIES_REPLY))

            return _stream_of(TextContent.of("STILL FAILING INSTRUCTIONS"))

        return GenerativeModel(generating=generating)

    async with ctx.scope(
        "test",
        _stale_model(),
        TemplatesRepository.volatile(subject="ORIGINAL INSTRUCTIONS PASSING"),
    ):
        await suite.add_case(cases[0].parameters, identifier=cases[0].identifier)

        result = await refine_instructions(
            Template.of("subject"),
            evaluator_suite=suite.prepared(),
            evaluator_cases=cases,
            rounds_limit=1,
        )

    assert result == "ORIGINAL INSTRUCTIONS PASSING"


@pytest.mark.asyncio
async def test_refine_instructions_uses_provided_instructions_content(tmp_path) -> None:
    storage_path = tmp_path / "cases.json"
    storage_path.write_text("[]")

    async def definition(
        parameters: _RefinementCase,
    ) -> Sequence[EvaluatorResult]:
        return [await _passing_instructions(parameters.value)]

    suite = evaluator_suite(_RefinementCase, storage=storage_path)(definition)
    cases = (EvaluatorSuiteCase(identifier="only", parameters=_RefinementCase(value="a")),)
    captured: list[str] = []

    def _capturing_model() -> GenerativeModel:
        def generating(
            *,
            instructions: str,
            tools: ModelTools,
            context: Sequence[ModelContextElement],
            output: Any,
            **extra: Any,
        ) -> AsyncIterable[ModelOutputChunk]:
            if "EXACTLY 2 DIFFERENT refinement strategies" in instructions:
                return _stream_of(TextContent.of(_STRATEGIES_REPLY))

            captured.append(instructions)
            return _stream_of(TextContent.of(_REFINED_INSTRUCTIONS))

        return GenerativeModel(generating=generating)

    async with ctx.scope(
        "test",
        _capturing_model(),
        TemplatesRepository.volatile(subject="STORED INSTRUCTIONS"),
    ):
        await suite.add_case(cases[0].parameters, identifier=cases[0].identifier)

        await refine_instructions(
            Template.of("subject"),
            instructions_content="OVERRIDDEN INSTRUCTIONS",
            evaluator_suite=suite.prepared(),
            evaluator_cases=cases,
            rounds_limit=1,
        )

    # the explicitly provided content is refined instead of the stored template
    assert captured
    assert all("OVERRIDDEN INSTRUCTIONS" in instructions for instructions in captured)


@evaluator(name="instructions_partial")
async def _partial_instructions(value: str) -> float:
    # a mid range score, never reaching the quality threshold
    return 0.5


@pytest.mark.asyncio
async def test_refine_instructions_explores_each_node_once_per_round(tmp_path) -> None:
    # suite performance is a percentage, a partial score must not read as
    # exceptional quality and abort the exploration before it starts
    storage_path = tmp_path / "cases.json"
    storage_path.write_text("[]")

    async def definition(
        parameters: _RefinementCase,
    ) -> Sequence[EvaluatorResult]:
        return [await _partial_instructions(parameters.value)]

    suite = evaluator_suite(_RefinementCase, storage=storage_path)(definition)
    cases = (EvaluatorSuiteCase(identifier="only", parameters=_RefinementCase(value="a")),)
    explored: list[str] = []

    def _counting_model() -> GenerativeModel:
        def generating(
            *,
            instructions: str,
            tools: ModelTools,
            context: Sequence[ModelContextElement],
            output: Any,
            **extra: Any,
        ) -> AsyncIterable[ModelOutputChunk]:
            if "EXACTLY 2 DIFFERENT refinement strategies" in instructions:
                explored.append("node")
                return _stream_of(TextContent.of(_STRATEGIES_REPLY))

            return _stream_of(TextContent.of("REFINED INSTRUCTIONS"))

        return GenerativeModel(generating=generating)

    async with ctx.scope(
        "test",
        _counting_model(),
        TemplatesRepository.volatile(subject="ORIGINAL INSTRUCTIONS"),
    ):
        await suite.add_case(cases[0].parameters, identifier=cases[0].identifier)

        await refine_instructions(
            Template.of("subject"),
            evaluator_suite=suite.prepared(),
            evaluator_cases=cases,
            rounds_limit=3,
        )

    # a binary tree explored to depth 3 expands 2**3 - 1 nodes, each exactly once
    assert len(explored) == 7
