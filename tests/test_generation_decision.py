import asyncio
import json
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from typing import Any

import pytest
from haiway import ValidationError, ctx

from draive import (
    DecisionAnswer,
    DecisionCheck,
    DecisionCheckAnswer,
    DecisionChoice,
    DecisionChoiceAnswer,
    DecisionGeneration,
    DecisionOption,
    DecisionQuestion,
    DecisionRating,
    DecisionRatingAnswer,
    DecisionSelection,
    DecisionSelectionAnswer,
    GenerationOutputMissing,
    generate_completion_decisions,
)
from draive.models import GenerativeModel, ModelContext, ModelOutputChunk, ModelTools
from draive.multimodal import Multimodal, TextContent


def _answering(
    answers: Sequence[DecisionAnswer],
    captured: dict[str, Any] | None = None,
) -> Any:
    async def generating(
        *,
        input: Multimodal,  # noqa: A002
        questions: Sequence[DecisionQuestion],
        **extra: Any,
    ) -> Sequence[DecisionAnswer]:
        if captured is not None:
            captured["input"] = input
            captured["questions"] = questions
            captured["extra"] = extra

        return answers

    return generating


def _options(*identifiers: str) -> Sequence[DecisionOption]:
    return tuple(DecisionOption(identifier=identifier) for identifier in identifiers)


@pytest.mark.asyncio
async def test_decide_forwards_questions_and_returns_backend_answers() -> None:
    captured: dict[str, Any] = {}
    questions: Sequence[DecisionQuestion] = (
        DecisionCheck(identifier="angry", criteria="Is the customer angry?"),
        DecisionChoice(
            identifier="route",
            criteria="Which team should handle this?",
            options=_options("billing", "shipping"),
        ),
    )
    # backends are responsible for answering every question in order
    answers: Sequence[DecisionAnswer] = (
        DecisionCheckAnswer(identifier="angry", value=True),
        DecisionChoiceAnswer(identifier="route", choice=DecisionOption(identifier="billing")),
    )

    async with ctx.scope("test", DecisionGeneration(generating=_answering(answers, captured))):
        result = await DecisionGeneration.decide(
            input="I was charged twice",
            questions=questions,
            trace_id="trace-1",
        )

    assert result == answers
    assert captured["input"] == "I was charged twice"
    assert captured["questions"] == questions
    assert captured["extra"] == {"trace_id": "trace-1"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "questions",
    [
        (),
        (
            DecisionCheck(identifier="same", criteria="Is it?"),
            DecisionRating(identifier="same", criteria="Rate", levels=_options("low", "high")),
        ),
    ],
)
async def test_decide_asserts_non_empty_unique_questions(
    questions: Sequence[DecisionQuestion],
) -> None:
    async with ctx.scope("test", DecisionGeneration(generating=_answering(()))):
        with pytest.raises(AssertionError):
            await DecisionGeneration.decide(input="any", questions=questions)


@pytest.mark.parametrize(
    "make",
    [
        lambda: DecisionOption(identifier=""),
        lambda: DecisionOption(identifier=" a"),
        lambda: DecisionCheck(identifier="", criteria="Is it?"),
        lambda: DecisionCheck(identifier="check", criteria="  "),
        lambda: DecisionChoice(identifier="choice", criteria="Pick", options=_options("a")),
        lambda: DecisionChoice(
            identifier="choice",
            criteria="Pick",
            options=_options("a", "a"),
        ),
        lambda: DecisionRating(identifier="rating", criteria="Rate", levels=_options("low")),
        lambda: DecisionRating(identifier="rating", criteria="Rate", levels=_options("low", "low")),
    ],
)
def test_malformed_questions_can_not_be_created(make: Callable[[], Any]) -> None:
    with pytest.raises(ValidationError):
        make()


@pytest.mark.asyncio
async def test_single_question_helpers_return_typed_answers() -> None:
    captured: dict[str, Any] = {}

    async with ctx.scope(
        "test",
        DecisionGeneration(
            generating=_answering((DecisionCheckAnswer(identifier="check", value=False),), captured)
        ),
    ):
        checked = await DecisionGeneration.check(
            input="any",
            criteria="Is it damaged?",
        )

    assert checked == DecisionCheckAnswer(identifier="check", value=False)
    assert captured["questions"] == (
        DecisionCheck(
            identifier="check",
            criteria="Is it damaged?",
        ),
    )

    options: Sequence[DecisionOption] = (
        DecisionOption(identifier="a", description="first"),
        DecisionOption(identifier="b"),
    )
    async with ctx.scope(
        "test",
        DecisionGeneration(
            generating=_answering(
                (DecisionChoiceAnswer(identifier="choice", choice=DecisionOption(identifier="b")),),
                captured,
            )
        ),
    ):
        chosen = await DecisionGeneration.choose(
            input="any",
            criteria="Pick",
            options=options,
        )

    assert chosen == DecisionChoiceAnswer(
        identifier="choice", choice=DecisionOption(identifier="b")
    )
    assert captured["questions"] == (
        DecisionChoice(identifier="choice", criteria="Pick", options=options),
    )

    async with ctx.scope(
        "test",
        DecisionGeneration(
            generating=_answering(
                (
                    DecisionRatingAnswer(
                        identifier="rating", level=DecisionOption(identifier="high")
                    ),
                ),
                captured,
            )
        ),
    ):
        rated = await DecisionGeneration.rate(
            input="any",
            criteria="Rate",
            levels=_options("low", "high"),
        )

    assert rated == DecisionRatingAnswer(
        identifier="rating", level=DecisionOption(identifier="high")
    )


_COMPLETIONS = DecisionGeneration(generating=generate_completion_decisions)


def _model_answering(
    results: Mapping[str, dict[str, Any]],
    captured: list[str] | None = None,
) -> GenerativeModel:
    async def generating(
        *,
        instructions: str,
        tools: ModelTools,
        context: ModelContext,
        output: Any,
        **extra: Any,
    ) -> AsyncGenerator[ModelOutputChunk]:
        assert tools == ModelTools.none
        assert len(context) == 1
        if captured is not None:
            captured.append(instructions)

        yield TextContent.of(json.dumps(results[output.__name__]))

    return GenerativeModel(generating=generating)


@pytest.mark.asyncio
async def test_default_decisions_use_generative_model_without_probabilities() -> None:
    instructions: list[str] = []
    model = _model_answering(
        {
            "_CheckResult": {"reasoning": "-", "value": True},
            "_ChoiceResult": {"reasoning": "-", "choice": " billing "},
            "_RatingResult": {"reasoning": "-", "level": "right now"},
        },
        instructions,
    )

    async with ctx.scope("test", model, _COMPLETIONS):
        result = await DecisionGeneration.decide(
            input="I was charged twice",
            questions=(
                DecisionCheck(identifier="angry", criteria="Is the customer angry?"),
                DecisionChoice(
                    identifier="route",
                    criteria="Which team?",
                    options=(
                        DecisionOption(
                            identifier="billing", description="payments", meta={"team_id": 42}
                        ),
                        DecisionOption(identifier="shipping"),
                    ),
                ),
                DecisionRating(
                    identifier="urgency",
                    criteria="How urgent?",
                    levels=(
                        DecisionOption(identifier="can wait"),
                        DecisionOption(identifier="today"),
                        DecisionOption(identifier="right now", meta={"page": True}),
                    ),
                ),
            ),
        )

    assert result == (
        DecisionCheckAnswer(identifier="angry", value=True, meta={"reasoning": "-"}),
        DecisionChoiceAnswer(
            identifier="route",
            choice=DecisionOption(
                identifier="billing",
                description="payments",
                meta={"team_id": 42},
            ),
            meta={"team_id": 42, "reasoning": "-"},
        ),
        DecisionRatingAnswer(
            identifier="urgency",
            level=DecisionOption(identifier="right now", meta={"page": True}),
            meta={"page": True, "reasoning": "-"},
        ),
    )
    assert any("- billing: payments\n- shipping" in element for element in instructions)
    assert any("- can wait\n- today\n- right now" in element for element in instructions)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("question", "results"),
    [
        (
            DecisionChoice(identifier="q", criteria="Pick", options=_options("a", "b")),
            {"_ChoiceResult": {"reasoning": "-", "choice": "c"}},
        ),
        (
            DecisionRating(identifier="q", criteria="Rate", levels=_options("low", "high")),
            {"_RatingResult": {"reasoning": "-", "level": "extreme"}},
        ),
        (
            DecisionCheck(identifier="q", criteria="Is it?"),
            {"_CheckResult": {"reasoning": "-"}},
        ),
    ],
)
async def test_default_decisions_reject_answers_outside_of_criteria(
    question: DecisionQuestion,
    results: Mapping[str, dict[str, Any]],
) -> None:
    async with ctx.scope("test", _model_answering(results), _COMPLETIONS):
        with pytest.raises(GenerationOutputMissing):
            await DecisionGeneration.decide(input="any", questions=(question,))


@pytest.mark.parametrize(
    "options",
    [(), _options("a", "a")],
)
def test_malformed_selections_can_not_be_created(options: Sequence[DecisionOption]) -> None:
    with pytest.raises(ValidationError):
        DecisionSelection(identifier="selection", criteria="Pick", options=options)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("selected", "expected"),
    [
        ([" c ", "a", "a"], ("a", "c")),
        ([], ()),
    ],
)
async def test_default_selection_uses_generative_model(
    selected: Sequence[str],
    expected: Sequence[str],
) -> None:
    instructions: list[str] = []
    model = _model_answering(
        {"_SelectionResult": {"reasoning": "-", "selected": selected}},
        instructions,
    )
    async with ctx.scope("test", model, _COMPLETIONS):
        answer = await DecisionGeneration.select(
            input="any",
            criteria="Which apply?",
            options=_options("a", "b", "c"),
        )

    assert answer == DecisionSelectionAnswer(
        identifier="selection",
        selected=_options(*expected),
        meta={"reasoning": "-"},
    )
    assert "any number of the available choices" in instructions[0]


@pytest.mark.asyncio
async def test_default_selection_rejects_unknown_options() -> None:
    model = _model_answering({"_SelectionResult": {"reasoning": "-", "selected": ["x"]}})
    async with ctx.scope("test", model, _COMPLETIONS):
        with pytest.raises(GenerationOutputMissing):
            await DecisionGeneration.select(
                input="any",
                criteria="Which apply?",
                options=_options("a", "b"),
            )


@pytest.mark.asyncio
async def test_default_decisions_bound_concurrent_completions() -> None:
    running: int = 0
    peak: int = 0

    async def generating(
        *,
        instructions: str,
        tools: ModelTools,
        context: ModelContext,
        output: Any,
        **extra: Any,
    ) -> AsyncGenerator[ModelOutputChunk]:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1
        yield TextContent.of(json.dumps({"reasoning": "-", "value": True}))

    async with ctx.scope("test", GenerativeModel(generating=generating), _COMPLETIONS):
        answers = await DecisionGeneration.decide(
            input="any",
            questions=tuple(
                DecisionCheck(identifier=f"q{index}", criteria="Is it?") for index in range(20)
            ),
        )

    assert [answer.identifier for answer in answers] == [f"q{index}" for index in range(20)]
    assert 1 < peak <= 8
