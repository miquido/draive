from collections.abc import Sequence
from typing import Any

import pytest
from haiway import ctx
from haiway.context.types import ContextStateMissing

from draive import (
    DecisionAnswer,
    DecisionCheck,
    DecisionCheckAnswer,
    DecisionGeneration,
    DecisionModel,
    DecisionQuestion,
    Multimodal,
    Template,
    TemplatesRepository,
)


def _backend(captured: list[dict[str, Any]] | None = None) -> DecisionModel:
    async def generating(
        *,
        input: Multimodal,  # noqa: A002
        questions: Sequence[DecisionQuestion],
        **extra: Any,
    ) -> Sequence[DecisionAnswer]:
        if captured is not None:
            captured.append({"input": input, "questions": questions, "extra": extra})

        return tuple(
            DecisionCheckAnswer(identifier=question.identifier, value=True)
            for question in questions
        )

    return DecisionModel(generating=generating)


def _questions(*identifiers: str) -> Sequence[DecisionQuestion]:
    return tuple(
        DecisionCheck(identifier=identifier, criteria="is it?") for identifier in identifiers
    )


def test_decision_model_requires_a_backend() -> None:
    with pytest.raises(TypeError):
        DecisionModel()  # pyright: ignore[reportCallIssue]


@pytest.mark.asyncio
async def test_decide_delegates_to_the_scoped_backend() -> None:
    captured: list[dict[str, Any]] = []

    async with ctx.scope("test", _backend(captured)):
        answers = await DecisionModel.decide(
            input="content",
            questions=_questions("a", "b"),
            option="value",
        )

    assert [answer.identifier for answer in answers] == ["a", "b"]
    assert captured[0]["input"] == "content"
    assert captured[0]["extra"] == {"option": "value"}


@pytest.mark.asyncio
async def test_decide_works_on_an_instance_without_a_scope() -> None:
    answers = await _backend().decide(input="content", questions=_questions("a"))

    assert [answer.identifier for answer in answers] == ["a"]


@pytest.mark.asyncio
@pytest.mark.parametrize("identifiers", [(), ("a", "a")])
async def test_decide_asserts_non_empty_unique_questions(identifiers: tuple[str, ...]) -> None:
    async with ctx.scope("test", _backend()):
        with pytest.raises(AssertionError):
            await DecisionModel.decide(input="content", questions=_questions(*identifiers))


@pytest.mark.asyncio
async def test_decide_without_a_bound_backend_fails() -> None:
    async with ctx.scope("test"):
        with pytest.raises(ContextStateMissing):
            await DecisionModel.decide(input="content", questions=_questions("a"))


@pytest.mark.asyncio
async def test_generation_facade_uses_the_scoped_decision_model_by_default() -> None:
    captured: list[dict[str, Any]] = []

    async with ctx.scope("test", _backend(captured)):
        answer = await DecisionGeneration.check(input="content", criteria="is it?", flag=1)

    assert answer.value is True
    assert captured[0]["input"] == "content"
    assert captured[0]["extra"] == {"flag": 1}


@pytest.mark.asyncio
async def test_generation_facade_explicit_backend_wins_over_decision_model() -> None:
    explicit: list[dict[str, Any]] = []
    scoped: list[dict[str, Any]] = []

    async def generating(
        *,
        input: Multimodal,  # noqa: A002
        questions: Sequence[DecisionQuestion],
        **extra: Any,
    ) -> Sequence[DecisionAnswer]:
        explicit.append({})
        return tuple(
            DecisionCheckAnswer(identifier=question.identifier, value=False)
            for question in questions
        )

    async with ctx.scope("test", _backend(scoped), DecisionGeneration(generating=generating)):
        answer = await DecisionGeneration.check(input="content", criteria="is it?")

    assert answer.value is False
    assert explicit and not scoped


@pytest.mark.asyncio
async def test_generation_facade_without_any_backend_fails() -> None:
    async with ctx.scope("test"):
        with pytest.raises(ContextStateMissing):
            await DecisionGeneration.check(input="content", criteria="is it?")


@pytest.mark.asyncio
async def test_generation_facade_resolves_template_input_before_deciding() -> None:
    captured: list[dict[str, Any]] = []

    async with ctx.scope(
        "test",
        _backend(captured),
        TemplatesRepository.volatile(ticket="Ticket {%subject%}"),
    ):
        await DecisionGeneration.check(
            input=Template.of("ticket", arguments={"subject": "refund"}),
            criteria="is it?",
        )

    assert "Ticket refund" in str(captured[0]["input"])
