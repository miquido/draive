from collections.abc import MutableSequence, Sequence
from typing import Any, Final

from haiway import Meta, State, concurrently

from draive.generation.types import GenerationOutputMissing
from draive.models import (
    DecisionAnswer,
    DecisionCheck,
    DecisionCheckAnswer,
    DecisionChoice,
    DecisionChoiceAnswer,
    DecisionOption,
    DecisionQuestion,
    DecisionRating,
    DecisionRatingAnswer,
    DecisionSelection,
    DecisionSelectionAnswer,
    GenerativeModel,
    ModelInput,
    ModelOutputStream,
    record_decision_invocation,
    record_decision_metrics,
)
from draive.multimodal import Multimodal, MultimodalContent, MultimodalContentPart
from draive.steps.decoding import decode_content

__all__ = ("generate_completion_decisions",)

# bounds the number of simultaneous completions within a single decision batch
_CONCURRENT_QUESTIONS: Final[int] = 8


async def generate_completion_decisions(
    *,
    input: Multimodal,  # noqa: A002
    questions: Sequence[DecisionQuestion],
    **extra: Any,
) -> Sequence[DecisionAnswer]:
    record_decision_invocation(
        provider="completion",
        questions=questions,
    )
    # each question is answered in isolation, same as dedicated decision models do
    return await concurrently(
        (
            _decide(
                question,
                input=input,
                **extra,
            )
            for question in questions
        ),
        concurrent_tasks=_CONCURRENT_QUESTIONS,
    )


async def _decide(
    question: DecisionQuestion,
    /,
    *,
    input: Multimodal,  # noqa: A002
    **extra: Any,
) -> DecisionAnswer:
    answer: DecisionAnswer = await _answer(
        question,
        input=input,
        **extra,
    )
    # counted when the request completes, so work of a failed batch is not lost
    record_decision_metrics(
        provider="completion",
        questions=1,
        requests=1,  # a completion for each of the questions
    )
    return answer


async def _answer(
    question: DecisionQuestion,
    /,
    *,
    input: Multimodal,  # noqa: A002
    **extra: Any,
) -> DecisionAnswer:
    match question:
        case DecisionCheck():
            return await _check(
                question,
                input=input,
                **extra,
            )

        case DecisionChoice():
            return await _choose(
                question,
                input=input,
                **extra,
            )

        case DecisionSelection():
            return await _select(
                question,
                input=input,
                **extra,
            )

        case DecisionRating():
            return await _rate(
                question,
                input=input,
                **extra,
            )


class _CheckResult(State, serializable=True):
    reasoning: str
    value: bool


async def _check(
    question: DecisionCheck,
    /,
    *,
    input: Multimodal,  # noqa: A002
    **extra: Any,
) -> DecisionCheckAnswer:
    result: _CheckResult = await _generate(
        _CheckResult,
        instructions=_instructions(
            question.criteria,
            choices="- true: yes, it applies\n- false: no, it does not apply",
            answer="`value` set to the outcome which applies",
        ),
        input=input,
        **extra,
    )

    return DecisionCheckAnswer(
        identifier=question.identifier,
        value=result.value,
        meta=Meta.of({"reasoning": result.reasoning}),
    )


class _ChoiceResult(State, serializable=True):
    reasoning: str
    choice: str


async def _choose(
    question: DecisionChoice,
    /,
    *,
    input: Multimodal,  # noqa: A002
    **extra: Any,
) -> DecisionChoiceAnswer:
    result: _ChoiceResult = await _generate(
        _ChoiceResult,
        instructions=_instructions(
            question.criteria,
            choices=_listed(question.options),
            answer="`choice` set to the exact identifier of the one option which applies",
        ),
        input=input,
        **extra,
    )

    choice: DecisionOption = _selected(result.choice, options=question.options)
    return DecisionChoiceAnswer(
        identifier=question.identifier,
        choice=choice,
        meta=choice.meta.merged_with({"reasoning": result.reasoning}),
    )


class _SelectionResult(State, serializable=True):
    reasoning: str
    selected: Sequence[str]


async def _select(
    question: DecisionSelection,
    /,
    *,
    input: Multimodal,  # noqa: A002
    **extra: Any,
) -> DecisionSelectionAnswer:
    result: _SelectionResult = await _generate(
        _SelectionResult,
        instructions=_instructions(
            question.criteria,
            choices=_listed(question.options),
            answer=(
                "`selected` set to the exact identifiers of every option which applies,"
                " judging each option independently, empty when none of them applies"
            ),
            single=False,
        ),
        input=input,
        **extra,
    )

    selected: set[str] = {
        _selected(identifier, options=question.options).identifier for identifier in result.selected
    }
    return DecisionSelectionAnswer(
        identifier=question.identifier,
        selected=tuple(option for option in question.options if option.identifier in selected),
        meta=Meta.of({"reasoning": result.reasoning}),
    )


class _RatingResult(State, serializable=True):
    reasoning: str
    level: str


async def _rate(
    question: DecisionRating,
    /,
    *,
    input: Multimodal,  # noqa: A002
    **extra: Any,
) -> DecisionRatingAnswer:
    result: _RatingResult = await _generate(
        _RatingResult,
        instructions=_instructions(
            question.criteria,
            choices=_listed(question.levels),
            answer=(
                "`level` set to the exact identifier of the one level which applies,"
                " levels are ordered from the lowest to the highest"
            ),
        ),
        input=input,
        **extra,
    )

    level: DecisionOption = _selected(result.level, options=question.levels)
    return DecisionRatingAnswer(
        identifier=question.identifier,
        level=level,
        meta=level.meta.merged_with({"reasoning": result.reasoning}),
    )


def _listed(
    options: Sequence[DecisionOption],
    /,
) -> str:
    return "\n".join(
        f"- {option.identifier}: {option.description}"
        if option.description
        else f"- {option.identifier}"
        for option in options
    )


def _selected(
    identifier: str,
    /,
    *,
    options: Sequence[DecisionOption],
) -> DecisionOption:
    selected: str = identifier.strip()
    for option in options:
        if option.identifier == selected:
            return option

    raise GenerationOutputMissing(
        f"Decision completed with an unknown option '{selected}'",
    )


async def _generate[Result: State](
    result: type[Result],
    /,
    *,
    instructions: str,
    input: Multimodal,  # noqa: A002
    **extra: Any,
) -> Result:
    # a single completion without tools, the answer is decoded from its structured output
    parts: MutableSequence[MultimodalContentPart] = []
    completion_stream: ModelOutputStream = GenerativeModel.completion(
        instructions=instructions,
        context=(ModelInput.of(MultimodalContent.of(input)),),
        output=result,
        **extra,
    )
    try:
        async for chunk in completion_stream:
            if isinstance(chunk, MultimodalContentPart):
                parts.append(chunk)

    finally:
        await completion_stream.aclose()

    try:
        return decode_content(MultimodalContent.of(*parts), result)

    except Exception as exc:
        raise GenerationOutputMissing("Decision completed without a decodable answer") from exc


def _instructions(
    criteria: str,
    /,
    *,
    choices: str,
    answer: str,
    single: bool = True,
) -> str:
    return (
        "Make a single decision about the provided input, evaluating the criteria"
        + (
            " with exactly one of the available choices."
            if single
            else " with any number of the available choices."
        )
        + f"\n\n<criteria>\n{criteria}\n</criteria>"
        f"\n\n<choices>\n{choices}\n</choices>"
        "\n\nBriefly explain your judgement in `reasoning` first,"
        f" then answer with {answer}."
    )
