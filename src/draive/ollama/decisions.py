from collections.abc import Mapping, MutableSequence, Sequence
from typing import Any, Final

from haiway import Meta, ctx
from ollama import (
    ResponseError,
    SystemOneAnswer,
    SystemOneChoiceAnswer,
    SystemOneChoiceQuestion,
    SystemOneNoulAnswer,
    SystemOneNoulQuestion,
    SystemOneQuestion,
    SystemOneResponse,
    SystemOneScoreAnswer,
    SystemOneScoreQuestion,
)

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
    ModelException,
    ModelInputInvalid,
    ModelOutputFailed,
    model_rate_limit,
    record_decision_invocation,
    record_decision_metrics,
    record_usage_metrics,
)
from draive.multimodal import ArtifactContent, Multimodal, MultimodalContent, TextContent
from draive.ollama.api import OllamaAPI
from draive.ollama.config import OllamaDecisionConfig

__all__ = ("OllamaDecisions",)

# ollama answers up to 64 questions within a single request
_MAX_QUESTIONS: Final[int] = 64
_INPUT_INVALID_STATUS_CODES: Final[frozenset[int]] = frozenset((400, 413))
_RATE_LIMIT_STATUS_CODES: Final[frozenset[int]] = frozenset((429, 503))


class OllamaDecisions(OllamaAPI):
    async def decide(
        self,
        *,
        input: Multimodal,  # noqa: A002
        questions: Sequence[DecisionQuestion],
        config: OllamaDecisionConfig | None = None,
        **extra: Any,
    ) -> Sequence[DecisionAnswer]:
        """Answer decision questions using an Ollama System One model.

        Requires a decision model pulled to the Ollama server, i.e. ``tev1``
        or ``nimble``. Answer meta carries calibrated ``probability`` (checks)
        or ``probabilities`` (choices, selections and ratings), the normalized
        expected ``score`` (ratings) and ``confidence``, the probability of the
        decided outcome instead of the concentration measure reported by the
        server.

        Parameters
        ----------
        input : Multimodal
            Textual content the decisions are made about. The Ollama client does
            not deliver images, use ``draive.systemone.SystemOne`` with
            ``images="images"`` to decide about images.
        questions : Sequence[DecisionQuestion]
            Questions with unique identifiers.
        config : OllamaDecisionConfig | None, default=None
            Configuration to use, resolved from the context when omitted.
        **extra : Any
            Unused, accepted for compatibility with ``DecisionGenerating``.

        Returns
        -------
        Sequence[DecisionAnswer]
            Calibrated answers in the order of their questions.

        Raises
        ------
        ModelInputInvalid
            When the input is not supported or the server rejected the request.
        ModelRateLimit
            When the server was rate limited or unavailable.
        ModelOutputFailed
            When the request failed otherwise or the response was malformed.
        """
        async with ctx.scope("model.decision"):
            config = config or ctx.state(OllamaDecisionConfig)
            record_decision_invocation(
                provider="ollama",
                model=config.model,
                questions=questions,
            )

            state: str
            request_questions: Mapping[str, SystemOneQuestion]
            try:
                state = _request_state(MultimodalContent.of(input))
                request_questions = _request_questions(questions)

            except ValueError as exc:
                raise ModelInputInvalid(
                    provider="ollama",
                    model=config.model,
                    reason=str(exc),
                ) from exc

            response: SystemOneResponse
            try:
                response = await self._client.systemone(
                    model=config.model,
                    state=state,
                    questions=request_questions,
                )

            except ResponseError as exc:
                raise _status_failure(
                    exc,
                    model=config.model,
                ) from exc

            except ModelException as exc:
                raise exc

            except Exception as exc:
                raise ModelOutputFailed(
                    provider="ollama",
                    model=config.model,
                    reason=str(exc),
                ) from exc

            record_decision_metrics(
                provider="ollama",
                model=config.model,
                questions=len(questions),
                requests=1,
            )
            record_usage_metrics(
                provider="ollama",
                model=config.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

            try:
                return tuple(
                    _answer(
                        question,
                        key=_question_key(index),
                        answers=response.answers,
                    )
                    for index, question in enumerate(questions)
                )

            except ValueError as exc:
                raise ModelOutputFailed(
                    provider="ollama",
                    model=config.model,
                    reason=str(exc),
                ) from exc


def _status_failure(
    error: ResponseError,
    /,
    *,
    model: str,
) -> ModelException:
    if error.status_code in _RATE_LIMIT_STATUS_CODES:
        return model_rate_limit(
            provider="ollama",
            model=model,
            retry_after=None,
        )

    if error.status_code in _INPUT_INVALID_STATUS_CODES:
        return ModelInputInvalid(
            provider="ollama",
            model=model,
            reason=error.error,
        )

    return ModelOutputFailed(
        provider="ollama",
        model=model,
        reason=f"Request failed with status {error.status_code}: {error.error}",
    )


def _request_state(
    content: MultimodalContent,
    /,
) -> str:
    elements: MutableSequence[str] = []
    for part in content.parts:
        match part:
            case TextContent():
                if part.text:
                    elements.append(part.text)

            case ArtifactContent():
                if not part.hidden:
                    elements.append(part.to_str())

            case _:
                raise ValueError(
                    "Ollama decisions support text input only,"
                    " use draive.systemone.SystemOne with images='images' for images"
                )

    if not elements:
        raise ValueError("Decision input can't be empty")

    return "\n".join(elements)


def _request_questions(
    questions: Sequence[DecisionQuestion],
    /,
) -> Mapping[str, SystemOneQuestion]:
    # questions are keyed by their position, selections are resolved as
    # an independent noul question for each of their options
    request: dict[str, SystemOneQuestion] = {}
    for index, question in enumerate(questions):
        key: str = _question_key(index)
        match question:
            case DecisionCheck():
                request[key] = SystemOneNoulQuestion(instructions=question.criteria)

            case DecisionChoice():
                request[key] = SystemOneChoiceQuestion(
                    instructions=question.criteria,
                    criteria={
                        option.identifier: option.description or None for option in question.options
                    },
                )

            case DecisionSelection():
                for option_index, option in enumerate(question.options):
                    request[_option_key(key, index=option_index)] = SystemOneNoulQuestion(
                        instructions=_option_instructions(question.criteria, option=option)
                    )

            case DecisionRating():
                request[key] = SystemOneScoreQuestion(
                    instructions=question.criteria,
                    criteria=[_level_criterion(level) for level in question.levels],
                )

    if len(request) > _MAX_QUESTIONS:
        raise ValueError(
            f"Decision supports up to {_MAX_QUESTIONS} questions,"
            " each selection option counts as a separate question"
        )

    return request


def _question_key(
    index: int,
    /,
) -> str:
    return f"q{index}"


def _option_key(
    key: str,
    /,
    *,
    index: int,
) -> str:
    return f"{key}_{index}"


def _option_instructions(
    criteria: str,
    /,
    *,
    option: DecisionOption,
) -> str:
    if option.description:
        return f"{criteria}\n\nDoes this option apply? {option.identifier}: {option.description}"

    return f"{criteria}\n\nDoes this option apply? {option.identifier}"


def _level_criterion(
    level: DecisionOption,
    /,
) -> str:
    # score criteria are plain descriptions, the level position is its value
    if level.description:
        return f"{level.identifier}: {level.description}"

    return level.identifier


def _answer(
    question: DecisionQuestion,
    /,
    *,
    key: str,
    answers: Mapping[str, SystemOneAnswer],
) -> DecisionAnswer:
    match question:
        case DecisionCheck():
            return _check_answer(question, answer=answers.get(key))

        case DecisionChoice():
            return _choice_answer(question, answer=answers.get(key))

        case DecisionSelection():
            return _selection_answer(question, key=key, answers=answers)

        case DecisionRating():
            return _rating_answer(question, answer=answers.get(key))


def _check_answer(
    question: DecisionCheck,
    /,
    *,
    answer: SystemOneAnswer | None,
) -> DecisionCheckAnswer:
    if not isinstance(answer, SystemOneNoulAnswer):
        raise ValueError("Decision completed without a matching noul answer")

    return DecisionCheckAnswer(
        identifier=question.identifier,
        value=answer.noul >= 0.5,  # noqa: PLR2004
        meta=Meta.of(
            {
                "probability": answer.noul,
                "confidence": max(answer.noul, 1.0 - answer.noul),
            }
        ),
    )


def _choice_answer(
    question: DecisionChoice,
    /,
    *,
    answer: SystemOneAnswer | None,
) -> DecisionChoiceAnswer:
    if not isinstance(answer, SystemOneChoiceAnswer):
        raise ValueError("Decision completed without a matching choice answer")

    probabilities: Mapping[str, float] = {
        option.identifier: answer.probabilities.get(option.identifier, 0.0)
        for option in question.options
    }
    for option in question.options:
        if option.identifier == answer.choice:
            return DecisionChoiceAnswer(
                identifier=question.identifier,
                choice=option,
                meta=option.meta.merged_with(
                    {
                        "probabilities": probabilities,
                        "confidence": probabilities[option.identifier],
                    }
                ),
            )

    raise ValueError("Decision completed without a matching choice answer")


def _selection_answer(
    question: DecisionSelection,
    /,
    *,
    key: str,
    answers: Mapping[str, SystemOneAnswer],
) -> DecisionSelectionAnswer:
    probabilities: dict[str, float] = {}
    for index, option in enumerate(question.options):
        answer: SystemOneAnswer | None = answers.get(_option_key(key, index=index))
        if not isinstance(answer, SystemOneNoulAnswer):
            raise ValueError("Decision completed without a matching noul answer")

        probabilities[option.identifier] = answer.noul

    return DecisionSelectionAnswer(
        identifier=question.identifier,
        selected=tuple(
            option
            for option in question.options
            if probabilities[option.identifier] >= 0.5  # noqa: PLR2004
        ),
        meta=Meta.of({"probabilities": probabilities}),
    )


def _rating_answer(
    question: DecisionRating,
    /,
    *,
    answer: SystemOneAnswer | None,
) -> DecisionRatingAnswer:
    if not isinstance(answer, SystemOneScoreAnswer):
        raise ValueError("Decision completed without a matching score answer")

    # probabilities are keyed by the stringified level index
    probabilities: Sequence[float] = tuple(
        answer.probabilities.get(str(index), 0.0) for index in range(len(question.levels))
    )
    last: int = len(question.levels) - 1
    index: int
    if any(probabilities):
        index = max(range(len(probabilities)), key=probabilities.__getitem__)

    else:
        index = min(max(round(answer.score), 0), last)

    level: DecisionOption = question.levels[index]
    return DecisionRatingAnswer(
        identifier=question.identifier,
        level=level,
        meta=level.meta.merged_with(
            {
                # the api scores in level positions, normalized to 0-1
                "score": min(max(answer.score / last, 0.0), 1.0),
                "probabilities": {
                    level.identifier: probability
                    for level, probability in zip(question.levels, probabilities, strict=True)
                },
                "confidence": probabilities[index],
            }
        ),
    )
