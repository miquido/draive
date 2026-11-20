import re
from collections.abc import Generator, Mapping, Sequence
from typing import Any, Final, cast

from google.genai._interactions import APIStatusError
from google.genai.decisions import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestionParam,
    CreateDecisionResponse,
    DecisionInputContentParam,
    NoulAnswer,
    NoulQuestionParam,
    QuestionParam,
    ScoreAnswer,
)
from haiway import Meta, ctx

from draive.gemini.api import GeminiAPI
from draive.gemini.config import GeminiDecisionConfig
from draive.gemini.utils import (
    RATE_LIMIT_STATUS_CODE,
    UNAVAILABLE_STATUS_CODE,
    quota_limit_details,
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
from draive.resources import ResourceContent, ResourceReference

__all__ = ("GeminiDecisions",)

# decisions are served only within the beta api surface
_API_VERSION: Final[str] = "v1beta"
_MAX_QUESTIONS: Final[int] = 1000
_MAX_CRITERIA: Final[int] = 255
_MAX_OPTION_LENGTH: Final[int] = 128
_CONTROL_CHARACTERS: Final[re.Pattern[str]] = re.compile(r"[\x00-\x1f\x7f]")
# error codes reported when the input was rejected by content policies
_SAFETY_ERROR_CODES: Final[frozenset[str]] = frozenset(
    (
        "safety",
        "prohibited_content",
        "blocklist",
        "image_safety",
        "content_blocked",
    )
)


class GeminiDecisions(GeminiAPI):
    async def decide(
        self,
        *,
        input: Multimodal,  # noqa: A002
        questions: Sequence[DecisionQuestion],
        config: GeminiDecisionConfig | None = None,
        **extra: Any,
    ) -> Sequence[DecisionAnswer]:
        """Answer decision questions using the Gemini Decisions API.

        Answer meta carries calibrated ``probability`` (checks) or
        ``probabilities`` (choices and ratings), the normalized expected
        ``score`` (ratings) and ``confidence``, the probability of the decided
        outcome.

        Parameters
        ----------
        input : Multimodal
            Content the decisions are made about.
        questions : Sequence[DecisionQuestion]
            Questions with unique identifiers.
        config : GeminiDecisionConfig | None, default=None
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
            When the input or questions exceed what the API accepts.
        ModelRateLimit
            When the request was rate limited or the service was unavailable.
        ModelOutputFailed
            When the input was blocked or the request failed otherwise.
        """
        async with ctx.scope("model.decision"):
            config = config or ctx.state(GeminiDecisionConfig)
            record_decision_invocation(
                provider="gemini",
                model=config.model,
                questions=questions,
            )

            # vertex can be also selected through the environment, the client resolves it
            if self._client.vertexai:
                raise ModelInputInvalid(
                    provider="gemini",
                    model=config.model,
                    reason="Decisions are not available through Vertex AI",
                )

            request_input: list[DecisionInputContentParam]
            request_questions: dict[str, QuestionParam]
            try:
                request_input = list(_request_input(MultimodalContent.of(input)))
                request_questions = _request_questions(questions)

            except ValueError as exc:
                raise ModelInputInvalid(
                    provider="gemini",
                    model=config.model,
                    reason=str(exc),
                ) from exc

            response: CreateDecisionResponse
            try:
                response = await self._client.aio.decisions.create(
                    model=config.model,
                    input=request_input,
                    questions=request_questions,
                    api_version=_API_VERSION,
                )

            except APIStatusError as exc:
                raise _status_failure(
                    exc,
                    model=config.model,
                ) from exc

            except ModelException as exc:
                raise exc

            except Exception as exc:
                raise ModelOutputFailed(
                    provider="gemini",
                    model=config.model,
                    reason=str(exc),
                ) from exc

            record_decision_metrics(
                provider="gemini",
                model=config.model,
                questions=len(request_questions),
                requests=1,
            )
            if response.usage is not None:
                record_usage_metrics(
                    provider="gemini",
                    model=config.model,
                    input_tokens=response.usage.total_input_tokens,
                    cached_input_tokens=response.usage.total_cached_tokens,
                    output_tokens=response.usage.total_output_tokens,
                )

            if response.prompt_feedback is not None and response.prompt_feedback.block_reason:
                raise ModelOutputFailed(
                    provider="gemini",
                    model=config.model,
                    reason=(
                        f"Input blocked ({response.prompt_feedback.block_reason}):"
                        f" {response.prompt_feedback.block_reason_message or ''}"
                    ),
                )

            answers: Mapping[str, Answer] = response.answers or {}
            try:
                return tuple(
                    _answer(
                        question,
                        key=_question_key(index),
                        answers=answers,
                    )
                    for index, question in enumerate(questions)
                )

            except ValueError as exc:
                raise ModelOutputFailed(
                    provider="gemini",
                    model=config.model,
                    reason=str(exc),
                ) from exc


def _status_failure(
    error: APIStatusError,
    /,
    *,
    model: str,
) -> Exception:
    if error.status_code == RATE_LIMIT_STATUS_CODE:
        return model_rate_limit(
            provider="gemini",
            model=model,
            retry_after=error.response.headers.get("retry-after"),
            quota_limit=quota_limit_details(error.body),
        )

    if error.status_code == UNAVAILABLE_STATUS_CODE:
        # temporary capacity failures are retryable, same as rate limits
        return model_rate_limit(
            provider="gemini",
            model=model,
            retry_after=error.response.headers.get("retry-after"),
            error_code="unavailable",
        )

    match error.body:
        case {"error": {"code": str() as code}} if code in _SAFETY_ERROR_CODES:
            return ModelOutputFailed(
                provider="gemini",
                model=model,
                reason=f"Input blocked ({code})",
            )

        case _:
            return ModelOutputFailed(
                provider="gemini",
                model=model,
                reason=str(error),
            )


def _request_input(
    content: MultimodalContent,
    /,
) -> Generator[DecisionInputContentParam]:
    produced: bool = False
    for part in content.parts:
        if isinstance(part, TextContent):
            if part.text:
                produced = True
                yield {"type": "text", "text": part.text}

        elif isinstance(part, ResourceContent):
            produced = True
            yield cast(
                DecisionInputContentParam,
                {
                    "type": _media_type(part.mime_type),
                    # resource data is already base64 encoded
                    "data": part.data,
                    "mime_type": part.mime_type,
                },
            )

        elif isinstance(part, ResourceReference):
            if part.uri.startswith("gs://"):
                raise ValueError("Decision input does not support gs:// references")

            produced = True
            yield cast(
                DecisionInputContentParam,
                {
                    "type": _media_type(part.mime_type),
                    "uri": part.uri,
                    "mime_type": part.mime_type,
                },
            )

        else:
            assert isinstance(part, ArtifactContent)  # nosec: B101
            if part.hidden:
                continue

            produced = True
            yield {"type": "text", "text": part.to_str()}

    if not produced:
        raise ValueError("Decision input can't be empty")


def _media_type(
    mime_type: str,
    /,
) -> str:
    if mime_type.startswith("image/"):
        return "image"

    if mime_type.startswith("audio/"):
        return "audio"

    if mime_type.startswith("video/"):
        return "video"

    if mime_type == "application/pdf":
        return "document"

    raise ValueError(f"Decision input does not support {mime_type} content")


def _request_questions(
    questions: Sequence[DecisionQuestion],
    /,
) -> dict[str, QuestionParam]:
    # questions are keyed by their position, selections are resolved as
    # an independent noul question for each of their options
    request: dict[str, QuestionParam] = {}
    for index, question in enumerate(questions):
        key: str = _question_key(index)
        match question:
            case DecisionCheck():
                request[key] = _noul_question(question.criteria)

            case DecisionChoice():
                request[key] = _choice_question(question)

            case DecisionSelection():
                for option_index, option in enumerate(question.options):
                    request[_option_key(key, index=option_index)] = _noul_question(
                        _option_instructions(question.criteria, option=option)
                    )

            case DecisionRating():
                _validate_criteria(question.identifier, criteria=question.levels)
                request[key] = {
                    "type": "score",
                    "instructions": question.criteria,
                    "criteria": [_level_criterion(level) for level in question.levels],
                }

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
    instructions: str,
    /,
    *,
    option: DecisionOption,
) -> str:
    if option.description:
        return (
            f"{instructions}\n\nDoes this option apply? {option.identifier}: {option.description}"
        )

    return f"{instructions}\n\nDoes this option apply? {option.identifier}"


def _noul_question(
    instructions: str,
    /,
) -> NoulQuestionParam:
    return {
        "type": "noul",
        "instructions": instructions,
    }


def _choice_question(
    question: DecisionChoice,
    /,
) -> ChoiceQuestionParam:
    key: str = question.identifier
    _validate_criteria(key, criteria=question.options)
    for option in question.options:
        if len(option.identifier) > _MAX_OPTION_LENGTH or _CONTROL_CHARACTERS.search(
            option.identifier
        ):
            raise ValueError(
                f"Decision question '{key}' option '{option.identifier}' has to be up to"
                f" {_MAX_OPTION_LENGTH} characters without control characters"
            )

    return {
        "type": "choice",
        "instructions": question.criteria,
        "criteria": {option.identifier: option.description or "" for option in question.options},
    }


def _validate_criteria(
    key: str,
    /,
    *,
    criteria: Sequence[Any],
) -> None:
    if len(criteria) > _MAX_CRITERIA:
        raise ValueError(f"Decision question '{key}' supports up to {_MAX_CRITERIA} criteria")


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
    answers: Mapping[str, Answer],
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
    answer: Answer | None,
) -> DecisionCheckAnswer:
    if not isinstance(answer, NoulAnswer):
        raise ValueError("Decision completed without a matching noul answer")

    return DecisionCheckAnswer(
        identifier=question.identifier,
        value=answer.noul >= 0.5,  # noqa: PLR2004
        meta=Meta.of(
            {
                "probability": answer.noul,
                "confidence": answer.confidence,
            }
        ),
    )


def _choice_answer(
    question: DecisionChoice,
    /,
    *,
    answer: Answer | None,
) -> DecisionChoiceAnswer:
    if not isinstance(answer, ChoiceAnswer):
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
                        "confidence": answer.confidence,
                    }
                ),
            )

    raise ValueError("Decision completed without a matching choice answer")


def _selection_answer(
    question: DecisionSelection,
    /,
    *,
    key: str,
    answers: Mapping[str, Answer],
) -> DecisionSelectionAnswer:
    probabilities: dict[str, float] = {}
    for index, option in enumerate(question.options):
        answer: Answer | None = answers.get(_option_key(key, index=index))
        if not isinstance(answer, NoulAnswer):
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
    answer: Answer | None,
) -> DecisionRatingAnswer:
    if not isinstance(answer, ScoreAnswer):
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

    return DecisionRatingAnswer(
        identifier=question.identifier,
        level=question.levels[index],
        meta=question.levels[index].meta.merged_with(
            {
                # the api scores in level positions, normalized to 0-1
                "score": min(max(answer.score / last, 0.0), 1.0),
                "probabilities": {
                    level.identifier: probability
                    for level, probability in zip(question.levels, probabilities, strict=True)
                },
                "confidence": answer.confidence,
            }
        ),
    )
