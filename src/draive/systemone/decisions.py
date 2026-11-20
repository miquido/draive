import json
from collections.abc import Mapping, MutableMapping, MutableSequence, Sequence
from typing import Any, Final
from uuid import uuid4

from haiway import BasicValue, HTTPClientError, HTTPResponse, Meta, ctx

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
from draive.systemone.api import SystemOneAPI, SystemOneImages
from draive.systemone.config import SystemOneConfig

__all__ = ("SystemOneDecisions",)

_DECISIONS_PATH: Final[str] = "/v1/systemone"
_SUCCESS_STATUS_CODE: Final[int] = 200
_INPUT_INVALID_STATUS_CODES: Final[frozenset[int]] = frozenset((400, 413, 422))
_RATE_LIMIT_STATUS_CODES: Final[frozenset[int]] = frozenset((429, 503))
# image formats accepted as TypeSafe assets
_ASSET_FORMATS: Final[Mapping[str, str]] = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
}
# TypeSafe requires an explicit opt-in for image assets
_IMAGES_HEADERS: Final[Mapping[str, str]] = {"TypeSafe-Danger-Zone-Images": "true"}


class SystemOneDecisions(SystemOneAPI):
    async def decide(
        self,
        *,
        input: Multimodal,  # noqa: A002
        questions: Sequence[DecisionQuestion],
        config: SystemOneConfig | None = None,
        **extra: Any,
    ) -> Sequence[DecisionAnswer]:
        """Answer decision questions using a System One API server.

        Answer meta carries calibrated ``probability`` (checks) or
        ``probabilities`` (choices and ratings), the normalized expected
        ``score`` (ratings) and ``confidence``, the probability of the decided
        outcome instead of the concentration measure reported by the server.

        Parameters
        ----------
        input : Multimodal
            Content the decisions are made about, text with images.
        questions : Sequence[DecisionQuestion]
            Questions with unique identifiers.
        config : SystemOneConfig | None, default=None
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
            config = config or ctx.state(SystemOneConfig)
            record_decision_invocation(
                provider="systemone",
                model=config.model,
                questions=questions,
            )

            request: Mapping[str, Any]
            try:
                request = _request(
                    MultimodalContent.of(input),
                    questions=questions,
                    model=config.model,
                    images=self._images,
                )

            except ValueError as exc:
                raise ModelInputInvalid(
                    provider="systemone",
                    model=config.model,
                    reason=str(exc),
                ) from exc

            response_body: BasicValue
            try:
                response: HTTPResponse = await self._http.post(
                    url=_DECISIONS_PATH,
                    headers=_IMAGES_HEADERS if "assets" in request else None,
                    body=json.dumps(request).encode(),
                )
                payload: bytes = await response.body()
                if response.status_code != _SUCCESS_STATUS_CODE:
                    raise _status_failure(
                        response,
                        payload=payload,
                        model=config.model,
                    )

                response_body = json.loads(payload)

            except ModelException as exc:
                raise exc

            except (HTTPClientError, ValueError) as exc:
                raise ModelOutputFailed(
                    provider="systemone",
                    model=config.model,
                    reason=str(exc),
                ) from exc

            record_decision_metrics(
                provider="systemone",
                model=config.model,
                questions=len(request["questions"]),
                requests=1,
            )
            try:
                return _answers(
                    response_body,
                    questions=questions,
                    model=config.model,
                )

            except ValueError as exc:
                raise ModelOutputFailed(
                    provider="systemone",
                    model=config.model,
                    reason=str(exc),
                ) from exc


def _status_failure(
    response: HTTPResponse,
    /,
    *,
    payload: bytes,
    model: str,
) -> ModelException:
    if response.status_code in _RATE_LIMIT_STATUS_CODES:
        return model_rate_limit(
            provider="systemone",
            model=model,
            retry_after=response.headers.get("retry-after"),
        )

    message: str = _error_message(payload)
    if response.status_code in _INPUT_INVALID_STATUS_CODES:
        return ModelInputInvalid(
            provider="systemone",
            model=model,
            reason=message,
        )

    return ModelOutputFailed(
        provider="systemone",
        model=model,
        reason=f"Request failed with status {response.status_code}: {message}",
    )


def _error_message(
    payload: bytes,
    /,
) -> str:
    body: BasicValue
    try:
        body = json.loads(payload)

    except ValueError:
        return payload.decode(errors="replace")

    match body:
        # OpenJev
        case {"error": {"message": str() as message}}:
            return message

        # Ollama
        case {"error": str() as message}:
            return message

        # TypeSafe validation errors
        case {"detail": list() as details}:
            return "; ".join(
                str(detail.get("msg", detail)) if isinstance(detail, dict) else str(detail)
                for detail in details
            )

        # TypeSafe request errors
        case {"detail": str() as message}:
            return message

        case _:
            return payload.decode(errors="replace")


def _request(
    content: MultimodalContent,
    /,
    *,
    questions: Sequence[DecisionQuestion],
    model: str,
    images: SystemOneImages,
) -> Mapping[str, Any]:
    request: MutableMapping[str, Any] = {
        "model": model,
        "questions": _request_questions(questions),
    }
    match images:
        case "assets":
            state, assets = _assets_state(content)
            request["state"] = state
            if assets:
                request["assets"] = assets

        case "images":
            state, attached = _images_state(content)
            request["state"] = state
            if attached:
                request["images"] = attached

        case "inline":
            request["state"] = _inline_state(content)

    return request


def _assets_state(
    content: MultimodalContent,
    /,
) -> tuple[str, Mapping[str, Mapping[str, str]]]:
    # images are referenced by placeholders placed where they appear within the content
    elements: MutableSequence[str] = []
    assets: MutableMapping[str, Mapping[str, str]] = {}
    for part in content.parts:
        if isinstance(part, ResourceContent):
            image_format: str | None = _ASSET_FORMATS.get(part.mime_type)
            if image_format is None:
                raise ValueError(f"Decision input does not support {part.mime_type} content")

            placeholder: str = str(uuid4())
            assets[placeholder] = {
                "type": "image",
                "format": image_format,
                "data_base64": part.data,
            }
            elements.append(placeholder)

        elif element := _text(part):
            elements.append(element)

    if not elements:
        raise ValueError("Decision input can't be empty")

    return "\n".join(elements), assets


def _images_state(
    content: MultimodalContent,
    /,
) -> tuple[str, Sequence[str]]:
    # images are shared by all questions, separately from the textual state
    texts: MutableSequence[str] = []
    images: MutableSequence[str] = []
    for part in content.parts:
        if isinstance(part, ResourceContent):
            if not part.mime_type.startswith("image/"):
                raise ValueError(f"Decision input does not support {part.mime_type} content")

            images.append(part.data)

        elif element := _text(part):
            texts.append(element)

    if not texts:
        # state is required even when deciding about images only
        raise ValueError("Decision input requires text content")

    return "\n".join(texts), images


def _inline_state(
    content: MultimodalContent,
    /,
) -> str | Mapping[str, str]:
    texts: MutableSequence[str] = []
    image: str | None = None
    for part in content.parts:
        if isinstance(part, ResourceContent):
            if not part.mime_type.startswith("image/"):
                raise ValueError(f"Decision input does not support {part.mime_type} content")

            if image is not None:
                raise ValueError("Decision input supports at most one image")

            # OpenJev reads a single image from a data url within the state
            image = f"data:{part.mime_type};base64,{part.data}"

        elif element := _text(part):
            texts.append(element)

    text: str = "\n".join(texts)
    if image is None:
        if not text:
            raise ValueError("Decision input can't be empty")

        return text

    if text:
        return {"text": text, "image": image}

    return {"image": image}


def _text(
    part: TextContent | ResourceReference | ArtifactContent,
    /,
) -> str | None:
    if isinstance(part, TextContent):
        return part.text or None

    if isinstance(part, ResourceReference):
        raise ValueError("Decision input does not support resource references")

    if part.hidden:
        return None

    return part.to_str()


def _request_questions(
    questions: Sequence[DecisionQuestion],
    /,
) -> Mapping[str, Mapping[str, Any]]:
    # questions are keyed by their position, selections are resolved as
    # an independent noul question for each of their options
    request: dict[str, Mapping[str, Any]] = {}
    for index, question in enumerate(questions):
        key: str = _question_key(index)
        if isinstance(question, DecisionSelection):
            for option_index, option in enumerate(question.options):
                request[_option_key(key, index=option_index)] = {
                    "type": "noul",
                    "instructions": _option_instructions(question.criteria, option=option),
                }

        else:
            request[key] = _request_question(question)

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


def _request_question(
    question: DecisionCheck | DecisionChoice | DecisionRating,
    /,
) -> Mapping[str, Any]:
    match question:
        case DecisionCheck():
            return {
                "type": "noul",
                "instructions": question.criteria,
            }

        case DecisionChoice():
            return {
                "type": "choice",
                "instructions": question.criteria,
                "criteria": {
                    option.identifier: option.description or None for option in question.options
                },
            }

        case DecisionRating():
            return {
                "type": "score",
                "instructions": question.criteria,
                "criteria": [_level_criterion(level) for level in question.levels],
            }


def _answers(
    body: BasicValue,
    /,
    *,
    questions: Sequence[DecisionQuestion],
    model: str,
) -> Sequence[DecisionAnswer]:
    match body:
        case {"answers": dict() as answers}:
            pass

        case _:
            raise ValueError("Decision response is missing answers")

    match body:
        case {"usage": {"input_tokens": int() as input_tokens, **usage}}:
            output_tokens: BasicValue = usage.get("output_tokens")
            record_usage_metrics(
                provider="systemone",
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens if isinstance(output_tokens, int) else None,
            )

        case _:
            pass

    return tuple(
        _selection_answer(
            question,
            key=_question_key(index),
            answers=answers,
        )
        if isinstance(question, DecisionSelection)
        else _answer(
            question,
            answer=answers.get(_question_key(index)),
        )
        for index, question in enumerate(questions)
    )


def _selection_answer(
    question: DecisionSelection,
    /,
    *,
    key: str,
    answers: Mapping[str, BasicValue],
) -> DecisionSelectionAnswer:
    probabilities: dict[str, float] = {}
    for index, option in enumerate(question.options):
        match answers.get(_option_key(key, index=index)):
            case {"type": "noul", "noul": float() | int() as probability}:
                probabilities[option.identifier] = float(probability)

            case _:
                raise ValueError("Decision completed without a matching noul answer")

    return DecisionSelectionAnswer(
        identifier=question.identifier,
        selected=tuple(
            option
            for option in question.options
            if probabilities[option.identifier] >= 0.5  # noqa: PLR2004
        ),
        meta=Meta.of({"probabilities": probabilities}),
    )


def _answer(
    question: DecisionCheck | DecisionChoice | DecisionRating,
    /,
    *,
    answer: BasicValue | None,
) -> DecisionAnswer:
    # confidence is reported as the probability of the decided outcome,
    # not the concentration measure reported by the server
    match (question, answer):
        case (DecisionCheck(), {"type": "noul", "noul": float() | int() as probability}):
            return DecisionCheckAnswer(
                identifier=question.identifier,
                value=probability >= 0.5,  # noqa: PLR2004
                meta=Meta.of(
                    {
                        "probability": float(probability),
                        "confidence": float(max(probability, 1.0 - probability)),
                    }
                ),
            )

        case (
            DecisionChoice(),
            {
                "type": "choice",
                "choice": str() as choice,
                "probabilities": dict() as probabilities,
            },
        ):
            resolved: Mapping[str, float] = {
                option.identifier: _probability(probabilities.get(option.identifier))
                for option in question.options
            }
            for option in question.options:
                if option.identifier == choice:
                    return DecisionChoiceAnswer(
                        identifier=question.identifier,
                        choice=option,
                        meta=option.meta.merged_with(
                            {
                                "probabilities": resolved,
                                "confidence": resolved[choice],
                            }
                        ),
                    )

            raise ValueError("Decision completed without a matching choice answer")

        case (
            DecisionRating(),
            {
                "type": "score",
                "score": float() | int() as score,
                "probabilities": dict() as probabilities,
            },
        ):
            # probabilities are keyed by the stringified level index
            levels: Sequence[float] = tuple(
                _probability(probabilities.get(str(index))) for index in range(len(question.levels))
            )
            last: int = len(question.levels) - 1
            index: int
            if any(levels):
                index = max(range(len(levels)), key=levels.__getitem__)

            else:
                index = min(max(round(score), 0), last)

            return DecisionRatingAnswer(
                identifier=question.identifier,
                level=question.levels[index],
                meta=question.levels[index].meta.merged_with(
                    {
                        # the api scores in level positions, normalized to 0-1
                        "score": min(max(score / last, 0.0), 1.0),
                        "probabilities": {
                            level.identifier: probability
                            for level, probability in zip(question.levels, levels, strict=True)
                        },
                        "confidence": levels[index],
                    }
                ),
            )

        case _:
            raise ValueError("Decision completed without a matching answer")


def _level_criterion(
    level: DecisionOption,
    /,
) -> str:
    # score criteria are plain descriptions, the level position is its value
    if level.description:
        return f"{level.identifier}: {level.description}"

    return level.identifier


def _probability(
    value: BasicValue | None,
    /,
) -> float:
    match value:
        case float() | int() if not isinstance(value, bool):
            return float(value)

        case _:
            return 0.0
