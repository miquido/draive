from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest
from haiway import ctx
from ollama import ResponseError, SystemOneResponse

from draive.generation import (
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
)
from draive.models import ModelInputInvalid, ModelOutputFailed, ModelRateLimit
from draive.multimodal import MultimodalContent
from draive.ollama import Ollama, OllamaDecisionConfig
from draive.resources import ResourceContent

_BILLING = DecisionOption(
    identifier="billing",
    description="payment issues",
    meta={"team_id": 42},
)
_SHIPPING = DecisionOption(identifier="shipping")
_LEVELS: Sequence[DecisionOption] = (
    DecisionOption(identifier="can wait"),
    DecisionOption(identifier="today", description="within the day", meta={"sla_hours": 8}),
    DecisionOption(identifier="right now"),
)
_QUESTIONS: Sequence[DecisionQuestion] = (
    DecisionCheck(identifier="angry", criteria="Is the customer angry?"),
    DecisionChoice(
        identifier="route",
        criteria="Which team should handle this?",
        options=(_BILLING, _SHIPPING),
    ),
    DecisionSelection(
        identifier="topics",
        criteria="Which topics apply?",
        options=(_BILLING, _SHIPPING),
    ),
    DecisionRating(identifier="urgency", criteria="How urgent is this?", levels=_LEVELS),
)


def _ollama(
    response: dict[str, Any] | Exception,
    captured: dict[str, Any] | None = None,
) -> Ollama:
    async def systemone(**kwargs: Any) -> SystemOneResponse:
        if captured is not None:
            captured.update(kwargs)

        if isinstance(response, Exception):
            raise response

        return SystemOneResponse.model_validate(response)

    ollama = Ollama(features=(DecisionGeneration,))
    ollama._client = SimpleNamespace(systemone=systemone)  # pyright: ignore[reportAttributeAccessIssue]
    return ollama


@pytest.mark.asyncio
async def test_decide_sends_questions_and_maps_answers() -> None:
    captured: dict[str, Any] = {}
    ollama = _ollama(
        {
            "model": "tev1",
            "answers": {
                "q0": {"type": "noul", "noul": 0.2},
                "q1": {
                    "type": "choice",
                    "choice": "billing",
                    "probabilities": {"billing": 0.9, "shipping": 0.1},
                    "confidence": 0.5,
                },
                "q2_0": {"type": "noul", "noul": 0.8},
                "q2_1": {"type": "noul", "noul": 0.3},
                "q3": {
                    "type": "score",
                    "score": 1.3,
                    "legend": {"0": "can wait", "1": "today", "2": "right now"},
                    "probabilities": {"0": 0.1, "1": 0.5, "2": 0.4},
                    "confidence": 0.2,
                },
            },
            "usage": {"input_tokens": 12, "output_tokens": 4},
        },
        captured,
    )
    async with ctx.scope(
        "test", OllamaDecisionConfig(), DecisionGeneration(generating=ollama.decide)
    ):
        answers = await DecisionGeneration.decide(
            input="I was charged twice",
            questions=_QUESTIONS,
        )

    assert captured["model"] == "tev1"
    assert captured["state"] == "I was charged twice"
    assert {key: question.model_dump() for key, question in captured["questions"].items()} == {
        "q0": {"type": "noul", "instructions": "Is the customer angry?", "criteria": None},
        "q1": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "payment issues", "shipping": None},
        },
        "q2_0": {
            "type": "noul",
            "instructions": (
                "Which topics apply?\n\nDoes this option apply? billing: payment issues"
            ),
            "criteria": None,
        },
        "q2_1": {
            "type": "noul",
            "instructions": "Which topics apply?\n\nDoes this option apply? shipping",
            "criteria": None,
        },
        "q3": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["can wait", "today: within the day", "right now"],
        },
    }
    # confidence is the probability of the decided outcome, not the server measure
    assert answers == (
        DecisionCheckAnswer(
            identifier="angry",
            value=False,
            meta={"probability": 0.2, "confidence": 0.8},
        ),
        DecisionChoiceAnswer(
            identifier="route",
            choice=_BILLING,
            meta={
                "team_id": 42,
                "probabilities": {"billing": 0.9, "shipping": 0.1},
                "confidence": 0.9,
            },
        ),
        DecisionSelectionAnswer(
            identifier="topics",
            selected=(_BILLING,),
            meta={"probabilities": {"billing": 0.8, "shipping": 0.3}},
        ),
        DecisionRatingAnswer(
            identifier="urgency",
            level=_LEVELS[1],
            meta={
                "sla_hours": 8,
                "score": 0.65,
                "probabilities": {"can wait": 0.1, "today": 0.5, "right now": 0.4},
                "confidence": 0.5,
            },
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("input", "questions"),
    [
        ("", (DecisionCheck(identifier="q", criteria="Is it?"),)),
        (
            MultimodalContent.of(
                "picture",
                ResourceContent.of(b"png", mime_type="image/png"),
            ),
            (DecisionCheck(identifier="q", criteria="Is it?"),),
        ),
        (
            "any",
            (
                DecisionSelection(
                    identifier="q",
                    criteria="Which apply?",
                    options=tuple(DecisionOption(identifier=str(index)) for index in range(65)),
                ),
            ),
        ),
    ],
)
async def test_decide_rejects_input_unsupported_by_ollama(
    input: Any,  # noqa: A002
    questions: Sequence[DecisionQuestion],
) -> None:
    captured: dict[str, Any] = {}
    async with ctx.scope("test", OllamaDecisionConfig()):
        with pytest.raises(ModelInputInvalid):
            await _ollama({}, captured).decide(input=input, questions=questions)

    assert captured == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (ResponseError("prompt too long", 400), ModelInputInvalid),
        (ResponseError("request too large", 413), ModelInputInvalid),
        (ResponseError("busy", 503), ModelRateLimit),
        (ResponseError("model 'tev1' not found", 404), ModelOutputFailed),
        (ConnectionError("unreachable"), ModelOutputFailed),
        (
            {
                "model": "tev1",
                "answers": {
                    "q0": {"type": "choice", "choice": "a", "probabilities": {}, "confidence": 1}
                },
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
            ModelOutputFailed,
        ),
    ],
)
async def test_decide_translates_failures(
    failure: dict[str, Any] | Exception,
    expected: type[Exception],
) -> None:
    async with ctx.scope("test", OllamaDecisionConfig()):
        with pytest.raises(expected):
            await _ollama(failure).decide(
                input="any",
                questions=(DecisionCheck(identifier="q", criteria="Is it?"),),
            )
