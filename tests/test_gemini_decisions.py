import json
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from google.genai._gaos.utils.serializers import unmarshal_json
from google.genai._interactions import APIStatusError
from google.genai.decisions import CreateDecisionRequest, CreateDecisionResponse
from haiway import ctx

from draive.gemini import Gemini, GeminiDecisionConfig
from draive.gemini.decisions import GeminiDecisions
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
from draive.models import ModelInputInvalid, ModelOutputFailed, ModelQuotaLimit, ModelRateLimit
from draive.multimodal import ArtifactContent, MultimodalContent, TextContent
from draive.resources import ResourceContent, ResourceReference

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
    DecisionCheck(
        identifier="angry",
        criteria="Is the customer angry?",
    ),
    DecisionChoice(
        identifier="route",
        criteria="Which team should handle this?",
        options=(_BILLING, _SHIPPING),
    ),
    DecisionRating(
        identifier="urgency",
        criteria="How urgent is this?",
        levels=_LEVELS,
    ),
)
_CHECK: Sequence[DecisionQuestion] = (DecisionCheck(identifier="angry", criteria="Is it?"),)


def _choice(*identifiers: str) -> Sequence[DecisionQuestion]:
    return (
        DecisionChoice(
            identifier="route",
            criteria="Pick",
            options=tuple(DecisionOption(identifier=identifier) for identifier in identifiers),
        ),
    )


def _decisions(
    response: Mapping[str, Any] | Exception,
    captured: dict[str, Any] | None = None,
    *,
    vertexai: bool | None = None,
) -> GeminiDecisions:
    async def create(**kwargs: Any) -> CreateDecisionResponse:
        if captured is not None:
            captured.update(kwargs)

        if isinstance(response, Exception):
            raise response

        # parsed the same way the sdk parses responses
        return unmarshal_json(json.dumps(response), CreateDecisionResponse)

    decisions = GeminiDecisions(api_key="test", vertexai=vertexai, http_options=None)
    decisions._client = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        vertexai=bool(vertexai),
        aio=SimpleNamespace(decisions=SimpleNamespace(create=create)),
    )
    return decisions


def _status_error(status: int, body: Any) -> APIStatusError:
    return APIStatusError(
        "failed",
        response=httpx.Response(
            status,
            request=httpx.Request("POST", "https://example.com/v1beta/decisions"),
        ),
        body=body,
    )


@pytest.mark.asyncio
async def test_decide_builds_request_and_maps_answers() -> None:
    captured: dict[str, Any] = {}
    decisions = _decisions(
        {
            "answers": {
                "q0": {"type": "noul", "noul": 0.82, "confidence": 0.82},
                "q1": {
                    "type": "choice",
                    "choice": "billing",
                    "probabilities": {"billing": 0.9, "shipping": 0.1},
                    "confidence": 0.9,
                },
                "q2": {
                    "type": "score",
                    "score": 1.3,
                    "legend": {"0": "can wait", "1": "today", "2": "right now"},
                    "probabilities": {"0": 0.1, "1": 0.5, "2": 0.4},
                    "confidence": 0.5,
                },
            },
            "usage": {"total_input_tokens": 42},
        },
        captured,
    )

    async with ctx.scope(
        "test",
        GeminiDecisionConfig(),
        DecisionGeneration(generating=decisions.decide),
    ):
        answers = await DecisionGeneration.decide(
            input=MultimodalContent.of(
                TextContent.of("I was charged twice"),
                ResourceContent.of(b"png", mime_type="image/png"),
                ResourceReference.of("https://example.com/a.pdf", mime_type="application/pdf"),
                ArtifactContent.of(
                    DecisionCheck(identifier="hidden", criteria="hidden"),
                    hidden=True,
                ),
            ),
            questions=_QUESTIONS,
        )

    assert captured["model"] == "gemini-decision-1"
    assert captured["api_version"] == "v1beta"
    assert captured["input"] == [
        {"type": "text", "text": "I was charged twice"},
        {"type": "image", "data": "cG5n", "mime_type": "image/png"},
        {"type": "document", "uri": "https://example.com/a.pdf", "mime_type": "application/pdf"},
    ]
    assert captured["questions"] == {
        "q0": {
            "type": "noul",
            "instructions": "Is the customer angry?",
        },
        "q1": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "payment issues", "shipping": ""},
        },
        "q2": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["can wait", "today: within the day", "right now"],
        },
    }
    assert answers == (
        DecisionCheckAnswer(
            identifier="angry",
            value=True,
            meta={"probability": 0.82, "confidence": 0.82},
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
        ("", _CHECK),
        (ResourceContent.of(b"{}", mime_type="application/json"), _CHECK),
        (ResourceReference.of("gs://bucket/a.png", mime_type="image/png"), _CHECK),
        ("any", _choice("a\tb", "b")),
        ("any", _choice("a" * 129, "b")),
        (
            "any",
            (
                DecisionRating(
                    identifier="rate",
                    criteria="Rate",
                    levels=tuple(DecisionOption(identifier=str(index)) for index in range(256)),
                ),
            ),
        ),
    ],
)
async def test_decide_rejects_input_unsupported_by_the_api(
    input: Any,  # noqa: A002
    questions: Sequence[DecisionQuestion],
) -> None:
    async with ctx.scope("test", GeminiDecisionConfig()):
        with pytest.raises(ModelInputInvalid):
            await _decisions({}).decide(input=input, questions=questions)


@pytest.mark.asyncio
async def test_decide_rejects_vertexai() -> None:
    async with ctx.scope("test", GeminiDecisionConfig()):
        with pytest.raises(ModelInputInvalid):
            await _decisions({}, vertexai=True).decide(
                input="any",
                questions=_CHECK,
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (_status_error(429, None), ModelRateLimit),
        (
            _status_error(
                429,
                {
                    "error": {
                        "details": [
                            {
                                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                                "violations": [{"quotaValue": "0"}],
                            }
                        ]
                    }
                },
            ),
            ModelQuotaLimit,
        ),
        (_status_error(503, None), ModelRateLimit),
        (_status_error(400, {"error": {"code": "safety"}}), ModelOutputFailed),
        (_status_error(500, "boom"), ModelOutputFailed),
        ({"prompt_feedback": {"block_reason": "safety"}}, ModelOutputFailed),
        ({"answers": {}}, ModelOutputFailed),
        (
            {"answers": {"q0": {"type": "choice", "choice": "a", "probabilities": {}}}},
            ModelOutputFailed,
        ),
        ({"answers": {"q0": {"type": "unknown"}}}, ModelOutputFailed),
    ],
)
async def test_decide_translates_failures(
    response: Mapping[str, Any] | Exception,
    expected: type[Exception],
) -> None:
    async with ctx.scope("test", GeminiDecisionConfig()):
        with pytest.raises(expected) as error:
            await _decisions(response).decide(
                input="any",
                questions=_CHECK,
            )

    if isinstance(response, Mapping) and "prompt_feedback" in response:
        assert "safety" in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mime_type",
    ["image/png", "image/webp", "audio/wav", "video/mp4", "application/pdf"],
)
async def test_decide_request_is_accepted_by_the_sdk_request_model(mime_type: str) -> None:
    captured: dict[str, Any] = {}
    async with ctx.scope("test", GeminiDecisionConfig()):
        await _decisions(
            {
                "answers": {
                    "q0": {"type": "noul", "noul": 0.2, "confidence": 0.8},
                    "q1": {
                        "type": "choice",
                        "choice": "shipping",
                        "probabilities": {"billing": 0.4, "shipping": 0.6},
                        "confidence": 0.6,
                    },
                    "q2": {
                        "type": "score",
                        "score": 0.0,
                        "legend": {},
                        "probabilities": {"0": 1.0},
                        "confidence": 1.0,
                    },
                },
            },
            captured,
        ).decide(
            input=MultimodalContent.of(
                TextContent.of("any"),
                ResourceContent.of(b"data", mime_type=mime_type),
                ResourceReference.of("https://example.com/data", mime_type=mime_type),
            ),
            questions=_QUESTIONS,
        )

    request = CreateDecisionRequest.model_validate(
        {
            "model": captured["model"],
            "input": captured["input"],
            "questions": captured["questions"],
        }
    )
    assert len(request.questions) == len(_QUESTIONS)


@pytest.mark.asyncio
async def test_decide_resolves_selection_as_noul_question_per_option() -> None:
    captured: dict[str, Any] = {}
    topics = DecisionSelection(
        identifier="topics",
        criteria="Which topics does the ticket mention?",
        options=(_BILLING, _SHIPPING, DecisionOption(identifier="refund")),
    )
    async with ctx.scope("test", GeminiDecisionConfig()):
        answers = await _decisions(
            {
                "answers": {
                    "q0": {"type": "noul", "noul": 0.3, "confidence": 0.7},
                    "q1_0": {"type": "noul", "noul": 0.95, "confidence": 0.95},
                    "q1_1": {"type": "noul", "noul": 0.1, "confidence": 0.9},
                    "q1_2": {"type": "noul", "noul": 0.6, "confidence": 0.6},
                },
            },
            captured,
        ).decide(
            input="I was charged twice",
            questions=(
                DecisionCheck(identifier="angry", criteria="Is the customer angry?"),
                topics,
            ),
        )

    assert captured["questions"] == {
        "q0": {"type": "noul", "instructions": "Is the customer angry?"},
        "q1_0": {
            "type": "noul",
            "instructions": "Which topics does the ticket mention?\n\nDoes this option apply?"
            " billing: payment issues",
        },
        "q1_1": {
            "type": "noul",
            "instructions": "Which topics does the ticket mention?\n\nDoes this option apply?"
            " shipping",
        },
        "q1_2": {
            "type": "noul",
            "instructions": "Which topics does the ticket mention?\n\nDoes this option apply?"
            " refund",
        },
    }
    CreateDecisionRequest.model_validate(
        {
            "model": captured["model"],
            "input": captured["input"],
            "questions": captured["questions"],
        }
    )
    assert answers[1] == DecisionSelectionAnswer(
        identifier="topics",
        selected=(_BILLING, DecisionOption(identifier="refund")),
        meta={"probabilities": {"billing": 0.95, "shipping": 0.1, "refund": 0.6}},
    )


@pytest.mark.asyncio
async def test_decide_counts_selection_options_against_question_limit() -> None:
    async with ctx.scope("test", GeminiDecisionConfig()):
        with pytest.raises(ModelInputInvalid):
            await _decisions({}).decide(
                input="any",
                questions=(
                    DecisionSelection(
                        identifier="topics",
                        criteria="Which apply?",
                        options=tuple(
                            DecisionOption(identifier=str(index)) for index in range(1001)
                        ),
                    ),
                ),
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("vertexai", "features", "expected"),
    [
        (False, None, True),
        (True, None, False),
        (True, (DecisionGeneration,), True),
    ],
)
async def test_client_provides_decisions_unless_defaulted_on_vertexai(
    vertexai: bool,
    features: Sequence[Any] | None,
    expected: bool,
) -> None:
    client = Gemini(api_key="test", features=features)

    async def initialize() -> None:
        client._client = SimpleNamespace(vertexai=vertexai)  # pyright: ignore[reportAttributeAccessIssue]

    client._initialize_client = initialize  # pyright: ignore[reportAttributeAccessIssue]
    async with ctx.scope("test"):
        state = await client.__aenter__()

    assert any(isinstance(element, DecisionGeneration) for element in state) is expected
