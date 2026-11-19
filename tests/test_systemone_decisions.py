import json
from collections.abc import Mapping, Sequence
from typing import Any

import httpx2
import pytest
from haiway import ctx

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
    DecisionSelectionAnswer,
)
from draive.models import ModelInputInvalid, ModelOutputFailed, ModelRateLimit
from draive.multimodal import ArtifactContent, Multimodal, MultimodalContent, TextContent
from draive.resources import ResourceContent, ResourceReference
from draive.systemone import SystemOne, SystemOneConfig

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

_ANSWERS: Mapping[str, Any] = {
    "q0": {"type": "noul", "noul": 0.2},
    "q1": {
        "type": "choice",
        "choice": "billing",
        "probabilities": {"billing": 0.9, "shipping": 0.1},
        "confidence": 0.8,
    },
    "q2": {
        "type": "score",
        "score": 1.3,
        "legend": {"0": "can wait", "1": "today", "2": "right now"},
        "probabilities": {"0": 0.1, "1": 0.5, "2": 0.4},
        "confidence": 0.3,
    },
}

_CHECK_ANSWER: Mapping[str, Any] = {"answers": {"q0": {"type": "noul", "noul": 0.7}}}


def _client(
    status: int,
    body: Any,
    captured: list[httpx2.Request] | None = None,
    **arguments: Any,
) -> SystemOne:
    def handle(request: httpx2.Request) -> httpx2.Response:
        if captured is not None:
            captured.append(request)

        return httpx2.Response(
            status,
            content=body if isinstance(body, bytes) else json.dumps(body).encode(),
        )

    return SystemOne(transport=httpx2.MockTransport(handle), **arguments)


@pytest.mark.asyncio
async def test_decide_sends_request_and_maps_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SYSTEMONE_SERVER_URL", raising=False)
    monkeypatch.delenv("SYSTEMONE_API_KEY", raising=False)
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret")
    captured: list[httpx2.Request] = []
    async with ctx.scope(
        "test",
        disposables=(
            _client(
                200,
                {
                    "model": "jev-1.13.0",
                    "answers": _ANSWERS,
                    "usage": {"input_tokens": 12, "output_tokens": 3},
                },
                captured,
            ),
        ),
    ):
        answers = await DecisionGeneration.decide(
            input="I was charged twice",
            questions=_QUESTIONS,
        )

    assert len(captured) == 1
    request = captured[0]
    assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
    assert request.headers["Authorization"] == "Bearer secret"
    assert "TypeSafe-Danger-Zone-Images" not in request.headers
    assert json.loads(request.content) == {
        "model": "jev-latest",
        "state": "I was charged twice",
        "questions": {
            "q0": {
                "type": "noul",
                "instructions": "Is the customer angry?",
            },
            "q1": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"billing": "payment issues", "shipping": None},
            },
            "q2": {
                "type": "score",
                "instructions": "How urgent is this?",
                "criteria": ["can wait", "today: within the day", "right now"],
            },
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
async def test_decide_sends_images_as_assets() -> None:
    captured: list[httpx2.Request] = []
    async with ctx.scope(
        "test",
        SystemOneConfig(model="jev-preview"),
        disposables=(
            _client(
                200,
                _CHECK_ANSWER,
                captured,
                server_url="https://systemone.local",
                api_key="secret",
                headers={"TypeSafe-Danger-Zone-Long-Context": "true"},
            ),
        ),
    ):
        await DecisionGeneration.check(
            input=MultimodalContent.of(
                TextContent.of("before"),
                ResourceContent.of(b"png", mime_type="image/png"),
                TextContent.of("between"),
                ResourceContent.of(b"jpg", mime_type="image/jpeg"),
                ArtifactContent.of(
                    DecisionCheck(identifier="hidden", criteria="hidden"),
                    hidden=True,
                ),
            ),
            criteria="Is it?",
        )

    request = captured[0]
    assert request.headers["TypeSafe-Danger-Zone-Images"] == "true"
    assert request.headers["TypeSafe-Danger-Zone-Long-Context"] == "true"
    body = json.loads(request.content)
    assert body["model"] == "jev-preview"
    first, second = body["assets"].keys()
    assert body["state"] == f"before\n{first}\nbetween\n{second}"
    assert body["assets"] == {
        first: {"type": "image", "format": "png", "data_base64": "cG5n"},
        second: {"type": "image", "format": "jpeg", "data_base64": "anBn"},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("input", "state"),
    [
        (
            MultimodalContent.of(
                TextContent.of("page"),
                ResourceContent.of(b"png", mime_type="image/png"),
            ),
            {"text": "page", "image": "data:image/png;base64,cG5n"},
        ),
        (
            ResourceContent.of(b"jpg", mime_type="image/webp"),
            {"image": "data:image/webp;base64,anBn"},
        ),
        ("text only", "text only"),
    ],
)
async def test_decide_sends_inline_image_within_state(
    input: Multimodal,  # noqa: A002
    state: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYSTEMONE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    captured: list[httpx2.Request] = []
    async with ctx.scope(
        "test",
        disposables=(
            _client(
                200,
                _CHECK_ANSWER,
                captured,
                server_url="http://localhost:3000",
                images="inline",
            ),
        ),
    ):
        await DecisionGeneration.check(input=input, criteria="Is it?")

    request = captured[0]
    assert str(request.url) == "http://localhost:3000/v1/systemone"
    assert "Authorization" not in request.headers
    assert "TypeSafe-Danger-Zone-Images" not in request.headers
    body = json.loads(request.content)
    assert body["state"] == state
    assert "assets" not in body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("images", "input"),
    [
        ("assets", ""),
        ("inline", ""),
        ("assets", ResourceContent.of(b"webp", mime_type="image/webp")),
        ("assets", ResourceContent.of(b"%PDF", mime_type="application/pdf")),
        ("inline", ResourceContent.of(b"%PDF", mime_type="application/pdf")),
        (
            "inline",
            MultimodalContent.of(
                ResourceContent.of(b"a", mime_type="image/png"),
                ResourceContent.of(b"b", mime_type="image/png"),
            ),
        ),
        ("assets", ResourceReference.of("https://example.com/a.png", mime_type="image/png")),
        ("inline", ResourceReference.of("https://example.com/a.png", mime_type="image/png")),
    ],
)
async def test_decide_rejects_unsupported_input(
    images: Any,
    input: Multimodal,  # noqa: A002
) -> None:
    captured: list[httpx2.Request] = []
    async with ctx.scope("test", disposables=(_client(200, {}, captured, images=images),)):
        with pytest.raises(ModelInputInvalid):
            await DecisionGeneration.check(input=input, criteria="Is it?")

    assert captured == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "body", "expected", "message"),
    [
        (
            422,
            {"detail": [{"loc": ["body", "questions"], "msg": "bad questions", "type": "x"}]},
            ModelInputInvalid,
            "bad questions",
        ),
        (400, {"detail": "state too long"}, ModelInputInvalid, "state too long"),
        (
            422,
            {"error": {"code": 422, "message": "missing state"}},
            ModelInputInvalid,
            "missing state",
        ),
        (429, b"", ModelRateLimit, None),
        (503, b"", ModelRateLimit, None),
        (
            401,
            {"error": {"code": 401, "message": "missing token"}},
            ModelOutputFailed,
            "missing token",
        ),
        (402, {"detail": "payment required"}, ModelOutputFailed, "payment required"),
        (500, b"boom", ModelOutputFailed, "boom"),
        (200, b"not json", ModelOutputFailed, None),
        (200, {"answers": {}}, ModelOutputFailed, None),
        (200, {"answers": {"q0": {"type": "choice", "choice": "a"}}}, ModelOutputFailed, None),
        (200, {}, ModelOutputFailed, None),
    ],
)
async def test_decide_translates_failures(
    status: int,
    body: Any,
    expected: type[Exception],
    message: str | None,
) -> None:
    async with ctx.scope("test", disposables=(_client(status, body),)):
        with pytest.raises(expected) as error:
            await DecisionGeneration.check(input="any", criteria="Is it?")

    if message is not None:
        assert message in str(error.value)


@pytest.mark.asyncio
async def test_decide_resolves_selection_as_noul_question_per_option() -> None:
    captured: list[httpx2.Request] = []
    refund = DecisionOption(identifier="refund")
    async with ctx.scope(
        "test",
        disposables=(
            _client(
                200,
                {
                    "answers": {
                        "q0_0": {"type": "noul", "noul": 0.9},
                        "q0_1": {"type": "noul", "noul": 0.2},
                        "q0_2": {"type": "noul", "noul": 0.5},
                    },
                },
                captured,
                api_key="secret",
            ),
        ),
    ):
        answer = await DecisionGeneration.select(
            input="I was charged twice",
            criteria="Which topics apply?",
            options=(_BILLING, _SHIPPING, refund),
        )

    assert json.loads(captured[0].content)["questions"] == {
        "q0_0": {
            "type": "noul",
            "instructions": (
                "Which topics apply?\n\nDoes this option apply? billing: payment issues"
            ),
        },
        "q0_1": {
            "type": "noul",
            "instructions": "Which topics apply?\n\nDoes this option apply? shipping",
        },
        "q0_2": {
            "type": "noul",
            "instructions": "Which topics apply?\n\nDoes this option apply? refund",
        },
    }
    # selected options keep their own meta, answer meta carries backend values only
    assert answer == DecisionSelectionAnswer(
        identifier="selection",
        selected=(_BILLING, refund),
        meta={"probabilities": {"billing": 0.9, "shipping": 0.2, "refund": 0.5}},
    )


@pytest.mark.asyncio
async def test_decide_sends_images_as_shared_list() -> None:
    captured: list[httpx2.Request] = []
    async with ctx.scope(
        "test",
        SystemOneConfig(model="tev1:0.8b"),
        disposables=(
            _client(
                200,
                _CHECK_ANSWER,
                captured,
                server_url="http://localhost:11434",
                images="images",
            ),
        ),
    ):
        await DecisionGeneration.check(
            input=MultimodalContent.of(
                TextContent.of("before"),
                ResourceContent.of(b"png", mime_type="image/png"),
                TextContent.of("after"),
                ResourceContent.of(b"webp", mime_type="image/webp"),
            ),
            criteria="Is it?",
        )

    request = captured[0]
    assert str(request.url) == "http://localhost:11434/v1/systemone"
    assert "TypeSafe-Danger-Zone-Images" not in request.headers
    body = json.loads(request.content)
    assert body["model"] == "tev1:0.8b"
    assert body["state"] == "before\nafter"
    assert body["images"] == ["cG5n", "d2VicA=="]
    assert "assets" not in body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input",
    [
        ResourceContent.of(b"png", mime_type="image/png"),
        ResourceContent.of(b"%PDF", mime_type="application/pdf"),
    ],
)
async def test_decide_rejects_input_unsupported_by_shared_images(
    input: Multimodal,  # noqa: A002
) -> None:
    captured: list[httpx2.Request] = []
    async with ctx.scope("test", disposables=(_client(200, {}, captured, images="images"),)):
        with pytest.raises(ModelInputInvalid):
            await DecisionGeneration.check(input=input, criteria="Is it?")

    assert captured == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (400, ModelInputInvalid),
        (413, ModelInputInvalid),
        (404, ModelOutputFailed),
    ],
)
async def test_decide_translates_ollama_failures(
    status: int,
    expected: type[Exception],
) -> None:
    async with ctx.scope(
        "test",
        disposables=(_client(status, {"error": "model rejected the request"}, images="images"),),
    ):
        with pytest.raises(expected) as error:
            await DecisionGeneration.check(input="any", criteria="Is it?")

    assert "model rejected the request" in str(error.value)
