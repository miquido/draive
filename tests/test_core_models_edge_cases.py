import math
from typing import Any

import pytest
from haiway import State, ctx

from draive.models import (
    ModelInput,
    ModelOutput,
    ModelQuotaLimit,
    ModelRateLimit,
    ModelReasoning,
    ModelReasoningChunk,
    ModelSessionEvent,
    ModelToolRequest,
    ModelToolResponse,
    model_output_blocks,
)
from draive.models.metrics import model_rate_limit
from draive.multimodal import MultimodalContent, TextContent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("retry_after", "expected"),
    [
        ("2", 2.0),
        (" 2 ", 2.0),
        (3, 3.0),
        (1.5, 1.5),
        (0, 0.0),
    ],
)
async def test_model_rate_limit_reads_retry_after(
    retry_after: str | float,
    expected: float,
) -> None:
    async with ctx.scope("test"):
        limit = model_rate_limit(provider="p", model="m", retry_after=retry_after)

    assert isinstance(limit, ModelRateLimit)
    assert limit.retry_after == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_after", [None, "", "soon", "Wed, 99 Foo 2026 07:28:00 GMT"])
async def test_model_rate_limit_falls_back_to_short_random_delay(retry_after: str | None) -> None:
    async with ctx.scope("test"):
        limit = model_rate_limit(provider="p", model="m", retry_after=retry_after)

    assert isinstance(limit, ModelRateLimit)
    assert 0.3 <= limit.retry_after <= 3.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "retry_after",
    ["nan", "inf", "-1", "1e400", float("nan"), float("inf"), -3.0],
)
async def test_model_rate_limit_delay_is_finite_and_not_negative(
    retry_after: str | float,
) -> None:
    async with ctx.scope("test"):
        limit = model_rate_limit(provider="p", model="m", retry_after=retry_after)

    assert isinstance(limit, ModelRateLimit)
    assert math.isfinite(limit.retry_after)
    assert limit.retry_after >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_code", "quota_limit"),
    [("insufficient_quota", None), ("billing_hard_limit_reached", None), (None, 0)],
)
async def test_model_rate_limit_recognizes_exhausted_quota(
    error_code: str | None,
    quota_limit: int | None,
) -> None:
    async with ctx.scope("test"):
        limit = model_rate_limit(
            provider="p",
            model="m",
            retry_after="5",
            error_code=error_code,
            quota_limit=quota_limit,
        )

    assert isinstance(limit, ModelQuotaLimit)
    assert not isinstance(limit, ModelRateLimit)


def _content(text: str) -> MultimodalContent:
    return MultimodalContent.of(text)


@pytest.mark.parametrize(
    "event",
    [
        ModelSessionEvent.turn_started(),
        ModelSessionEvent.turn_interrupted(meta={"reason": "user"}),
        ModelSessionEvent.context_updated(
            (ModelInput.of(_content("hi")), ModelOutput.of(_content("yo")))
        ),
        ModelSessionEvent.of("custom", meta={"a": 1}),
    ],
)
def test_session_events_without_state_payload_round_trip_json(event: ModelSessionEvent) -> None:
    assert ModelSessionEvent.from_json(event.to_json()) == event


@pytest.mark.parametrize(
    "content",
    [
        ModelInput.of(_content("hi")),
        ModelOutput.of(_content("yo")),
    ],
)
def test_turn_completed_event_round_trips_json(content: ModelInput | ModelOutput) -> None:
    event = ModelSessionEvent.turn_completed(content)

    assert ModelSessionEvent.from_json(event.to_json()) == event


def test_model_context_round_trips_json() -> None:
    request = ModelToolRequest.of(
        "a",
        tool="t",
        arguments={"x": [1, 2, {"y": None}]},
        handling="output",
        meta={"k": "v"},
    )
    response = ModelToolResponse.of("a", tool="t", status="error", content="boom")
    elements: tuple[State, ...] = (
        ModelInput.of(_content("hi"), response, meta={"m": 1}),
        ModelOutput.of(_content("x"), ModelReasoning.of("think", meta={"sig": "s"}), request),
    )

    for element in elements:
        assert type(element).from_json(element.to_json()) == element


def test_tool_request_arguments_are_isolated_from_the_source_mapping() -> None:
    source: dict[str, Any] = {"items": [1, 2], "nested": {"a": 1}}
    request = ModelToolRequest.of("1", tool="t", arguments=source)
    source["items"].append(3)
    source["nested"]["a"] = 2
    source["new"] = 1

    assert request.arguments == {"items": (1, 2), "nested": {"a": 1}}


def test_reasoning_blocks_split_on_final_chunks_and_keep_identity() -> None:
    chunks = [
        ModelReasoningChunk.of(TextContent.of("a"), meta={"sig": "1"}),
        ModelReasoningChunk.of(TextContent.of("b"), final=True, meta={"sig": "2"}),
        ModelReasoningChunk.of(TextContent.of("c"), meta={"x": "1"}),
    ]

    blocks = ModelReasoning.blocks(chunks)

    assert [(block.reasoning.to_str(), dict(block.meta)) for block in blocks] == [
        ("ab", {"sig": "2"}),
        ("c", {"x": "1"}),
    ]


def test_output_blocks_keep_stream_order_across_kinds() -> None:
    request = ModelToolRequest.of("1", tool="t")
    blocks = model_output_blocks(
        [
            ModelReasoningChunk.of(TextContent.of("r1")),
            TextContent.of("a"),
            TextContent.of("b"),
            ModelReasoningChunk.of(TextContent.of("r2")),
            request,
            TextContent.of("c"),
        ]
    )

    assert [type(block).__name__ for block in blocks] == [
        "ModelReasoning",
        "MultimodalContent",
        "ModelReasoning",
        "ModelToolRequest",
        "MultimodalContent",
    ]
    assert blocks[1] == MultimodalContent.of("a", "b")


def test_output_blocks_of_nothing_is_empty() -> None:
    assert model_output_blocks([]) == ()
