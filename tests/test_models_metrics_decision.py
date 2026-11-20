import json
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from haiway import ctx

from draive import (
    DecisionCheck,
    DecisionChoice,
    DecisionOption,
    DecisionRating,
    DecisionSelection,
    GenerativeModel,
    generate_completion_decisions,
)
from draive.models import (
    ModelContext,
    ModelOutputChunk,
    ModelTools,
    record_decision_invocation,
    record_decision_metrics,
)
from draive.multimodal import TextContent

_OPTIONS = (DecisionOption(identifier="a"), DecisionOption(identifier="b"))


def _counters(records: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for record in records:
        metric: str = record.get("metric", "")
        if metric.startswith("decision."):
            totals[metric] = totals.get(metric, 0) + record["value"]

    return totals


@pytest.fixture
def records(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    recorded: list[dict[str, Any]] = []
    monkeypatch.setattr(ctx, "record_info", lambda **kwargs: recorded.append(kwargs))
    return recorded


@pytest.mark.asyncio
async def test_invocation_records_question_identifiers_and_kinds(
    records: list[dict[str, Any]],
) -> None:
    async with ctx.scope("test"):
        record_decision_invocation(
            provider="p",
            model="m",
            questions=(
                DecisionCheck(identifier="c", criteria="x"),
                DecisionChoice(identifier="o", criteria="x", options=_OPTIONS),
                DecisionSelection(identifier="s", criteria="x", options=_OPTIONS),
                DecisionRating(identifier="r", criteria="x", levels=_OPTIONS),
            ),
            extra="value",
        )

    assert records == [
        {
            "attributes": {
                "decision.provider": "p",
                "decision.model": "m",
                "decision.questions": ["c", "o", "s", "r"],
                "decision.question_types": ["check", "choice", "selection", "rating"],
                "decision.extra": "value",
            }
        }
    ]


@pytest.mark.asyncio
async def test_invocation_model_is_optional(records: list[dict[str, Any]]) -> None:
    async with ctx.scope("test"):
        record_decision_invocation(provider="p", questions=())

    assert records[0]["attributes"]["decision.model"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"questions": 3, "requests": 1}, {"decision.questions": 3, "decision.requests": 1}),
        ({"questions": 0}, {"decision.questions": 0}),
        ({"requests": 2}, {"decision.requests": 2}),
        ({}, {}),
    ],
)
async def test_metrics_record_only_given_counters(
    records: list[dict[str, Any]],
    arguments: dict[str, int],
    expected: dict[str, int],
) -> None:
    async with ctx.scope("test"):
        record_decision_metrics(provider="p", model="m", **arguments)

    assert {record["metric"]: record["value"] for record in records} == expected
    assert all(
        record["attributes"] == {"decision.provider": "p", "decision.model": "m"}
        and record["kind"] == "counter"
        and record["unit"] == "count"
        for record in records
    )


@pytest.mark.asyncio
async def test_metrics_omit_the_model_when_not_given(records: list[dict[str, Any]]) -> None:
    async with ctx.scope("test"):
        record_decision_metrics(provider="p", questions=1)

    assert records[0]["attributes"] == {"decision.provider": "p"}


@pytest.mark.asyncio
async def test_completion_decisions_record_a_request_for_each_question(
    records: list[dict[str, Any]],
) -> None:
    async def generating(
        *,
        instructions: str,
        tools: ModelTools,
        context: ModelContext,
        output: Any,
        **extra: Any,
    ) -> AsyncGenerator[ModelOutputChunk]:
        yield TextContent.of(json.dumps({"reasoning": "-", "value": True}))

    async with ctx.scope("test", GenerativeModel(generating=generating)):
        await generate_completion_decisions(
            input="text",
            questions=(
                DecisionCheck(identifier="one", criteria="x"),
                DecisionCheck(identifier="two", criteria="x"),
            ),
        )

    invocation = next(
        record["attributes"]
        for record in records
        if "decision.provider" in record.get("attributes", {}) and "metric" not in record
    )
    assert invocation["decision.provider"] == "completion"
    assert invocation["decision.model"] is None
    assert invocation["decision.questions"] == ["one", "two"]
    assert _counters(records) == {"decision.questions": 2, "decision.requests": 2}


@pytest.mark.asyncio
async def test_completion_decisions_do_not_count_failed_batches(
    records: list[dict[str, Any]],
) -> None:
    async def generating(**_: Any) -> AsyncGenerator[ModelOutputChunk]:
        raise RuntimeError("model down")
        yield TextContent.of("")

    async with ctx.scope("test", GenerativeModel(generating=generating)):
        with pytest.raises(Exception, match="model down"):
            await generate_completion_decisions(
                input="text",
                questions=(DecisionCheck(identifier="one", criteria="x"),),
            )

    assert not [record for record in records if record.get("metric", "").startswith("decision.")]


@pytest.mark.asyncio
async def test_completion_decisions_keep_counters_of_requests_completed_before_a_failure(
    records: list[dict[str, Any]],
) -> None:
    async def generating(
        *,
        instructions: str,
        tools: ModelTools,
        context: ModelContext,
        output: Any,
        **extra: Any,
    ) -> AsyncGenerator[ModelOutputChunk]:
        if "fail" in instructions:
            yield TextContent.of("not a json")

        else:
            yield TextContent.of(json.dumps({"reasoning": "-", "value": True}))

    async with ctx.scope("test", GenerativeModel(generating=generating)):
        with pytest.raises(Exception):  # noqa: B017
            await generate_completion_decisions(
                input="text",
                questions=(
                    DecisionCheck(identifier="good", criteria="fine"),
                    DecisionCheck(identifier="bad", criteria="fail"),
                ),
            )

    assert _counters(records) == {"decision.questions": 1, "decision.requests": 1}
