import asyncio
from collections.abc import Sequence
from typing import Any

import pytest
from haiway import ctx

from draive import (
    ModelToolRequest,
    ModelToolResponse,
    ModelTools,
    Toolbox,
    tool,
)


def _request(
    identifier: str,
    tool_name: str,
    /,
    **arguments: Any,
) -> ModelToolRequest:
    return ModelToolRequest.of(identifier, tool=tool_name, arguments=arguments)


async def _responses(
    toolbox: Toolbox,
    requests: Sequence[ModelToolRequest],
) -> Sequence[ModelToolResponse]:
    return [
        chunk async for chunk in toolbox.handle(requests) if isinstance(chunk, ModelToolResponse)
    ]


@tool
async def _add(a: int, b: int = 2) -> str:
    return str(a + b)


@tool
async def _returns_number() -> Any:
    return 5


@tool
async def _returns_none() -> Any:
    return None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {"a": "x"},
        {},
        {"a": 1, "unexpected": 3},
        {"a": [1]},
    ],
)
async def test_invalid_arguments_become_error_responses(arguments: dict[str, Any]) -> None:
    async with ctx.scope("test"):
        responses = await _responses(Toolbox.of(_add), [_request("1", "_add", **arguments)])

    assert [(response.identifier, response.status) for response in responses] == [("1", "error")]


@pytest.mark.asyncio
async def test_valid_arguments_apply_defaults() -> None:
    async with ctx.scope("test"):
        responses = await _responses(Toolbox.of(_add), [_request("1", "_add", a=1)])

    assert responses[0].status == "success"
    assert responses[0].content.to_str() == "3"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["_returns_number", "_returns_none"])
async def test_non_multimodal_result_becomes_error_response(tool_name: str) -> None:
    async with ctx.scope("test"):
        responses = await _responses(
            Toolbox.of(_returns_number, _returns_none),
            [_request("1", tool_name)],
        )

    assert [(response.identifier, response.status) for response in responses] == [("1", "error")]


@pytest.mark.asyncio
async def test_many_concurrent_requests_all_receive_a_response() -> None:
    @tool
    async def delayed(index: int) -> str:
        await asyncio.sleep(0.0005 * (index % 7))
        return str(index)

    requests = [_request(str(index), "delayed", index=index) for index in range(300)]
    async with ctx.scope("test"):
        responses = await _responses(Toolbox.of(delayed), requests)

    assert sorted(int(response.identifier) for response in responses) == list(range(300))
    assert all(response.status == "success" for response in responses)
    assert all(response.content.to_str() == response.identifier for response in responses)


@pytest.mark.asyncio
async def test_unknown_and_failing_requests_do_not_block_others() -> None:
    @tool
    async def failing() -> str:
        raise RuntimeError("boom")

    async with ctx.scope("test"):
        responses = await _responses(
            Toolbox.of(_add, failing),
            [
                _request("1", "missing"),
                _request("2", "failing"),
                _request("3", "_add", a=1),
            ],
        )

    assert {response.identifier: response.status for response in responses} == {
        "1": "error",
        "2": "error",
        "3": "success",
    }


@pytest.mark.asyncio
async def test_error_response_does_not_leak_exception_details() -> None:
    @tool
    async def failing() -> str:
        raise RuntimeError("super-secret-token")

    async with ctx.scope("test"):
        responses = await _responses(Toolbox.of(failing), [_request("1", "failing")])

    assert "super-secret-token" not in responses[0].content.to_str()


@pytest.mark.asyncio
async def test_tool_cancelled_from_within_still_answers_every_request() -> None:
    inner = asyncio.get_running_loop().create_future()
    inner.cancel()

    @tool
    async def cancelled_inside() -> str:
        await inner  # raises CancelledError without the caller being cancelled
        return "unreachable"

    @tool
    async def regular() -> str:
        await asyncio.sleep(0.01)
        return "ok"

    answered: set[str] = set()
    cancelled: bool = False
    async with ctx.scope("test"):
        try:
            async for chunk in Toolbox.of(cancelled_inside, regular).handle(
                [_request("1", "cancelled_inside"), _request("2", "regular")]
            ):
                if isinstance(chunk, ModelToolResponse):
                    answered.add(chunk.identifier)

        except asyncio.CancelledError:
            cancelled = True

    # either the failure is surfaced or every request gets a response, never a silent drop
    assert cancelled or answered == {"1", "2"}


@pytest.mark.asyncio
async def test_handle_closing_early_releases_running_tools() -> None:
    released: list[str] = []

    @tool
    async def slow() -> str:
        try:
            await asyncio.sleep(10)

        except asyncio.CancelledError:
            released.append("slow")
            raise

        return "unreachable"

    @tool
    async def fast() -> str:
        return "fast"

    async with ctx.scope("test"):
        stream = Toolbox.of(slow, fast).handle([_request("1", "fast"), _request("2", "slow")])
        async for chunk in stream:
            assert isinstance(chunk, ModelToolResponse)
            assert chunk.identifier == "1"
            break

        await stream.aclose()

    assert released == ["slow"]


@pytest.mark.asyncio
async def test_calling_unknown_tool_raises_tool_exception() -> None:
    from draive.tools import ToolException

    async with ctx.scope("test"):
        with pytest.raises(ToolException) as exc_info:
            await Toolbox.empty.call("missing")

    assert exc_info.value.tool == "missing"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("suggesting", "iteration", "expected"),
    [
        (True, 0, "required"),
        (True, 1, "auto"),
        (2, 1, "required"),
        (2, 2, "auto"),
        (0, 0, "auto"),
        (False, 0, "auto"),
        (None, 0, "auto"),
    ],
)
async def test_suggestion_selection_by_iteration(
    suggesting: bool | int | None,
    iteration: int,
    expected: str,
) -> None:
    async with ctx.scope("test"):
        tools: ModelTools = Toolbox.of(_add, suggesting=suggesting).model_tools(iteration=iteration)

    assert tools.selection == expected


@pytest.mark.asyncio
async def test_suggested_tool_missing_from_toolbox_falls_back_to_auto() -> None:
    @tool
    async def other() -> str:
        return "other"

    toolbox = Toolbox.of(_add).with_suggestion(other)
    async with ctx.scope("test"):
        assert toolbox.model_tools(iteration=0).selection == "auto"


def test_toolbox_does_not_alias_the_provided_tools_mapping() -> None:
    tools: dict[str, Any] = {"_add": _add}
    toolbox = Toolbox(tools=tools, suggesting=Toolbox.empty.suggesting)
    tools["other"] = _add

    assert list(toolbox.tools) == ["_add"]


def test_later_duplicate_tools_replace_earlier_ones() -> None:
    @tool(name="_add")
    async def replacement(a: int) -> str:
        return "replaced"

    assert Toolbox.of(_add, replacement).tools["_add"] is replacement


def test_filtered_with_a_single_string_matches_exact_names_only() -> None:
    @tool
    async def calc() -> str:
        return "calc"

    @tool
    async def cal() -> str:
        return "cal"

    filtered = Toolbox.of(calc, cal).filtered(tools="calc")

    assert list(filtered.tools) == ["calc"]


@pytest.mark.asyncio
async def test_responses_follow_request_order_with_seeded_random_delays() -> None:
    import random

    rng = random.Random(7)
    delays: list[float] = [rng.random() * 0.01 for _ in range(60)]

    @tool
    async def delayed(index: int) -> str:
        await asyncio.sleep(delays[index])
        return str(index)

    order: list[int] = list(range(60))
    rng.shuffle(order)
    requests = [_request(f"r{index}", "delayed", index=index) for index in order]
    async with ctx.scope("test"):
        responses = await _responses(Toolbox.of(delayed), requests)

    assert [response.identifier for response in responses] == [f"r{index}" for index in order]
