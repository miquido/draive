from typing import Any

import pytest

from draive.models import ModelInput, ModelOutput, ModelToolRequest, ModelToolResponse
from draive.multimodal import MultimodalContent, TextContent
from draive.postgres import memory as postgres_memory


@pytest.mark.asyncio
async def test_load_context_returns_chronological_order(monkeypatch: pytest.MonkeyPatch) -> None:
    expected_elements = (
        ModelInput.of(MultimodalContent.of(TextContent.of("first"))),
        ModelOutput.of(MultimodalContent.of(TextContent.of("second"))),
        ModelInput.of(MultimodalContent.of(TextContent.of("third"))),
    )
    descending_rows = tuple(
        {"content": element.to_json()} for element in reversed(expected_elements)
    )

    async def fake_fetch(
        statement: str,
        *parameters: Any,
    ) -> tuple[dict[str, str], ...]:
        return descending_rows

    monkeypatch.setattr(postgres_memory.Postgres, "fetch", fake_fetch)

    recalled = await postgres_memory._load_context(identifier="demo-session", limit=None)

    assert recalled == expected_elements


@pytest.mark.asyncio
async def test_load_context_filters_tool_usage_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    user_turn = ModelInput.of(MultimodalContent.of(TextContent.of("user")))
    tool_request = ModelOutput.of(
        ModelToolRequest.of(
            "tool-1",
            tool="search",
            arguments={"query": "state of tool"},
        )
    )
    tool_response = ModelInput.of(
        ModelToolResponse.of(
            "tool-1",
            tool="search",
            content=MultimodalContent.of(TextContent.of("result")),
        )
    )
    final_turn = ModelOutput.of(MultimodalContent.of(TextContent.of("answer")))

    chronological_elements = (user_turn, tool_request, tool_response, final_turn)
    descending_rows = tuple(
        {"content": element.to_json()} for element in reversed(chronological_elements)
    )

    async def fake_fetch(
        statement: str,
        *parameters: Any,
    ) -> tuple[dict[str, str], ...]:
        return descending_rows

    monkeypatch.setattr(postgres_memory.Postgres, "fetch", fake_fetch)

    recalled = await postgres_memory._load_context(identifier="demo-session", limit=None)

    assert recalled == (
        user_turn,
        tool_request,
        tool_response,
        final_turn,
    )


@pytest.mark.asyncio
async def test_load_context_drops_orphan_tool_response(monkeypatch: pytest.MonkeyPatch) -> None:
    tool_response = ModelInput.of(
        ModelToolResponse.of(
            "tool-1",
            tool="search",
            content=MultimodalContent.of(TextContent.of("result")),
        )
    )
    final_turn = ModelOutput.of(MultimodalContent.of(TextContent.of("answer")))

    descending_rows = (
        {"content": final_turn.to_json()},
        {"content": tool_response.to_json()},
    )

    async def fake_fetch(
        statement: str,
        *parameters: Any,
    ) -> tuple[dict[str, str], ...]:
        assert parameters[-1] == 2
        return descending_rows

    monkeypatch.setattr(postgres_memory.Postgres, "fetch", fake_fetch)

    recalled = await postgres_memory._load_context(identifier="demo-session", limit=2)

    assert recalled == ()


@pytest.mark.asyncio
async def test_load_context_drops_leading_tool_only_input(monkeypatch: pytest.MonkeyPatch) -> None:
    tool_response = ModelInput.of(
        ModelToolResponse.of(
            "tool-1",
            tool="search",
            content=MultimodalContent.of(TextContent.of("result")),
        )
    )
    user_turn = ModelInput.of(MultimodalContent.of(TextContent.of("user")))
    assistant_turn = ModelOutput.of(MultimodalContent.of(TextContent.of("done")))

    chronological_elements = (tool_response, user_turn, assistant_turn)
    descending_rows = tuple(
        {"content": element.to_json()} for element in reversed(chronological_elements)
    )

    async def fake_fetch(
        statement: str,
        *parameters: Any,
    ) -> tuple[dict[str, str], ...]:
        return descending_rows

    monkeypatch.setattr(postgres_memory.Postgres, "fetch", fake_fetch)

    recalled = await postgres_memory._load_context(identifier="demo-session", limit=None)

    assert recalled == (user_turn, assistant_turn)


@pytest.mark.asyncio
async def test_load_context_drops_leading_tool_request(monkeypatch: pytest.MonkeyPatch) -> None:
    tool_request = ModelOutput.of(
        ModelToolRequest.of(
            "tool-1",
            tool="search",
            arguments={"query": "state of tool"},
        )
    )
    assistant_turn = ModelOutput.of(MultimodalContent.of(TextContent.of("final")))

    descending_rows = (
        {"content": assistant_turn.to_json()},
        {"content": tool_request.to_json()},
    )

    async def fake_fetch(
        statement: str,
        *parameters: Any,
    ) -> tuple[dict[str, str], ...]:
        assert parameters[-1] == 2
        return descending_rows

    monkeypatch.setattr(postgres_memory.Postgres, "fetch", fake_fetch)

    recalled = await postgres_memory._load_context(identifier="demo-session", limit=2)

    assert recalled == ()


@pytest.mark.asyncio
async def test_load_context_enforces_leading_input(monkeypatch: pytest.MonkeyPatch) -> None:
    older_output = ModelOutput.of(MultimodalContent.of(TextContent.of("old-output")))
    latest_input = ModelInput.of(MultimodalContent.of(TextContent.of("latest-input")))

    descending_rows = (
        {"content": latest_input.to_json()},
        {"content": older_output.to_json()},
    )

    async def fake_fetch(
        statement: str,
        *parameters: Any,
    ) -> tuple[dict[str, str], ...]:
        assert parameters[-1] == 2
        return descending_rows

    monkeypatch.setattr(postgres_memory.Postgres, "fetch", fake_fetch)

    recalled = await postgres_memory._load_context(identifier="demo-session", limit=2)

    assert recalled == (latest_input,)


@pytest.mark.asyncio
async def test_load_context_drops_unmatched_single_input(monkeypatch: pytest.MonkeyPatch) -> None:
    user_turn = ModelInput.of(MultimodalContent.of(TextContent.of("user")))

    descending_rows = ({"content": user_turn.to_json()},)

    async def fake_fetch(
        statement: str,
        *parameters: Any,
    ) -> tuple[dict[str, str], ...]:
        return descending_rows

    monkeypatch.setattr(postgres_memory.Postgres, "fetch", fake_fetch)

    recalled = await postgres_memory._load_context(identifier="demo-session", limit=1)

    assert recalled == (user_turn,)


@pytest.mark.asyncio
async def test_load_context_drops_stale_input_before_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    stale_input = ModelInput.of(MultimodalContent.of(TextContent.of("stale")))
    latest_input = ModelInput.of(MultimodalContent.of(TextContent.of("latest")))
    assistant_turn = ModelOutput.of(MultimodalContent.of(TextContent.of("answer")))

    descending_rows = (
        {"content": assistant_turn.to_json()},
        {"content": latest_input.to_json()},
        {"content": stale_input.to_json()},
    )

    async def fake_fetch(
        statement: str,
        *parameters: Any,
    ) -> tuple[dict[str, str], ...]:
        return descending_rows

    monkeypatch.setattr(postgres_memory.Postgres, "fetch", fake_fetch)

    recalled = await postgres_memory._load_context(identifier="demo-session", limit=None)

    assert recalled == (stale_input, latest_input, assistant_turn)


@pytest.mark.asyncio
async def test_load_context_removes_input_with_content_and_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_with_tool = ModelInput.of(
        MultimodalContent.of(TextContent.of("user with data")),
        ModelToolResponse.of(
            "tool-0",
            tool="lookup",
            content=MultimodalContent.of(TextContent.of("embedded")),
        ),
    )
    assistant_turn = ModelOutput.of(MultimodalContent.of(TextContent.of("answer")))

    descending_rows = (
        {"content": assistant_turn.to_json()},
        {"content": user_with_tool.to_json()},
    )

    async def fake_fetch(
        statement: str,
        *parameters: Any,
    ) -> tuple[dict[str, str], ...]:
        return descending_rows

    monkeypatch.setattr(postgres_memory.Postgres, "fetch", fake_fetch)

    recalled = await postgres_memory._load_context(identifier="demo-session", limit=None)

    assert recalled == ()
