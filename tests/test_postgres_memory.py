from typing import Any

import pytest

from draive.models import ModelInput, ModelOutput
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
