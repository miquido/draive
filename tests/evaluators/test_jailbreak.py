import pytest

from draive.evaluators import jailbreak_evaluator


@pytest.mark.asyncio
async def test_jailbreak_evaluator_treats_empty_content_as_safe() -> None:
    result = await jailbreak_evaluator("")

    assert result.score == 1.0
    assert result.meta["comment"] == "Input was empty - no jailbreak content."
