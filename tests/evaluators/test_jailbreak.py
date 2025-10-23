import pytest

from draive.evaluators import jailbreak_evaluator


@pytest.mark.asyncio
async def test_jailbreak_evaluator_returns_empty_score_for_missing_content() -> None:
    result = await jailbreak_evaluator("")

    assert result.score == 0.0
    assert result.meta["comment"] == "Input was empty!"
