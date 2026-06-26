import pytest

from draive.evaluators import relevance_evaluator


@pytest.mark.asyncio
async def test_relevance_evaluator_treats_whitespace_content_as_empty() -> None:
    result = await relevance_evaluator(
        "   ",
        reference="The ferry to the island departs every hour on the half hour.",
    )

    assert result.score == 0.0
    assert result.meta["comment"] == "Input was empty!"
