import pytest

from draive.evaluators import consistency_evaluator


@pytest.mark.asyncio
async def test_consistency_evaluator_treats_whitespace_content_as_empty() -> None:
    result = await consistency_evaluator(
        "   ",
        reference="The ferry to the island departs every hour on the half hour.",
    )

    assert result.score == 0.0
    assert result.meta["comment"] == "Input was empty!"


@pytest.mark.asyncio
async def test_consistency_evaluator_treats_whitespace_reference_as_empty() -> None:
    result = await consistency_evaluator(
        "The ferry departs every hour.",
        reference="   ",
    )

    assert result.score == 0.0
    assert result.meta["comment"] == "Reference was empty!"
