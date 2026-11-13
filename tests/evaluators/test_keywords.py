import pytest

from draive.evaluators import forbidden_keywords_evaluator, required_keywords_evaluator


@pytest.mark.asyncio
async def test_required_keywords_scores_matching_share() -> None:
    result = await required_keywords_evaluator(
        "The quick brown fox",
        keywords=["quick", "slow"],
        require_all=False,
    )

    assert result.score == 0.5


@pytest.mark.asyncio
async def test_forbidden_keywords_detects_keyword() -> None:
    result = await forbidden_keywords_evaluator(
        "contains secret value",
        keywords=["secret"],
    )

    assert result.score == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("evaluator", (required_keywords_evaluator, forbidden_keywords_evaluator))
async def test_keyword_evaluators_reject_blank_keywords(evaluator) -> None:
    result = await evaluator("some text", keywords=["valid", " "])

    assert result.score == 0.0
    assert "blank" in str(result.meta["error"])
