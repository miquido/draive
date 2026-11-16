import pytest
from haiway import ctx

from draive.evaluators.keywords import forbidden_keywords_evaluator, required_keywords_evaluator


@pytest.mark.asyncio
async def test_required_keywords_match_case_insensitively_with_casefold() -> None:
    async with ctx.scope("test"):
        result = await required_keywords_evaluator("Wir lesen die STRASSE", keywords=["straße"])

    assert result.score == 1.0


@pytest.mark.asyncio
async def test_forbidden_keywords_match_case_insensitively_with_casefold() -> None:
    async with ctx.scope("test"):
        result = await forbidden_keywords_evaluator("Die STRASSE", keywords=["straße"])

    assert result.score == 0.0
