from types import SimpleNamespace
from typing import Any

import pytest
from haiway import ctx

from draive.guardrails import GuardrailsModerationException
from draive.multimodal import TextContent
from draive.openai.config import OpenAIModerationConfig
from draive.openai.moderation import OpenAIContentModeration

_CATEGORIES = (
    "harassment",
    "harassment_threatening",
    "hate",
    "hate_threatening",
    "self_harm",
    "self_harm_instructions",
    "self_harm_intent",
    "sexual",
    "sexual_minors",
    "violence",
    "violence_graphic",
    "illicit",
    "illicit_violent",
)


def _moderation(
    *,
    flagged: str,
    score: float,
) -> OpenAIContentModeration:
    async def create(**_: Any) -> Any:
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    categories=SimpleNamespace(
                        **{category: category == flagged for category in _CATEGORIES}
                    ),
                    category_scores=SimpleNamespace(
                        **{
                            category: score if category == flagged else 0.0
                            for category in _CATEGORIES
                        }
                    ),
                )
            ]
        )

    model = object.__new__(OpenAIContentModeration)
    model._client = SimpleNamespace(moderations=SimpleNamespace(create=create))
    return model


@pytest.mark.asyncio
async def test_configured_threshold_suppresses_the_model_flag() -> None:
    # a configured threshold replaces the model provided flag instead of only adding to it
    model = _moderation(flagged="violence", score=0.1)

    async with ctx.scope("test"):
        await model.content_moderation(
            TextContent.of("test"),
            config=OpenAIModerationConfig(violence_threshold=0.9),
        )


@pytest.mark.asyncio
async def test_configured_threshold_reports_score_above_it() -> None:
    model = _moderation(flagged="violence", score=0.8)

    async with ctx.scope("test"):
        with pytest.raises(GuardrailsModerationException) as exception:
            await model.content_moderation(
                TextContent.of("test"),
                config=OpenAIModerationConfig(violence_threshold=0.2),
            )

    assert exception.value.violations == {"violence": 0.8}


@pytest.mark.asyncio
async def test_model_flag_is_used_without_a_threshold() -> None:
    model = _moderation(flagged="violence", score=0.1)

    async with ctx.scope("test"):
        with pytest.raises(GuardrailsModerationException):
            await model.content_moderation(
                TextContent.of("test"),
                config=OpenAIModerationConfig(),
            )


@pytest.mark.asyncio
async def test_threshold_of_another_category_does_not_suppress_the_flag() -> None:
    model = _moderation(flagged="violence", score=0.1)

    async with ctx.scope("test"):
        with pytest.raises(GuardrailsModerationException):
            await model.content_moderation(
                TextContent.of("test"),
                config=OpenAIModerationConfig(hate_threshold=0.9),
            )
