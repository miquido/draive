from types import SimpleNamespace
from typing import Any

import pytest
from haiway import ctx

from draive.guardrails import GuardrailsModerationException
from draive.mistral.config import MistralModerationConfig
from draive.mistral.moderation import MistralContentModeration
from draive.multimodal import MultimodalContent, TextContent
from draive.resources import ResourceReference


@pytest.mark.asyncio
async def test_mistral_moderation_thresholds_require_score_at_or_above_threshold() -> None:
    async def _moderate_async(**_: Any) -> Any:
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    categories={"violence": True},
                    category_scores={"violence": 0.1},
                )
            ]
        )

    model = object.__new__(MistralContentModeration)
    model._client = SimpleNamespace(
        classifiers=SimpleNamespace(
            moderate_async=_moderate_async,
        )
    )

    async with ctx.scope("test"):
        await model.content_verification(
            TextContent.of("test"),
            config=MistralModerationConfig(
                category_thresholds={"violence": 0.9},
            ),
        )


@pytest.mark.asyncio
async def test_mistral_moderation_thresholds_block_score_above_threshold() -> None:
    async def _moderate_async(**_: Any) -> Any:
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    categories={"violence": True},
                    category_scores={"violence": 0.8},
                )
            ]
        )

    model = object.__new__(MistralContentModeration)
    model._client = SimpleNamespace(
        classifiers=SimpleNamespace(
            moderate_async=_moderate_async,
        )
    )

    async with ctx.scope("test"):
        with pytest.raises(GuardrailsModerationException):
            await model.content_verification(
                TextContent.of("test"),
                config=MistralModerationConfig(
                    category_thresholds={"violence": 0.2},
                ),
            )


@pytest.mark.asyncio
async def test_mistral_moderation_does_not_send_resource_reference_uri() -> None:
    sent: list[Any] = []

    async def _moderate_async(**kwargs: Any) -> Any:
        sent.append(kwargs["inputs"])
        return SimpleNamespace(results=[])

    model = object.__new__(MistralContentModeration)
    model._client = SimpleNamespace(
        classifiers=SimpleNamespace(
            moderate_async=_moderate_async,
        )
    )

    async with ctx.scope("test"):
        await model.content_verification(
            MultimodalContent.of(
                TextContent.of("test"),
                ResourceReference.of(
                    "https://user:secret@example.com/file.txt?token=secret",
                    mime_type="text/plain",
                ),
            ),
            config=MistralModerationConfig(),
        )

    assert sent == [["test"]]
