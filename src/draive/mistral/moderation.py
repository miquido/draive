from collections.abc import Mapping
from typing import Any, cast

from haiway import MISSING, ctx
from mistralai import ModerationResponse

from draive.guardrails import GuardrailsModeration, GuardrailsModerationException
from draive.mistral.api import MistralAPI
from draive.mistral.config import MistralModerationConfig
from draive.multimodal import Multimodal, MultimodalContent
from draive.multimodal.artifact import ArtifactContent
from draive.multimodal.text import TextContent
from draive.resources.types import ResourceContent, ResourceReference

__all__ = ("MistralContentModeration",)


class MistralContentModeration(MistralAPI):
    def content_guardrails(self) -> GuardrailsModeration:
        return GuardrailsModeration(input_checking=self.content_verification)

    async def content_verification(  # noqa: C901, PLR0912
        self,
        content: Multimodal,
        /,
        config: MistralModerationConfig | None = None,
        **extra: Any,
    ) -> None:
        moderation_config: MistralModerationConfig = config or ctx.state(MistralModerationConfig)
        async with ctx.scope("moderation"):
            ctx.record_info(
                attributes={
                    "guardrails.provider": "mistral",
                    "guardrails.model": moderation_config.model,
                },
            )
            content = MultimodalContent.of(content)
            moderated_content: list[str] = []
            for part in content.parts:
                if isinstance(part, TextContent):
                    moderated_content.append(part.text)

                elif isinstance(part, ResourceContent):
                    ctx.log_warning(
                        f"Mistral moderation: unsupported media {part.mime_type}; "
                        "verifying as text."
                    )
                    moderated_content.append(part.to_str(include_data=False))

                elif isinstance(part, ResourceReference):
                    ctx.log_warning(
                        f"Mistral moderation: unsupported media {part.mime_type}; "
                        "verifying as text."
                    )
                    moderated_content.append(part.to_str())

                else:
                    assert isinstance(part, ArtifactContent)  # nosec: B101
                    if part.hidden:
                        continue  # skip hidden

                    moderated_content.append(part.artifact.to_str())

            response: ModerationResponse = await self._client.classifiers.moderate_async(
                model=moderation_config.model,
                inputs=moderated_content,
            )

            violations: dict[str, float] = {}
            for result in response.results:
                if result.categories is None or result.category_scores is None:
                    continue  # skip empty, it should be always present though

                if moderation_config.category_thresholds is MISSING:
                    for category, flagged in result.categories.items():
                        if flagged:
                            violations[category] = result.category_scores.get(category, 1.0)

                        # skip not flagged

                else:
                    category_thresholds: Mapping[str, float] = cast(
                        Mapping[str, float], moderation_config.category_thresholds
                    )
                    for category, flagged in result.categories.items():
                        if category in category_thresholds:
                            score: float = result.category_scores.get(
                                category, 1.0 if flagged else 0.0
                            )
                            if category_thresholds[category] >= score:
                                violations[category] = score

                        elif flagged:
                            violations[category] = result.category_scores.get(category, 1.0)

                        # skip not flagged

            if violations:
                raise GuardrailsModerationException(
                    f"Content violated rule(s): {violations}",
                    violations=violations,
                    content=content,
                )
