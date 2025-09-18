from collections.abc import Mapping
from typing import Any, cast

from haiway import MISSING, ObservabilityLevel, ctx
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
        return GuardrailsModeration(
            input_checking=self.content_verification,
            output_checking=self.content_verification,
        )

    async def content_verification(  # noqa: C901, PLR0912
        self,
        content: Multimodal,
        /,
        config: MistralModerationConfig | None = None,
        **extra: Any,
    ) -> None:
        moderation_config: MistralModerationConfig = config or ctx.state(MistralModerationConfig)
        async with ctx.scope("moderation"):
            ctx.record(
                ObservabilityLevel.INFO,
                attributes={
                    "guardrails.provider": "mistral",
                    "guardrails.model": moderation_config.model,
                },
            )
            content = MultimodalContent.of(content)
            moderated_content: list[str] = []
            for part in content.parts:
                match part:
                    case TextContent() as text:
                        moderated_content.append(text.text)

                    case ResourceContent() as media:
                        ctx.log_warning(
                            f"Mistral moderation: unsupported media {media.mime_type}; "
                            "verifying as text."
                        )
                        moderated_content.append(media.to_str(include_data=False))

                    case ResourceReference() as media_ref:
                        ctx.log_warning(
                            f"Mistral moderation: unsupported media {media_ref.mime_type}; "
                            "verifying as text."
                        )
                        moderated_content.append(media_ref.to_str())

                    case ArtifactContent() as artifact:
                        if artifact.hidden:
                            continue  # skip hidden

                        moderated_content.append(artifact.artifact.to_str())

                    case other:
                        ctx.log_warning(
                            f"Attempting to use moderation on unsupported content ({type(other)}),"
                            " verifying as text..."
                        )
                        moderated_content.append(other.to_str())

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
