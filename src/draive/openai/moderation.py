from collections.abc import Generator, Sequence
from typing import Any, Final

from haiway import Missing, ctx, not_missing
from openai.types import ModerationCreateResponse, ModerationMultiModalInputParam

from draive.guardrails import GuardrailsModerationException
from draive.models import record_guardrails_invocation
from draive.multimodal import ArtifactContent, Multimodal, MultimodalContent, TextContent
from draive.openai.api import OpenAIAPI
from draive.openai.config import OpenAIModerationConfig
from draive.resources import ResourceContent, ResourceReference

__all__ = ("OpenAIContentModeration",)

# categories reported by the moderation models, each one pairing a flag within
# `categories` with a score within `category_scores` and a `<name>_threshold`
# configuration field
_MODERATION_CATEGORIES: Final[Sequence[str]] = (
    "harassment",
    "harassment_threatening",
    "hate",
    "hate_threatening",
    "illicit",
    "illicit_violent",
    "self_harm",
    "self_harm_instructions",
    "self_harm_intent",
    "sexual",
    "sexual_minors",
    "violence",
    "violence_graphic",
)


class OpenAIContentModeration(OpenAIAPI):
    async def content_moderation(
        self,
        content: Multimodal,
        /,
        *,
        config: OpenAIModerationConfig | None = None,
        **extra: Any,
    ) -> None:
        moderation_config: OpenAIModerationConfig = config or ctx.state(OpenAIModerationConfig)
        async with ctx.scope("guardrails.invocation"):
            record_guardrails_invocation(
                provider="openai",
                model=moderation_config.model,
            )
            content = MultimodalContent.of(content)
            moderated_content: list[ModerationMultiModalInputParam] = list(
                _moderated_parts(content)
            )

            response: ModerationCreateResponse = await self._client.moderations.create(
                model=moderation_config.model,
                input=moderated_content,
            )

            violations: dict[str, float] = {}
            for result in response.results:
                for category in _MODERATION_CATEGORIES:
                    score: float = getattr(result.category_scores, category)
                    threshold: float | Missing = getattr(
                        moderation_config,
                        f"{category}_threshold",
                    )
                    # a configured threshold replaces the model provided category flag,
                    # the flag is used only when no threshold was configured
                    if not_missing(threshold):
                        if score >= threshold:
                            violations[category] = score

                    elif getattr(result.categories, category):
                        violations[category] = score

            if violations:
                raise GuardrailsModerationException(
                    f"Content violated rule(s): {violations}",
                    violations=violations,
                    content=content,
                )


def _moderated_parts(
    content: MultimodalContent,
    /,
) -> Generator[ModerationMultiModalInputParam]:
    # the endpoint accepts text and images only, anything else is verified as its
    # textual representation instead of being dropped
    for part in content.parts:
        if isinstance(part, TextContent):
            yield {
                "type": "text",
                "text": part.text,
            }

        elif isinstance(part, ResourceContent):
            if part.mime_type.startswith("image"):
                yield {
                    "type": "image_url",
                    "image_url": {"url": part.to_data_uri()},
                }

            else:
                ctx.log_warning(
                    f"OpenAI moderation: unsupported media {part.mime_type}; verifying as text."
                )
                yield {
                    "type": "text",
                    "text": part.to_str(include_data=False),
                }

        elif isinstance(part, ResourceReference):
            if part.mime_type.startswith("image"):
                yield {
                    "type": "image_url",
                    "image_url": {"url": part.uri},
                }

            else:
                # the uri is omitted, it can carry credentials within its userinfo or query
                ctx.log_warning("OpenAI moderation: unsupported resource reference; skipping.")

        else:
            assert isinstance(part, ArtifactContent)  # nosec: B101
            if part.hidden:
                continue  # skip hidden

            yield {
                "type": "text",
                "text": part.to_str(),
            }
