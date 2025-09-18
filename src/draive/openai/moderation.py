from typing import Any

from haiway import ObservabilityLevel, ctx, not_missing
from openai.types import ModerationCreateResponse, ModerationMultiModalInputParam

from draive.guardrails import GuardrailsModeration, GuardrailsModerationException
from draive.multimodal import ArtifactContent, Multimodal, MultimodalContent, TextContent
from draive.openai.api import OpenAIAPI
from draive.openai.config import OpenAIModerationConfig
from draive.resources import ResourceContent, ResourceReference

__all__ = ("OpenAIContentModeration",)


class OpenAIContentModeration(OpenAIAPI):
    def moderation_guardrails(self) -> GuardrailsModeration:
        return GuardrailsModeration(
            input_checking=self.content_moderation,
            output_checking=self.content_moderation,
        )

    async def content_moderation(  # noqa: C901, PLR0912, PLR0915
        self,
        content: Multimodal,
        /,
        *,
        config: OpenAIModerationConfig | None = None,
        **extra: Any,
    ) -> None:
        moderation_config: OpenAIModerationConfig = config or ctx.state(OpenAIModerationConfig)
        async with ctx.scope("moderation"):
            ctx.record(
                ObservabilityLevel.INFO,
                attributes={
                    "guardrails.provider": "openai",
                    "guardrails.model": moderation_config.model,
                },
            )
            content = MultimodalContent.of(content)
            moderated_content: list[ModerationMultiModalInputParam] = []
            for part in content.parts:
                match part:
                    case TextContent() as text:
                        moderated_content.append(
                            {
                                "type": "text",
                                "text": text.text,
                            }
                        )

                    case ResourceContent() as media:
                        if media.mime_type.startswith("image"):
                            moderated_content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": media.to_data_uri(),
                                    },
                                }
                            )
                        else:
                            ctx.log_warning(
                                f"OpenAI moderation: unsupported media {media.mime_type}; "
                                "verifying as text."
                            )
                            moderated_content.append(
                                {
                                    "type": "text",
                                    "text": media.to_str(include_data=False),
                                }
                            )

                    case ResourceReference() as media_ref:
                        if media_ref.mime_type and media_ref.mime_type.startswith("image"):
                            moderated_content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": media_ref.uri},
                                }
                            )
                        else:
                            ctx.log_warning(
                                "OpenAI moderation: unsupported resource reference;"
                                " verifying as text."
                            )
                            moderated_content.append(
                                {"type": "text", "text": media_ref.uri},
                            )

                    case ArtifactContent() as artifact:
                        if artifact.hidden:
                            continue  # skip hidden

                        moderated_content.append(
                            {
                                "type": "text",
                                "text": artifact.artifact.to_str(),
                            }
                        )

                    case other:
                        ctx.log_warning(
                            f"Attempting to use moderation on unsupported content ({type(other)}),"
                            " verifying as text..."
                        )
                        moderated_content.append(
                            {
                                "type": "text",
                                "text": other.to_str(),
                            }
                        )

            response: ModerationCreateResponse = await self._client.moderations.create(
                model=moderation_config.model,
                input=moderated_content,
            )

            violations: dict[str, float] = {}
            for result in response.results:
                if (
                    not_missing(moderation_config.harassment_threshold)
                    and result.category_scores.harassment >= moderation_config.harassment_threshold
                ):
                    violations["harassment"] = result.category_scores.harassment

                elif result.categories.harassment:
                    violations["harassment"] = result.category_scores.harassment

                if (
                    not_missing(moderation_config.harassment_threatening_threshold)
                    and result.category_scores.harassment_threatening
                    >= moderation_config.harassment_threatening_threshold
                ):
                    violations["harassment_threatening"] = (
                        result.category_scores.harassment_threatening
                    )

                elif result.categories.harassment_threatening:
                    violations["harassment_threatening"] = (
                        result.category_scores.harassment_threatening
                    )

                if (
                    not_missing(moderation_config.hate_threshold)
                    and result.category_scores.hate >= moderation_config.hate_threshold
                ):
                    violations["hate"] = result.category_scores.hate

                elif result.categories.hate:
                    violations["hate"] = result.category_scores.hate

                if (
                    not_missing(moderation_config.hate_threatening_threshold)
                    and result.category_scores.hate_threatening
                    >= moderation_config.hate_threatening_threshold
                ):
                    violations["hate_threatening"] = result.category_scores.hate_threatening

                elif result.categories.hate_threatening:
                    violations["hate_threatening"] = result.category_scores.hate_threatening

                if (
                    not_missing(moderation_config.self_harm_threshold)
                    and result.category_scores.self_harm >= moderation_config.self_harm_threshold
                ):
                    violations["self_harm"] = result.category_scores.self_harm

                elif result.categories.self_harm:
                    violations["self_harm"] = result.category_scores.self_harm

                if (
                    not_missing(moderation_config.self_harm_instructions_threshold)
                    and result.category_scores.self_harm_instructions
                    >= moderation_config.self_harm_instructions_threshold
                ):
                    violations["self_harm_instructions"] = (
                        result.category_scores.self_harm_instructions
                    )

                elif result.categories.self_harm_instructions:
                    violations["self_harm_instructions"] = (
                        result.category_scores.self_harm_instructions
                    )

                if (
                    not_missing(moderation_config.self_harm_intent_threshold)
                    and result.category_scores.self_harm_intent
                    >= moderation_config.self_harm_intent_threshold
                ):
                    violations["self_harm_intent"] = result.category_scores.self_harm_intent

                elif result.categories.self_harm_intent:
                    violations["self_harm_intent"] = result.category_scores.self_harm_intent

                if (
                    not_missing(moderation_config.sexual_threshold)
                    and result.category_scores.sexual >= moderation_config.sexual_threshold
                ):
                    violations["sexual"] = result.category_scores.sexual

                elif result.categories.sexual:
                    violations["sexual"] = result.category_scores.sexual

                if (
                    not_missing(moderation_config.sexual_minors_threshold)
                    and result.category_scores.sexual_minors
                    >= moderation_config.sexual_minors_threshold
                ):
                    violations["sexual_minors"] = result.category_scores.sexual_minors

                elif result.categories.sexual_minors:
                    violations["sexual_minors"] = result.category_scores.sexual_minors

                if (
                    not_missing(moderation_config.violence_threshold)
                    and result.category_scores.violence >= moderation_config.violence_threshold
                ):
                    violations["violence"] = result.category_scores.violence

                elif result.categories.violence:
                    violations["violence"] = result.category_scores.violence

                if (
                    not_missing(moderation_config.violence_graphic_threshold)
                    and result.category_scores.violence_graphic
                    >= moderation_config.violence_graphic_threshold
                ):
                    violations["violence_graphic"] = result.category_scores.violence_graphic

                elif result.categories.violence_graphic:
                    violations["violence_graphic"] = result.category_scores.violence_graphic

                if (
                    not_missing(moderation_config.illicit_threshold)
                    and result.category_scores.illicit >= moderation_config.illicit_threshold
                ):
                    violations["illicit"] = result.category_scores.illicit

                elif result.categories.illicit:
                    violations["illicit"] = result.category_scores.illicit

                if (
                    not_missing(moderation_config.illicit_violent_threshold)
                    and result.category_scores.illicit_violent
                    >= moderation_config.illicit_violent_threshold
                ):
                    violations["illicit_violent"] = result.category_scores.illicit_violent

                elif result.categories.illicit_violent:
                    violations["illicit_violent"] = result.category_scores.illicit_violent

            if violations:
                raise GuardrailsModerationException(
                    f"Content violated rule(s): {violations}",
                    violations=violations,
                    content=content,
                )
