from typing import Any

from haiway import ctx, not_missing
from openai.types import ModerationCreateResponse, ModerationMultiModalInputParam

from draive.guardrails import GuardrailsModeration, GuardrailsModerationException
from draive.multimodal import ArtifactContent, Multimodal, MultimodalContent, TextContent
from draive.openai.api import OpenAIAPI
from draive.openai.config import OpenAIModerationConfig
from draive.resources import ResourceContent, ResourceReference

__all__ = ("OpenAIContentModeration",)


class OpenAIContentModeration(OpenAIAPI):
    def moderation_guardrails(self) -> GuardrailsModeration:
        return GuardrailsModeration(input_checking=self.content_moderation)

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
            ctx.record_info(
                attributes={
                    "guardrails.provider": "openai",
                    "guardrails.model": moderation_config.model,
                },
            )
            content = MultimodalContent.of(content)
            moderated_content: list[ModerationMultiModalInputParam] = []
            for part in content.parts:
                if isinstance(part, TextContent):
                    moderated_content.append(
                        {
                            "type": "text",
                            "text": part.text,
                        }
                    )

                elif isinstance(part, ResourceContent):
                    if part.mime_type.startswith("image"):
                        moderated_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": part.to_data_uri(),
                                },
                            }
                        )
                    else:
                        ctx.log_warning(
                            f"OpenAI moderation: unsupported media {part.mime_type}; "
                            "verifying as text."
                        )
                        moderated_content.append(
                            {
                                "type": "text",
                                "text": part.to_str(include_data=False),
                            }
                        )

                elif isinstance(part, ResourceReference):
                    if part.mime_type and part.mime_type.startswith("image"):
                        moderated_content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": part.uri},
                            }
                        )
                    else:
                        ctx.log_warning(
                            "OpenAI moderation: unsupported resource reference; verifying as text."
                        )
                        moderated_content.append(
                            {"type": "text", "text": part.uri},
                        )

                else:
                    assert isinstance(part, ArtifactContent)  # nosec: B101
                    if part.hidden:
                        continue  # skip hidden

                    moderated_content.append(
                        {
                            "type": "text",
                            "text": part.artifact.to_str(),
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
