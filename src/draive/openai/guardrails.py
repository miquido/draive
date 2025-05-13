from typing import Any

from haiway import ObservabilityLevel, ctx, not_missing
from openai.types import ModerationCreateResponse, ModerationMultiModalInputParam

from draive.guardrails import GuardrailsModeration, GuardrailsModerationException
from draive.multimodal import MediaData, MediaReference, Multimodal, MultimodalContent, TextContent
from draive.openai.api import OpenAIAPI
from draive.openai.config import OpenAIModerationConfig
from draive.openai.utils import unwrap_missing

__all__ = ("OpenAIContentModereation",)


class OpenAIContentModereation(OpenAIAPI):
    def content_guardrails(self) -> GuardrailsModeration:
        return GuardrailsModeration(input_checking=self.content_verification)

    async def content_verification(  # noqa: C901, PLR0912, PLR0915
        self,
        content: Multimodal,
        /,
        *,
        config: OpenAIModerationConfig | None = None,
        **extra: Any,
    ) -> None:
        moderation_config: OpenAIModerationConfig = config or ctx.state(OpenAIModerationConfig)
        with ctx.scope("openai_moderation", moderation_config):
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

                    case MediaData() as media_data:
                        moderated_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": media_data.to_data_uri(safe_encoding=False),
                                },
                            }
                        )

                    case MediaReference() as media_referencve:
                        moderated_content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": media_referencve.uri},
                            }
                        )

                    case other:
                        ctx.log_warning(
                            f"Attempting to use moderation on unsupported content ({type(other)}),"
                            " verifying as json text..."
                        )
                        moderated_content.append(
                            {
                                "type": "text",
                                "text": other.to_json(),
                            }
                        )

            response: ModerationCreateResponse = await self._client.moderations.create(
                model=moderation_config.model,
                input=moderated_content,
                timeout=unwrap_missing(moderation_config.timeout),
            )

            violations: set[str] = set()
            for result in response.results:
                if (
                    not_missing(moderation_config.harassment_threshold)
                    and result.category_scores.harassment >= moderation_config.harassment_threshold
                ):
                    violations.add("harassment")

                elif result.categories.harassment:
                    violations.add("harassment")

                if (
                    not_missing(moderation_config.harassment_threatening_threshold)
                    and result.category_scores.harassment_threatening
                    >= moderation_config.harassment_threatening_threshold
                ):
                    violations.add("harassment_threatening")

                elif result.categories.harassment_threatening:
                    violations.add("harassment_threatening")

                if (
                    not_missing(moderation_config.hate_threshold)
                    and result.category_scores.hate >= moderation_config.hate_threshold
                ):
                    violations.add("hate")

                elif result.categories.hate:
                    violations.add("hate")

                if (
                    not_missing(moderation_config.hate_threatening_threshold)
                    and result.category_scores.hate_threatening
                    >= moderation_config.hate_threatening_threshold
                ):
                    violations.add("hate_threatening")

                elif result.categories.hate_threatening:
                    violations.add("hate_threatening")

                if (
                    not_missing(moderation_config.self_harm_threshold)
                    and result.category_scores.self_harm >= moderation_config.self_harm_threshold
                ):
                    violations.add("self_harm")

                elif result.categories.self_harm:
                    violations.add("self_harm")

                if (
                    not_missing(moderation_config.self_harm_instructions_threshold)
                    and result.category_scores.self_harm_instructions
                    >= moderation_config.self_harm_instructions_threshold
                ):
                    violations.add("self_harm_instructions")

                elif result.categories.self_harm_instructions:
                    violations.add("self_harm_instructions")

                if (
                    not_missing(moderation_config.self_harm_intent_threshold)
                    and result.category_scores.self_harm_intent
                    >= moderation_config.self_harm_intent_threshold
                ):
                    violations.add("self_harm_intent")

                elif result.categories.self_harm_intent:
                    violations.add("self_harm_intent")

                if (
                    not_missing(moderation_config.sexual_threshold)
                    and result.category_scores.sexual >= moderation_config.sexual_threshold
                ):
                    violations.add("sexual")

                elif result.categories.sexual:
                    violations.add("sexual")

                if (
                    not_missing(moderation_config.sexual_minors_threshold)
                    and result.category_scores.sexual_minors
                    >= moderation_config.sexual_minors_threshold
                ):
                    violations.add("sexual_minors")

                elif result.categories.sexual_minors:
                    violations.add("sexual_minors")

                if (
                    not_missing(moderation_config.violence_threshold)
                    and result.category_scores.violence >= moderation_config.violence_threshold
                ):
                    violations.add("violence")

                elif result.categories.violence:
                    violations.add("violence")

                if (
                    not_missing(moderation_config.violence_graphic_threshold)
                    and result.category_scores.violence_graphic
                    >= moderation_config.violence_graphic_threshold
                ):
                    violations.add("violence_graphic")

                elif result.categories.violence_graphic:
                    violations.add("violence_graphic")

                if (
                    not_missing(moderation_config.illicit_threshold)
                    and result.category_scores.illicit >= moderation_config.illicit_threshold
                ):
                    violations.add("illicit")

                elif result.categories.illicit:
                    violations.add("illicit")

                if (
                    not_missing(moderation_config.illicit_violent_threshold)
                    and result.category_scores.illicit_violent
                    >= moderation_config.illicit_violent_threshold
                ):
                    violations.add("illicit_violent")

                elif result.categories.illicit_violent:
                    violations.add("illicit_violent")

            if violations:
                raise GuardrailsModerationException(
                    f"Content violated rule(s): {violations}",
                    violations=tuple(violations),
                    content=content,
                )
