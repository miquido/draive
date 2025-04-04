from asyncio import gather
from itertools import chain
from typing import Any

from haiway import ctx, not_missing
from openai.types.moderation_create_response import ModerationCreateResponse

from draive.multimodal import MultimodalContent
from draive.openai.api import OpenAIAPI
from draive.openai.config import OpenAIModerationConfig
from draive.openai.utils import unwrap_missing
from draive.safeguards import ContentGuardrails, GuardrailsException
from draive.splitters import split_text

__all__ = ("OpenAIContentFiltering",)


class OpenAIContentFiltering(OpenAIAPI):
    async def content_guardrails(self) -> ContentGuardrails:
        # TODO: refine guardrails
        return self.filter_content

    async def filter_content(  # noqa: C901, PLR0912
        self,
        content: MultimodalContent,
        /,
        *,
        config: OpenAIModerationConfig | None = None,
        **extra: Any,
    ) -> MultimodalContent:
        moderation_config: OpenAIModerationConfig = config or ctx.state(
            OpenAIModerationConfig
        ).updated(**extra)

        # TODO: add multimodal support
        content_parts: list[str] = split_text(
            content.as_string(),
            part_size=2000,
            part_overlap_size=200,
            count_size=len,
        )

        results: list[ModerationCreateResponse] = await gather(
            *[
                self._client.moderations.create(
                    model=moderation_config.model,
                    input=part,
                    timeout=unwrap_missing(moderation_config.timeout),
                )
                for part in content_parts
            ]
        )

        violations: set[str] = set()
        for result in chain.from_iterable([result.results for result in results]):
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

        if violations:
            raise GuardrailsException(f"Content violated rule(s): {violations}")

        return content  # no violation - use content
