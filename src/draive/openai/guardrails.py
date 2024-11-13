from asyncio import gather
from typing import Any

from haiway import ctx, not_missing
from openai.types.moderation import Moderation

from draive.multimodal import MultimodalContent
from draive.openai.client import OpenAIClient
from draive.openai.config import OpenAIModerationConfig
from draive.safeguards import ContentGuardrails, GuardrailsException
from draive.splitters import split_text

__all__ = [
    "openai_content_guardrails",
]


async def openai_content_guardrails(  # noqa: C901, PLR0915
    client: OpenAIClient | None = None,
    /,
) -> ContentGuardrails:
    client = client or OpenAIClient.shared()

    async def openai_content_guardrails(  # noqa: C901, PLR0912
        content: MultimodalContent,
        /,
        **extra: Any,
    ) -> MultimodalContent:
        config: OpenAIModerationConfig = ctx.state(OpenAIModerationConfig).updated(**extra)

        content_parts: list[str] = split_text(
            content.as_string(),
            part_size=2000,
            part_overlap_size=200,
            count_size=len,
        )
        results: list[Moderation] = await gather(
            *[client.moderation_check(text=part) for part in content_parts]
        )

        violations: set[str] = set()
        for result in results:
            if (
                not_missing(config.harassment_threshold)
                and result.category_scores.harassment >= config.harassment_threshold
            ):
                violations.add("harassment")

            elif result.categories.harassment:
                violations.add("harassment")

            if (
                not_missing(config.harassment_threatening_threshold)
                and result.category_scores.harassment_threatening
                >= config.harassment_threatening_threshold
            ):
                violations.add("harassment_threatening")

            elif result.categories.harassment_threatening:
                violations.add("harassment_threatening")

            if (
                not_missing(config.hate_threshold)
                and result.category_scores.hate >= config.hate_threshold
            ):
                violations.add("hate")

            elif result.categories.hate:
                violations.add("hate")

            if (
                not_missing(config.hate_threatening_threshold)
                and result.category_scores.hate_threatening >= config.hate_threatening_threshold
            ):
                violations.add("hate_threatening")

            elif result.categories.hate_threatening:
                violations.add("hate_threatening")

            if (
                not_missing(config.self_harm_threshold)
                and result.category_scores.self_harm >= config.self_harm_threshold
            ):
                violations.add("self_harm")

            elif result.categories.self_harm:
                violations.add("self_harm")

            if (
                not_missing(config.self_harm_instructions_threshold)
                and result.category_scores.self_harm_instructions
                >= config.self_harm_instructions_threshold
            ):
                violations.add("self_harm_instructions")

            elif result.categories.self_harm_instructions:
                violations.add("self_harm_instructions")

            if (
                not_missing(config.self_harm_intent_threshold)
                and result.category_scores.self_harm_intent >= config.self_harm_intent_threshold
            ):
                violations.add("self_harm_intent")

            elif result.categories.self_harm_intent:
                violations.add("self_harm_intent")

            if (
                not_missing(config.sexual_threshold)
                and result.category_scores.sexual >= config.sexual_threshold
            ):
                violations.add("sexual")

            elif result.categories.sexual:
                violations.add("sexual")

            if (
                not_missing(config.sexual_minors_threshold)
                and result.category_scores.sexual_minors >= config.sexual_minors_threshold
            ):
                violations.add("sexual_minors")

            elif result.categories.sexual_minors:
                violations.add("sexual_minors")

            if (
                not_missing(config.violence_threshold)
                and result.category_scores.violence >= config.violence_threshold
            ):
                violations.add("violence")

            elif result.categories.violence:
                violations.add("violence")

            if (
                not_missing(config.violence_graphic_threshold)
                and result.category_scores.violence_graphic >= config.violence_graphic_threshold
            ):
                violations.add("violence_graphic")

            elif result.categories.violence_graphic:
                violations.add("violence_graphic")

        if violations:
            raise GuardrailsException(f"Content violated rule(s): {violations}")

        return content  # no violation - use content

    return openai_content_guardrails
