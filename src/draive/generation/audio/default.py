from typing import Any

from haiway import concurrently, ctx

from draive.guardrails import GuardrailsModeration, GuardrailsQualityVerification, GuardrailsSafety
from draive.models import (
    GenerativeModel,
    ModelInput,
    ModelInstructions,
    ModelOutput,
)
from draive.multimodal import MultimodalContent
from draive.resources import ResourceContent, ResourceReference

__all__ = ("generate_audio",)


async def generate_audio(
    *,
    instructions: ModelInstructions,
    input: MultimodalContent,  # noqa: A002
    **extra: Any,
) -> ResourceContent | ResourceReference:
    async with ctx.scope("generate_audio"):
        # run input moderation in parallel - TODO: should we use sanitized input?
        result: ModelOutput = await GuardrailsModeration.input_guarded(
            input,
            GenerativeModel.completion(
                instructions=instructions,
                context=[
                    # sanitize all inputs based on safety guardrails
                    ModelInput.of(await GuardrailsSafety.sanitize(input))
                ],
                output="audio",
                **extra,
            ),
        )

        # verify all outputs based on quality and moderation guardrails
        await concurrently(
            (
                GuardrailsModeration.check_output(result.content),
                GuardrailsQualityVerification.verify(result.content),
            ),
            concurrent_tasks=2,
        )

        for audio in result.content.audio():
            return audio

        raise ValueError("Failed to generate a valid audio")
