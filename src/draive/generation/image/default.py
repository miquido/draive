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

__all__ = ("generate_image",)


async def generate_image(
    *,
    instructions: ModelInstructions,
    input: MultimodalContent,  # noqa: A002
    **extra: Any,
) -> ResourceContent | ResourceReference:
    async with ctx.scope("generate_image"):
        # run input moderation in parallel - TODO: should we use sanitized input?
        result: ModelOutput = await GuardrailsModeration.input_guarded(
            input,
            GenerativeModel.completion(
                instructions=instructions,
                # sanitize all inputs based on safety guardrails
                context=[
                    # sanitize all inputs based on safety guardrails
                    ModelInput.of(await GuardrailsSafety.sanitize(input))
                ],
                output="image",
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

        for image in result.content.images():
            return image

        raise ValueError("Failed to generate a valid image")
