from collections.abc import Iterable
from typing import Any

from haiway import concurrently, ctx

from draive.guardrails import GuardrailsModeration, GuardrailsQualityVerification, GuardrailsSafety
from draive.models import (
    GenerativeModel,
    ModelInput,
    ModelInstructions,
    ModelOutput,
    Toolbox,
)
from draive.multimodal import MultimodalContent

__all__ = ("generate_text",)


async def generate_text(
    *,
    instructions: ModelInstructions,
    input: MultimodalContent,  # noqa: A002
    toolbox: Toolbox,
    examples: Iterable[tuple[MultimodalContent, str]],
    **extra: Any,
) -> str:
    async with ctx.scope("generate_text"):
        # run input moderation in parallel - TODO: should we use sanitized input?
        result: ModelOutput = await GuardrailsModeration.input_guarded(
            input,
            GenerativeModel.loop(
                instructions=instructions,
                context=[
                    *[
                        message
                        for example in examples
                        for message in [
                            ModelInput.of(example[0]),
                            ModelOutput.of(MultimodalContent.of(example[1])),
                        ]
                    ],
                    # sanitize all inputs based on safety guardrails
                    ModelInput.of(await GuardrailsSafety.sanitize(input)),
                ],
                toolbox=toolbox,
                output="text",
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

        return result.content.to_str()
