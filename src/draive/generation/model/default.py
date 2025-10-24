from collections.abc import Iterable
from typing import Any, Literal, cast

from haiway import concurrently, ctx

from draive.generation.model.types import ModelGenerationDecoder
from draive.guardrails import GuardrailsModeration, GuardrailsQualityVerification, GuardrailsSafety
from draive.models import (
    GenerativeModel,
    ModelInput,
    ModelInstructions,
    ModelOutput,
    Toolbox,
)
from draive.multimodal import MultimodalContent
from draive.parameters import DataModel

__all__ = ("generate_model",)


async def generate_model[Generated: DataModel](
    generated: type[Generated],
    /,
    *,
    instructions: ModelInstructions,
    input: MultimodalContent,  # noqa: A002
    schema_injection: Literal["full", "simplified", "skip"],
    toolbox: Toolbox,
    examples: Iterable[tuple[MultimodalContent, Generated]],
    decoder: ModelGenerationDecoder[Generated] | None,
    **extra: Any,
) -> Generated:
    async with ctx.scope("generate_model"):
        resolved_instructions: ModelInstructions
        match schema_injection:
            case "full":
                resolved_instructions = instructions.format(
                    model_schema=generated.json_schema(indent=2),
                )

            case "simplified":
                resolved_instructions = instructions.format(
                    model_schema=generated.simplified_schema(indent=2),
                )

            case "skip":  # instruction is not modified
                resolved_instructions = instructions

        # run input moderation in parallel - TODO: should we use sanitized input?
        result: ModelOutput = await GuardrailsModeration.input_guarded(
            input,
            GenerativeModel.loop(
                instructions=resolved_instructions,
                toolbox=toolbox,
                context=[
                    *[
                        message
                        for example in examples
                        for message in [
                            ModelInput.of(example[0]),
                            ModelOutput.of(MultimodalContent.of(example[1].to_json(indent=2))),
                        ]
                    ],
                    # sanitize all inputs based on safety guardrails
                    ModelInput.of(await GuardrailsSafety.sanitize(input)),
                ],
                output="auto" if decoder is not None else generated,
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

        try:
            if decoder is not None:
                ctx.log_debug("...decoding result...")
                return decoder(result.content)

            elif artifacts := result.content.artifacts(generated):
                ctx.log_debug("...direct artifact found!")
                return cast(Generated, artifacts[0].artifact)

            else:  # fallback to default decoding
                ctx.log_debug("...decoding result...")
                return generated.from_json(result.content.to_str())

        except Exception as exc:
            ctx.log_error(
                f"Failed to decode {generated.__name__} model due to an error: {type(exc)}",
                exception=exc,
            )
            raise exc
