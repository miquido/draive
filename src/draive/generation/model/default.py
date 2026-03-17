from collections.abc import Iterable
from typing import Any

from haiway import State, ctx

from draive.generation.model.types import ModelGenerationDecoder
from draive.models import (
    ModelInput,
    ModelInstructions,
    ModelOutput,
)
from draive.multimodal import ArtifactContent, Multimodal, MultimodalContent
from draive.steps import Step
from draive.tools import Toolbox

__all__ = ("generate_model",)


async def generate_model[Generated: State](
    generated: type[Generated],
    /,
    *,
    instructions: ModelInstructions,
    input: Multimodal,  # noqa: A002
    toolbox: Toolbox,
    examples: Iterable[tuple[Multimodal, Generated]],
    decoder: ModelGenerationDecoder[Generated] | None,
    **extra: Any,
) -> Generated:
    completion: MultimodalContent = await Step.looping_completion(
        instructions=instructions,
        tools=toolbox,
        output="json" if decoder is not None else generated,
        **extra,
    ).run(
        (
            *(
                message
                for example in examples
                for message in (
                    ModelInput.of(MultimodalContent.of(example[0])),
                    ModelOutput.of(MultimodalContent.of(ArtifactContent.of(example[1]))),
                )
            ),
            ModelInput.of(MultimodalContent.of(input)),
        )
    )

    try:
        if decoder is not None:
            ctx.log_debug("...decoding result...")
            return decoder(completion)

        elif artifacts := completion.artifacts(category="json"):
            ctx.log_debug("...direct artifact found!")
            return artifacts[0].to_state(generated)

        else:  # fallback to default decoding
            ctx.log_debug("...decoding result...")
            return generated.from_json(completion.to_str())

    except Exception as exc:
        ctx.log_error(
            f"Failed to decode {generated.__name__} model due to an error: {type(exc)}",
            exception=exc,
        )
        raise exc
