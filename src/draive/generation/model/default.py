from collections.abc import Iterable
from typing import Any

from haiway import ctx

from draive.generation.model.types import ModelGenerationDecoder
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
    toolbox: Toolbox,
    examples: Iterable[tuple[MultimodalContent, Generated]],
    decoder: ModelGenerationDecoder[Generated] | None,
    **extra: Any,
) -> Generated:
    result: ModelOutput = await GenerativeModel.loop(
        instructions=instructions,
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
            ModelInput.of(input),
        ],
        output="auto" if decoder is not None else generated,
        stream=False,
        **extra,
    )

    try:
        if decoder is not None:
            ctx.log_debug("...decoding result...")
            return decoder(result.content)

        elif artifacts := result.content.artifacts(generated):
            ctx.log_debug("...direct artifact found!")
            return artifacts[0].artifact

        else:  # fallback to default decoding
            ctx.log_debug("...decoding result...")
            return generated.from_json(result.content.to_str())

    except Exception as exc:
        ctx.log_error(
            f"Failed to decode {generated.__name__} model due to an error: {type(exc)}",
            exception=exc,
        )
        raise exc
