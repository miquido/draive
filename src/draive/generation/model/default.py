from collections.abc import Iterable
from typing import Any

from haiway import State

from draive.generation.model.types import ModelGenerationDecoder
from draive.models import (
    ModelInput,
    ModelInstructions,
    ModelOutput,
)
from draive.multimodal import ArtifactContent, Multimodal, MultimodalContent
from draive.steps import Step
from draive.steps.decoding import decode_content
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
        output=generated,
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

    return decode_content(
        completion,
        generated,
        decoder=decoder,
    )
