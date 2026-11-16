from typing import Any

from draive.generation.types import GenerationOutputMissing
from draive.models import (
    ModelInput,
    ModelInstructions,
)
from draive.multimodal import Multimodal, MultimodalContent
from draive.resources import ResourceContent, ResourceReference
from draive.steps import Step

__all__ = ("generate_audio",)


async def generate_audio(
    *,
    instructions: ModelInstructions,
    input: Multimodal,  # noqa: A002
    **extra: Any,
) -> ResourceContent | ResourceReference:
    completion: MultimodalContent = await Step.generating_completion(
        instructions=instructions,
        output="audio",
        **extra,
    ).run((ModelInput.of(MultimodalContent.of(input)),))

    if audio := completion.audio():
        return audio[-1]

    raise GenerationOutputMissing("Generation completed without a valid audio output")
