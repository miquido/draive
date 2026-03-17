from typing import Any

from draive.models import (
    ModelInput,
    ModelInstructions,
)
from draive.multimodal import Multimodal, MultimodalContent
from draive.resources import ResourceContent, ResourceReference
from draive.steps import Step

__all__ = ("generate_image",)


async def generate_image(
    *,
    instructions: ModelInstructions,
    input: Multimodal,  # noqa: A002
    **extra: Any,
) -> ResourceContent | ResourceReference:
    completion: MultimodalContent = await Step.generating_completion(
        instructions=instructions,
        output="image",
        **extra,
    ).run((ModelInput.of(MultimodalContent.of(input)),))

    for image in completion.images():
        return image  # TODO: consider resource content chunks requiring content merge

    raise ValueError("Failed to generate a valid image")
