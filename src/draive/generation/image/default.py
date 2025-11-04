from typing import Any

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
    result: ModelOutput = await GenerativeModel.completion(
        instructions=instructions,
        context=[ModelInput.of(input)],
        output="image",
        stream=False,
        **extra,
    )

    for image in result.content.images():
        return image

    raise ValueError("Failed to generate a valid image")
