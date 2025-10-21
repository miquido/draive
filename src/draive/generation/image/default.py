from typing import Any

from haiway import ctx

from draive.models import (
    GenerativeModel,
    ModelInput,
    ModelInstructions,
    ModelOutput,
)
from draive.multimodal import MultimodalContent, Template, TemplatesRepository
from draive.resources import ResourceContent, ResourceReference

__all__ = ("generate_image",)


async def generate_image(
    *,
    instructions: Template | ModelInstructions,
    input: MultimodalContent,  # noqa: A002
    **extra: Any,
) -> ResourceContent | ResourceReference:
    async with ctx.scope("generate_image"):
        result: ModelOutput = await GenerativeModel.completion(
            instructions=await TemplatesRepository.resolve_str(instructions),
            context=[ModelInput.of(input)],
            output="image",
            **extra,
        )

        for image in result.content.images():
            return image

        raise ValueError("Failed to generate a valid image")
