from typing import Any

from haiway import ctx

from draive.models import (
    GenerativeModel,
    InstructionsRepository,
    ModelInput,
    ModelOutput,
    ResolveableInstructions,
)
from draive.multimodal import MediaContent, MultimodalContent

__all__ = ("generate_image",)


async def generate_image(
    *,
    instructions: ResolveableInstructions,
    input: MultimodalContent,  # noqa: A002
    **extra: Any,
) -> MediaContent:
    async with ctx.scope("generate_image"):
        result: ModelOutput = await GenerativeModel.completion(
            instructions=await InstructionsRepository.resolve(instructions),
            context=[ModelInput.of(input)],
            output="image",
            **extra,
        )

        for image in result.content.media("image"):
            return image

        raise ValueError("Failed to generate a valid image")
