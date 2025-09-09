from typing import Any

from haiway import ctx

from draive.models import (
    GenerativeModel,
    InstructionsRepository,
    ModelInput,
    ModelOutput,
    ResolveableInstructions,
)
from draive.multimodal import MultimodalContent
from draive.resources import ResourceContent, ResourceReference

__all__ = ("generate_audio",)


async def generate_audio(
    *,
    instructions: ResolveableInstructions,
    input: MultimodalContent,  # noqa: A002
    **extra: Any,
) -> ResourceContent | ResourceReference:
    async with ctx.scope("generate_audio"):
        result: ModelOutput = await GenerativeModel.completion(
            instructions=await InstructionsRepository.resolve(instructions),
            context=[ModelInput.of(input)],
            output="audio",
            **extra,
        )

        for audio in result.content.audio():
            return audio

        raise ValueError("Failed to generate a valid audio")
