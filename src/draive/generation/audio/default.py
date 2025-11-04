from typing import Any

from draive.models import (
    GenerativeModel,
    ModelInput,
    ModelInstructions,
    ModelOutput,
)
from draive.multimodal import MultimodalContent
from draive.resources import ResourceContent, ResourceReference

__all__ = ("generate_audio",)


async def generate_audio(
    *,
    instructions: ModelInstructions,
    input: MultimodalContent,  # noqa: A002
    **extra: Any,
) -> ResourceContent | ResourceReference:
    result: ModelOutput = await GenerativeModel.completion(
        instructions=instructions,
        context=[ModelInput.of(input)],
        output="audio",
        stream=False,
        **extra,
    )

    for audio in result.content.audio():
        return audio

    raise ValueError("Failed to generate a valid audio")
