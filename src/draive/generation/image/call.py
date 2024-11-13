from typing import Any

from haiway import ctx

from draive.generation.image.state import ImageGeneration
from draive.instructions import Instruction
from draive.multimodal import MediaContent, Multimodal

__all__ = [
    "generate_image",
]


async def generate_image(
    *,
    instruction: Instruction | str,
    input: Multimodal | None = None,  # noqa: A002
    **extra: Any,
) -> MediaContent:
    return await ctx.state(ImageGeneration).generate(
        instruction=instruction,
        input=input,
        **extra,
    )
